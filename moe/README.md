# Continual Mixture of Response Experts

同一个模型预测 Emission 和 Uptake，分别对应自然甲烷排放与 upland 土壤甲烷交换。Environmental JEPA 把每个点或格点区域的日、月 Query 编为 128 维向量。`ContinualMethaneMoE` 对同一向量计算共享基础预测 \(b(z)\in\mathbb R^2\)、各专家的残差 \(r_k(z)\in\mathbb R^2\)，以及每个任务单独的路由

\[
\pi_{q,k}(z)=\operatorname{softmax}_k\!\left(\frac{g_{q,k}(z)-\beta_q d_{k,q}(z)}{\tau}\right),
\qquad \hat y_q=b_q(z)+\sum_k\pi_{q,k}(z)r_{k,q}(z).
\]

通量为正表示向大气释放，为负表示吸收。Chamber 的正通量表示该土壤在观测窗口内为净源。四个初始专家各用一个共享隐藏表示的 MLP 和两个有符号线性输出头。两个任务使用参数独立的 Dense Router，以及共享的 Expert Bank。路由根据任务的 Signature 距离进行可选调整。

默认温度固定为 1.0。通过 `moe.routing_temperature` 设定初始温度，通过 `training.common.temperature_schedule` 或阶段覆盖配置 `{kind: linear, end: 0.5, steps: 10000}` 开启线性退火。进度按成功的 optimizer updates 计算。默认的 `{kind: fixed}` 维持初始温度。`balance_weight` 可开启低使用率惩罚，默认权重为零。

## JEPA 权重与 Query

- `encoder.checkpoint: auto` 从 `encoding/model_save/*/` 查找与验证 `jepa_loss` 最佳轮次相符、且保存于该轮的导出权重。本仓库找到 `encoding/model_save/soil_5km_trial_4/encoder.pt`。同目录 `last.pt` 记录 `epoch=4` 和完整验证历史。`train.log` 中第 4 轮 `jepa_loss=0.0142367887`，第 3 轮为 `0.0156018262`。含方差惩罚的总 `loss` 则在第 3 轮较低（`0.0382674098` 对 `0.0386355824`）。
- 加载时逐张量对照导出权重与 `last.pt` 中的 `online_encoder.*`，构造 `EnvironmentalEncoder`，并保存 checkpoint 路径、SHA-256、模型配置、驱动表、采样配置和 Query / preprocessing 版本。显式路径可指向 Encoder 导出或完整 JEPA checkpoint。若有意使用无法验证轮次的独立导出，将 `expected_best_epoch` 设为 `null`，并提供用于读取采样参数的 `encoder.jepa_config`。
- Online 模式复用 `QuerySpec`、`MultiResolutionGatherer` 和 `EnvironmentalCollator`。时间区间为 UTC 左闭右开。默认读取 Query 起点以前的 14 天历史。使用当期环境驱动估计通量时可配置 `query_window`，此时覆盖整个日或月。切换模式会改变模型输入的时间信息，训练与部署应保持一致。
- 当前产品默认使用 `encoder.spatial_selection: intersects`，读取与矩形有正面积相交的原生像元，每轴完整宽度最多 2°。从原第 4 轮 JEPA 切换到该策略时使用 5,120 个 tokens 的产品预算，可通过 `encoder.max_tokens` 调整。显式指定 `centers` 可重放原采样策略。空间规则和实际采样策略参与缓存校验，原版本的 embedding 缓存需重新生成。
- MoE 的所有训练和候选适配均使用同一份冻结 Encoder。它全程处于 eval 模式。Online 和 Cache 构建均用 FP32 生成 latent，MoE 本身可用 FP32、BF16 或 FP16。独立 JEPA 训练入口位于 [encoding](../encoding/README.md)。

## Expert Signature

每个专家和任务保存 \(S_{k,q}=(\mu_{k,q},\sigma^2_{k,q},\rho_{k,q})\)。中心和逐维方差通过有效样本的路由权重更新。用于路由的距离为

\[
d_{k,q}(z)=\frac{1}{D}\sum_d\frac{(z_d-\mu_{k,q,d})^2}{\sigma^2_{k,q,d}+\epsilon}.
\]

尚无统计量时保持可学习路由。Response Fingerprint 是专家在任务固定 anchor 集上的残差值序列。Anchor 从真实训练 batch 的 latent 选择，随 checkpoint 保存并记录来源和版本，可用于比较响应函数。

## 归一化、Loss 与验证

按任务从固定 Training 分区计算总体均值与标准差。全量模拟直接合并 Task 1 扫描位图时逐 tile 累计的精确矩，观测使用固定的 Training 标签。统计使用每条有效标签一次，检查均值有限和标准差大于 `1e-8`。

| 全量模拟 Training | 样本数 | Mean | Std |
| --- | ---: | ---: | ---: |
| Emission | 694,352,596 | 131.356484 | 324.594291 |
| Uptake | 27,773,015 | −0.595501 | 2.062101 |

模型和 Expert Matching 使用归一化输出 `(y - mean_task) / std_task`。验证和 `model.eval().predict_physical(latent)` 使用 `y_norm * std_task + mean_task`。默认 Loss 是归一化 MSE，Huber 可显式选择。每个 batch 先按有效 mask 与样本权重计算每个任务 × 来源的风险，再乘上 `task_weights` 与 `source_weights` 求和。两个任务的默认权重均为 1.0。TEM 的日排放和月氧化分别计损失。Replay、蒸馏、专家匹配和候选比较沿用同一个标签尺度。

验证按任务、来源 × 任务、产品报告物理单位 RMSE、MAE、R² 和归一化 MSE。每条样本等权的指标之外，观测验证还给出站点等权的 `site_equal` 指标，各站点先在站内平均。模型选择使用配置指定的 `weighted_normalized_mse`，即对应权重方案下的任务 × 来源归一化 MSE 加权和。训练目标中启用的 Huber、蒸馏、Replay、均衡和物理正则项各自记录。

源校准和物理约束默认权重为零。可选源校准按来源和任务在归一化空间计算仿射变换。符号约束用物理零点，惩罚按任务标准差缩放。启用 `physical_response_weight` 时，在线 batch 须包含可信的 `physical_response[task]["reference"]` 和可选 `weights`，形状与 `environment["values"]` 的 `[B, token, driver]` 一致。参考值单位为通量率除以驱动的原始物理单位。输入导数经过冻结 JEPA，再乘任务标准差。

## 训练阶段

1. **Simulation Pretraining** 同时训练共享预测、Router 和专家。每条 TEM / MeMo 标签用原生支持的 Query 预测通量率，全量训练维持 Task 1 的 tile 和 batch 顺序规则。
2. **Observation Fine-tuning** 用一个阶段训练 MoE。提供模拟 checkpoint 时继承其 Training 归一化。直接用观测启动时，从初始观测 Training 计算归一化。模拟和历史观测 Replay 各自配置 `training.finetune.replay.simulation` 与 `replay.observation` 的 `weight`、`sources` 和可选 cache 参数，默认关闭。用已有观测 checkpoint 微调新数据时，指定新的 `data.observation_cohort`，历史观测 Training 由该 checkpoint 的清单继承。独立的 `distillation_weight` 默认是零，启用后使用阶段入口快照在当前 Training Query 上蒸馏。Teacher 与自己的拓扑一起随 checkpoint 保存。
3. **Continual Growth** 继续继承归一化。在训练窗口估计专家距离与归一化响应损失，两项阈值持续超出时提出候选。候选只更新复制的专家和新增路由，可启用任一历史 Training Replay 和入口 Teacher。用当前 Validation 估计新数据改善，并用 `growth_validation_sources` 指定历史 Validation 估计遗忘。默认历史来源为 FLUXNET / Chamber。样本下限、遗忘上限、增长收益与接受流程由现有 Growth Manager 处理。

拆分时，两份相同分支在温度缩放后各减去 \(\log 2\)，平分原路由概率。候选内部可按配置退火，验收时双方在正式模型当前温度下比较。距离与损失阈值应按当前归一化尺度重新标定。接受的专家结构、ID、Signatures 和历史都随 checkpoint 保存。

## 本地数据与监督单位

`data/methane_data` 链接本地数据。统一物理标签单位为 `mg_ch4_m2_day`（mg CH₄ m⁻² d⁻¹）。Query 保留标签的日或月时间支持，面积分母仍按源产品定义。

| 产品和相对路径 | 本地字段与原始质量基准 | 时间和面积支持及处理 |
| --- | --- | --- |
| TEM `simulations/TEM_EMISSION/CH4_emission_intensity_{year}.nc` | `CH4_emission(latitude, longitude, month, day)`，360 × 720 × 12 × 31。数据提供者已确认以 mg CH₄ 计，文件缺少单位属性。 | 40 个年度文件，1979–2018，0.5° 日格点。原值已经是日速率。跳过公历无效日期，过滤全部负值与无效值，保留零。文件未注明面积分母是整格、湿地面积还是其他面积。 |
| TEM `simulations/TEM_ch4_wetland_soilsink_1950_2020_mgm2day.nc4` | `ch4_oxidation(month, year, longitude, latitude)`，12 × 71 × 720 × 360。文件标注 `mg/m2/day`，CH₄ 质量基准按本项目确认。 | 文件覆盖 1950–2020，正式模拟选取 1979–2020。0.5° 整月标签已经是日速率。扫描所选年份确认负值占多数，保留原始符号和少量正通量。面积分母及正值的过程来源细分未记录。 |
| MeMo `simulations/MeMo_1990-2009_monthly_corrected.nc` | `CH4uptake(Time, Lat, Lon)`，240 × 180 × 360。文件明确标注 `mgCH4/m2/month`，`Time` 从 1 开始对应 1990-01。 | 1990–2009，1°。正吸收量除以当月公历天数后取负，包含闰年二月。负的原始吸收量按无效值过滤。文件未注明面积分母的土地覆盖口径。 |
| FLUXNET `observations/FLUXNET-CH4/FLUXNET_T1_DD.csv` 与 `FLUXNET_CH4_2024.csv` | 日表 `FCH4_F_ANNOPTLM` 为 gap-filled 净生态系统交换，量纲为 nmol CH₄ m⁻² s⁻¹。以规范化后的 `Site` 和 `SITE_ID` 连接站点元数据。 | 涡度协方差通量按站点通量足迹解释，当前 CSV 无逐条足迹面积或本地时区字段。默认 `WET`、有限且非负，从原始单位一次乘以 1.386072。可选月模式按合格日速率取均值，默认至少 20 天。 |
| Chamber `observations/chamber_processed.csv` | 末列 `FCH4_monthly_processed`，日模式使用 `FCH4_daily_mean`。按项目确认使用 mg CH₄ m⁻² d⁻¹，CSV 缺少单位属性。 | 默认以给定月速率监督整月 Query。同站点、研究、坐标和月份的重复行只计一次。当前表的站点均标为 upland，保留正、负和零。`nonpositive_uptake: true` 可过滤正值。具体 chamber 面积和逐条月内采样覆盖未记录。 |

默认假设湿地 FLUXNET 的非负净交换能提供 emission 代理监督。通量包含净源汇共同作用，无法从这列直接分离 gross emission。当前本地日表有 `FCH4_F` 对照列，缺少正式 QC 标志和 ANN 填补不确定度。数据审计统计可用的 QC 列、缺测及筛选计数，维持现有 gap-filled 标签使用方式。[FLUXNET-CH₄ 数据发布说明](https://essd.copernicus.org/articles/13/3607/2021/#appendix-B)

Chamber 月模式直接使用先前研究确认的 `FCH4_monthly_processed`，有效零与正月值均进入 uptake。当前 14,484 行中 7,565 行是同站点、研究、坐标和月份的重复行，得到 6,919 条月标签，其中 5,987 条为负、932 条为零，缺测和其他适配器过滤均为 0。旧 Site Split 另去除 2 条跨集合副本，得到 6,917 条。当前站内时间划分去除 83 条跨集合副本，得到 6,836 条。被移除的月值在留出集合中均有相同 Query 和数值的副本。

观测 CSV 以名义日期的 UTC 日界生成 Query。默认使用点 Query，可按产品配置 `spatial_support`。可选 `quality_filters` 支持 `allowed`、`min`、`max`，`sample_weight_column` 可传入已有权重。

## TEM 网格中心

TEM 两个产品都缺少显式坐标边界属性。日排放纬度递增，范围为 −90° 至 89.5°。完整覆盖两极对应的是南边界，中心纬度为原纬度加半格。月氧化纬度递减，范围为 90° 至 −89.5°。其纬度为北边界，中心纬度为原纬度减半格。两个产品的经度均从 −180° 到 179.5°，作为西边界加半格后归一到 [−180°, 180°)。偏移按实际间距计算，原始标签索引保持对应，支持窗口仍为 0.5° × 0.5°。MeMo 已采用格心。预测或制图可复用 `GridProduct.lat`、`GridProduct.lon` 和 `GridProduct.support`。

## 全量模拟与固定划分

- 正式配置 `simulation_mode: full` 使用 1979 年起全部可用的合法网格标签。默认 `simulation_split_mode: temporal` 按产品将末段约 20% 的整年划给 Validation，前段用于 Training。当前 TEM 日排放留出 2011–2018，TEM 月氧化留出 2013–2020，MeMo 留出 2006–2009。可用 `simulation_split_mode: spatial` 恢复全球共享 5° 空间块的 80/20 划分，同样遍历全部有效标签。模拟阶段只有 Training 和 Validation。
- 清单记录 native tile、有效标签数、位图偏移和 SHA-256。训练按需读取 tile，打乱 shard 和 shard 内的连续 128-batch 区间。每个完整 epoch 覆盖全部 Training 和 Validation。全量 batch size 8 时有 90,266,303 个训练 batch，其中 TEM 日排放 86,794,659 个，Uptake 3,471,644 个。每批属于同一产品，数据按原始频率进入 optimizer。
- `log_interval_steps`、`validation_interval_steps`、`checkpoint_interval_steps` 以成功的 optimizer updates 为单位。全量配置每 10,000 步用各产品固定的 16 个 Validation batch 选 `best.pt`。epoch 末报告完整 Validation，并用同一固定样本比较最佳模型、更新调度与早停。当 `validation_probe_batches_per_product: 0` 时以完整 Validation 选最佳。全量 Training、Validation 的 batch 游标与累计统计可以从 `last.pt` 恢复。
- 1979 年初的历史窗口只从 `simulation_daily_start: '1979-01-01'` 读取实际可用的日驱动。没有日历史时使用已有静态环境。之后缺驱动文件或没有任何环境 token 的有效标签会报错。
- `samples_per_simulation_source` 仅在 `simulation_mode: debug` 下决定候选上限。独立的 [moe_debug.yaml](configs/moe_debug.yaml) 用于小规模运行。

## 观测划分和 Replay

- `observation_split_mode: temporal` 在每一个 FLUXNET 站按日、每一个 Chamber 站按去重后的月对有效标签排序。同一个日期或月份的记录一起进入一组。有至少三个时段时站内按连续的约 60% / 20% / 20% 划分，按有效时段数取整并保证各集合都有时段。各站时间边界独立。默认 `observation_short_sites: train_only` 将一至两个有效时段的站点留在 Training。可选 `train_validation` 对两个时段各留一个训练和验证，或设为 `exclude`。可用 `observation_split_mode: spatial` 恢复按站点末次记录时间排序的 Site Split。
- 同一标签或精确 Query 的跨集合副本保留在最严格的留出集合。单站时间支持有重叠时数据审计报错。审计记录各集合站点数、重叠站点、最近大圆距离、共享空间块和研究编号。站内划分的相同站点最近距离可为零。
- `data.split_seed` 控制划分所需的随机性，`training.common.training_seed` 控制训练随机性。默认 v2 清单在 `moe/runs/splits-v2`。CSV 记录 SHA-256，模拟 NetCDF 记录路径、大小和修改时间。检查点记录清单 SHA-256，启动时验证历史清单、模拟位图及源文件版本。旧 v1 清单保留在原路径，v2 阶段使用新的划分与 checkpoint。
- 初始微调使用 `initial_observation.json`。新 cohort 设置 `training.continual.data.observation_cohort` 和数据范围。时间模式下同站新批次须晚于已冻结记录，空间模式下历史站点沿用原集合。两种模式都保护原 Testing 标签和 Query。Replay 监督来自冻结的历史 Training，历史 Validation 单独供 Growth 验收，Testing 保留在原清单。
- 离线审计按来源和划分报告原始量、有效量、正负零、缺测、过滤原因及分布。全量模拟另外记录各年精确计数、公历有效槽位、完整 pass 的 steps 和位图大小。模拟分位数使用小样本近似，其余统计和训练覆盖来自全量扫描。
- 已复核 TEM 月氧化的两份审计。旧 `native_signs` 是 1950–2020 全文件的 31,282,608 负、1,689 正、21,951,219 零。v2 覆盖 1979–2020，为 18,530,419 负、1,101 正、12,959,912 零。直接扫描原文件复现了旧计数，日期范围解释了差异。

## 配置和启动

在仓库根目录运行，需要项目的 `fm` 环境和 [encoding/requirements.txt](../encoding/requirements.txt) 中的依赖。

使用 `--prepare-data` 生成或核验清单并在标准输出查看统计。指定上阶段的 `--checkpoint` 时同时检查 Replay 来源和跨阶段交集。

~~~bash
conda activate fm
python -m moe.train --config moe/configs/moe.yaml --stage pretrain --prepare-data
python -m moe.train --config moe/configs/moe.yaml --stage finetune --prepare-data
python -m moe.train --config moe/configs/moe_debug.yaml --stage pretrain --prepare-data
~~~

切换划分或数据版本前，将相应 YAML 的 `data.split_manifest_dir` 设为新目录，再运行对应 `--prepare-data` 命令。已保存的清单和位图可用于继续之前的训练。

默认 [moe.yaml](configs/moe.yaml) 使用当前文件中的有效初始观测。可用 `training.finetune.data.observation_site_ids` 或 `observation_date_range` 为后续 cohort 留出范围，并为实验指定独立的 `data.split_manifest_dir`。Continual 运行需要唯一的 cohort ID 与新增观测范围。

~~~bash
python -m moe.train --config moe/configs/moe.yaml --stage pretrain
python -m moe.train --config moe/configs/moe.yaml --stage finetune \
  --checkpoint moe/runs/pretrain/best.pt
python -m moe.train --config moe/configs/moe.yaml --stage continual \
  --checkpoint moe/runs/finetune/best.pt
~~~

用 `--resume <run_dir>/last.pt` 从同一阶段已保存的游标继续，训练有效配置须匹配。`--checkpoint` 启动新阶段，同一 lineage 的 Encoder、输入语义和已有 MoE 拓扑必须匹配。不同阶段的训练设置按 common 再按该阶段合并。可用 `--output-dir` 指定实验目录。CUDA 支持 `mixed_precision: fp32`、`fp16` 或 `bf16`，调试配置默认 CPU FP32。

`effective_config.json` 与 checkpoint 记录合并后的设置、Encoder 输入清单、标签统计来源、实际启用的约束和固定 Validation 选择方案。`last.pt`、`best.pt` 保存模型拓扑、参数与 buffers、Python / NumPy / PyTorch 随机状态、optimizer、适用的 scheduler / scaler、固定 Teacher 和进度。旧 MoE v1 checkpoint 会明确报错。迁移时用原 JEPA 权重和 Task 1 的 v2 清单启动新的 MoE 训练。

### Embedding Cache

配置 `data.cache_path` 为目录，`data.cache_dtype` 选 `float32` 或 `float16`。构建与恢复使用同一个命令。

~~~bash
python -m moe.train --config moe/configs/moe.yaml --stage pretrain --build-cache
~~~

完成后设 `data.embedding_mode: cached` 开始相应阶段。构建只覆盖该阶段的固定 Training 和 Validation，可将多个阶段的 Query 存到同一兼容目录。不同历史池可在 `replay.simulation.data`、`replay.observation.data`、`growth_validation_data` 指定 `embedding_mode`、`cache_path`、`cache_dtype`。

Cache 使用按 Query 年份分片的 SQLite，按 batch 查索引和按页读取。构建按 `cache_chunk_rows` 提交并记录游标，恢复时跳过已保存的 Query。版本检查包括 Encoder 实际参数和 buffers 的 SHA-256、Query 版本、preprocessing 代码指纹、采样、gather mode 与起始覆盖。Driver 清单记录解析路径、inode、大小、mtime 和 ctime，小文件同时核对内容 SHA-256。原文件被替换或原路径的内容被改写后需在新目录重建。标签改变且 Query 与环境版本相同时可复用已有行。

全量 Training 和 Validation 的 128 维向量粗计 FP32 约 462 GB、FP16 约 231 GB，另有 Query 索引与页开销。FP16 是存储精度选项，模型读入时转为 FP32。

### 指标与专家记录

`train.log` 按 optimizer interval 记录各任务 batch 数、成功更新数、各任务和来源的样本数、Loss、Replay 计数和非有限梯度跳步。Dense Routing 报告每个任务的平均权重、最高权重出现比例、Entropy 和 `exp(mean_entropy)`。梯度诊断按 `gradient_diagnostic_interval_steps` 记录各任务 Router、各专家、各任务对共享 MLP 隐藏层的梯度范数。两任务均有梯度时可通过 `task_gradient_cosine` 记录 cosine。非有限 Loss 停止当前扫描，非有限梯度跳过 optimizer、Signature 和 anchor 更新。

`metrics.json` 与 `metrics.csv` 保存完整 epoch 和增长记录。用 `moe.checkpoint.load_checkpoint` 与 `restore_model` 恢复拓扑，`restore_encoder` 核对冻结 Encoder。专家的 ID、父 ID、出生 checkpoint 和增长结果见 `model.expert_records`。固定 anchors、统计和 fingerprints 来自 `model.signatures`。

## 测试

~~~bash
smoke_dir=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp "$smoke_dir" moe/tests
test_status=$?
rm -rf -- "$smoke_dir"
exit "$test_status"
~~~

测试覆盖固定划分、归一化、Dense Router、冻结 Encoder、独立 Replay 和 Teacher、非有限保护、候选接受、checkpoint 恢复与缓存失效。真实 JEPA 已用于 CPU 和 CUDA 小规模在线和缓存验证。
