# Continual Mixture of Response Experts

同一个模型预测自然甲烷排放和 upland 土壤甲烷净交换。Environmental JEPA 把每个点或格点区域的日、月 Query 编为 128 维向量。`ContinualMethaneMoE` 对同一向量计算共享基础预测 \(b(z)\in\mathbb R^2\)、各专家的残差 \(r_k(z)\in\mathbb R^2\)，以及每个任务单独的路由

\[
\pi_{q,k}(z)=\operatorname{softmax}_k\!\left(\frac{g_{q,k}(z)-\beta_q d_{k,q}(z)}{\tau}\right),
\qquad \hat y_q=b_q(z)+\sum_k\pi_{q,k}(z)r_{k,q}(z).
\]

通量为正表示向大气释放，为负表示吸收。Uptake 任务包含 upland soil 净交换观测，Chamber 的正通量表示该土壤在观测窗口内为净源。输出层给出有符号通量，可选的符号惩罚默认权重为零。两项任务共享专家库，分别对路由权重归一化。路由可配置 dense 或 top-k，以及是否使用 Signature。标签影响监督、统计更新和增长决策。来源用于分组、加权和可选的观测校准。

## JEPA 权重与 Query

- `encoder.checkpoint: auto` 从 `encoding/model_save/*/` 查找与验证 `jepa_loss` 最佳轮次相符、且保存于该轮的导出权重。本仓库找到 `encoding/model_save/soil_5km_trial_4/encoder.pt`。同目录 `last.pt` 记录 `epoch=4` 和完整验证历史。`train.log` 中第 4 轮 `jepa_loss=0.0142367887`，第 3 轮为 `0.0156018262`。含方差惩罚的总 `loss` 则在第 3 轮较低（`0.0382674098` 对 `0.0386355824`）。
- 加载时逐张量对照导出权重与 `last.pt` 中的 `online_encoder.*`，构造 `EnvironmentalEncoder`，并保存 checkpoint 路径、SHA-256、模型配置、驱动表、采样配置和 Query / preprocessing 版本。显式路径可指向 Encoder 导出或完整 JEPA checkpoint。若有意使用无法验证轮次的独立导出，将 `expected_best_epoch` 设为 `null`，并提供用于读取采样参数的 `encoder.jepa_config`。
- Online 模式复用 `QuerySpec`、`MultiResolutionGatherer` 和 `EnvironmentalCollator`。时间区间为 UTC 左闭右开。默认读取 Query 起点以前的 14 天历史。使用当期环境驱动估计通量时可配置 `query_window`，此时覆盖整个日或月。切换模式会改变模型输入的时间信息，训练与部署应保持一致。
- 当前产品默认使用 `encoder.spatial_selection: intersects`，读取与矩形有正面积相交的原生像元，每轴完整宽度最多 2°。从原第 4 轮 JEPA 切换到该策略时使用 5,120 个 tokens 的产品预算，可通过 `encoder.max_tokens` 调整。显式指定 `centers` 可重放原采样策略。空间规则和实际采样策略参与缓存校验，原版本的 embedding 缓存需重新生成。
- Cached 模式使用已有的 `[N, 128]` embeddings 和对应 Query / 来源元数据。缓存必须匹配 JEPA 权重、Query、preprocessing 和 gather 模式。部分解冻阶段使用 online 模式。

## Expert Signature

每个专家和任务保存 \(S_{k,q}=(\mu_{k,q},\sigma^2_{k,q},\rho_{k,q})\)。中心和逐维方差通过有效样本的路由权重更新。用于路由的距离为

\[
d_{k,q}(z)=\frac{1}{D}\sum_d\frac{(z_d-\mu_{k,q,d})^2}{\sigma^2_{k,q,d}+\epsilon}.
\]

尚无统计量时保持可学习路由。Response Fingerprint 是专家在任务固定 anchor 集上的残差值序列。Anchor 从真实训练 batch 的 latent 选择，随 checkpoint 保存并记录来源和版本，可用于比较响应函数。

## 三个训练阶段

1. **Simulation Pretraining** 同时训练共享预测、Router 和专家。TEM 日排放监督 emission，TEM / MeMo 月吸收监督 uptake。按来源配置模型损失权重、Huber / MSE、可选校准和软负载均衡。每条样本保留原始网格和时间支持，以同一支持的 Query 直接预测通量率。
2. **Observation Fine-tuning** 继承模拟 checkpoint。Phase A 冻结 JEPA 并调整 MoE，Phase B 用分开的较低学习率局部解冻。Phase B 可使用模拟 Replay 和 Phase A 模型的预测蒸馏。默认以湿地 FLUXNET 非负净交换代理 emission，以 Chamber upland 土壤净交换监督 uptake。
3. **Continual Growth** 先用训练窗口上的质量合格样本估计已有专家的潜在空间距离和单专家响应损失，并检测正式模型的持续高误差。两项阈值同时满足才提出候选。复制综合距离最小的父专家，连同其签名和分流能力建立独立候选模型。在父专家和其它旧参数冻结时，只训练新专家和新增路由。历史 replay 与旧模型蒸馏可加入候选损失。独立验证分别计算新数据改善和历史 replay 损失变化。增长收益为新数据改善减去正向遗忘惩罚和新增参数成本。验证样本数、遗忘上限与收益阈值共同决定 `accepted`、`rejected` 或 `pending_insufficient_data`。

增长瞬间，父专家的原路由概率在两份相同的响应间平分。用 softmax 的有效 logit 表示时，两份 logit 在温度缩放后的空间各减去 \(\log 2\)。硬 top-k 情况保留原选择单元，再在其内部拆分概率，避免增加专家时挤掉其它原本入选的专家。候选状态独立于正式模型，只有通过验证才替换正式模型并保存拓扑、专家 ID、Signatures 和历史。

物理响应约束默认关闭。启用 `physical_response_weight` 时，自定义数据接入需为 batch 提供可信的 `physical_response[task]["reference"]` 和可选 `weights`。参考张量对应 `environment["values"]` 的 `[B, token, driver]` 形状，单位为预测通量率除以原始驱动物理单位。梯度经过在线 JEPA 的有效值输入。latent \(Z\) 的导数对应不同目标，缓存无法提供原始驱动 \(X\) 的梯度。

## 本地数据与监督单位

`data/methane_data` 链接本地数据。统一物理标签单位为 `mg_ch4_m2_day`（mg CH₄ m⁻² d⁻¹）。Query 保留标签的日或月时间支持，面积分母仍按源产品定义。

| 产品和相对路径 | 本地字段与原始质量基准 | 时间和面积支持及处理 |
| --- | --- | --- |
| TEM `simulations/TEM_EMISSION/CH4_emission_intensity_{year}.nc` | `CH4_emission(latitude, longitude, month, day)`，360 × 720 × 12 × 31。数据提供者已确认以 mg CH₄ 计，文件缺少单位属性。 | 40 个年度文件，1979–2018，0.5° 日格点。原值已经是日速率。跳过公历无效日期，过滤全部负值与无效值，保留零。文件未注明面积分母是整格、湿地面积还是其他面积。 |
| TEM `simulations/TEM_ch4_wetland_soilsink_1950_2020_mgm2day.nc4` | `ch4_oxidation(month, year, longitude, latitude)`，12 × 71 × 720 × 360。文件标注 `mg/m2/day`，CH₄ 质量基准按本项目确认。 | 文件覆盖 1950–2020，正式模拟选取 1979–2020。0.5° 整月标签已经是日速率。扫描所选年份确认负值占多数，保留原始符号和少量正通量。面积分母及正值的过程来源细分未记录。 |
| MeMo `simulations/MeMo_1990-2009_monthly_corrected.nc` | `CH4uptake(Time, Lat, Lon)`，240 × 180 × 360。文件明确标注 `mgCH4/m2/month`，`Time` 从 1 开始对应 1990-01。 | 1990–2009，1°。正吸收量除以当月公历天数后取负，包含闰年二月。负的原始吸收量按无效值过滤。文件未注明面积分母的土地覆盖口径。 |
| FLUXNET `observations/FLUXNET-CH4/FLUXNET_T1_DD.csv` 与 `FLUXNET_CH4_2024.csv` | 日表 `FCH4_F_ANNOPTLM` 为 gap-filled 净生态系统交换，量纲为 nmol CH₄ m⁻² s⁻¹。以规范化后的 `Site` 和 `SITE_ID` 连接站点元数据。 | 涡度协方差通量按站点通量足迹解释，当前 CSV 无逐条足迹面积或本地时区字段。默认 `WET`、有限且非负，从原始单位一次乘以 1.386072。可选月模式按合格日速率取均值，默认至少 20 天。 |
| Chamber `observations/chamber_processed.csv` | 末列 `FCH4_monthly_processed`，日模式使用 `FCH4_daily_mean`。按项目确认使用 mg CH₄ m⁻² d⁻¹，CSV 缺少单位属性。 | 默认以给定月速率监督整月 Query。同站点、研究、坐标和月份的重复行只计一次。当前表的站点均标为 upland，保留正、负和零。`nonpositive_uptake: true` 可过滤正值。具体 chamber 面积和逐条月内采样覆盖未记录。 |

默认假设湿地 FLUXNET 的非负净交换能提供 emission 代理监督。通量包含净源汇共同作用，无法从这列直接分离 gross emission。当前本地日表有 `FCH4_F` 对照列，缺少正式 QC 标志和 ANN 填补不确定度。数据审计统计可用的 QC 列、缺测及筛选计数，维持现有 gap-filled 标签使用方式。也可通过 `training.finetune.fluxnet_target_type: net` 启用净通量监督。 [FLUXNET-CH₄ 数据发布说明](https://essd.copernicus.org/articles/13/3607/2021/#appendix-B)

Chamber 月模式直接使用先前研究确认的 `FCH4_monthly_processed`，有效零与正月值均进入 uptake。当前 14,484 行中 7,565 行是同站点、研究、坐标和月份的重复行，得到 6,919 条月标签，其中 5,987 条为负、932 条为零，缺测和其他适配器过滤均为 0。旧 Site Split 另去除 2 条跨集合副本，得到 6,917 条。当前站内时间划分去除 83 条跨集合副本，得到 6,836 条。被移除的月值在留出集合中均有相同 Query 和数值的副本。

观测 CSV 以名义日期的 UTC 日界生成 Query。默认使用点 Query，可按产品配置 `spatial_support`。可选 `quality_filters` 支持 `allowed`、`min`、`max`，`sample_weight_column` 可传入已有权重。

## TEM 网格中心

TEM 两个产品都缺少显式坐标边界属性。日排放纬度递增，范围为 −90° 至 89.5°。完整覆盖两极对应的是南边界，中心纬度为原纬度加半格。月氧化纬度递减，范围为 90° 至 −89.5°。其纬度为北边界，中心纬度为原纬度减半格。两个产品的经度均从 −180° 到 179.5°，作为西边界加半格后归一到 [−180°, 180°)。偏移按实际间距计算，原始标签索引保持对应，支持窗口仍为 0.5° × 0.5°。MeMo 已采用格心。预测或制图可复用 `GridProduct.lat`、`GridProduct.lon` 和 `GridProduct.support`。

## 全量模拟与固定划分

- 正式配置 `simulation_mode: full` 使用 1979 年起全部可用的合法网格标签。默认 `simulation_split_mode: temporal` 按产品将末段约 20% 的整年划给 Validation，前段用于 Training。当前 TEM 日排放留出 2011–2018，TEM 月氧化留出 2013–2020，MeMo 留出 2006–2009。可用 `simulation_split_mode: spatial` 恢复全球共享 5° 空间块的 80/20 划分，同样遍历全部有效标签。模拟阶段只有 Training 和 Validation。
- 清单保存每个分块的原生索引范围、有效标签数、位图偏移和 SHA-256。准备时扫描 NetCDF，训练时按需读取连续 tile，并按位图构造该 batch 的 Query。默认目标分块 16 MiB，loader 限量复用文件句柄和 tile。训练打乱 shard 和 shard 内连续 128 个 batch 的区间，保留区间内环境读取的局部性。每个完整 epoch 遍历全部 Training，再评估全部 Validation。默认每 10,000 个已处理 batch 写进度和 `last.pt`，可用 `checkpoint_interval_steps` 调整。恢复时从已提交的游标继续，并恢复该 pass 的统计。完整 Validation 决定学习率调度与最佳 checkpoint。
- 1979 年初的历史窗口只从 `simulation_daily_start: '1979-01-01'` 读取实际可用的日驱动。没有日历史时使用已有静态环境。之后缺驱动文件或没有任何环境 token 的有效标签会报错。
- `samples_per_simulation_source` 仅在 `simulation_mode: debug` 下决定候选上限。独立的 [moe_debug.yaml](configs/moe_debug.yaml) 用于小规模运行。

## 观测划分和 Replay

- `observation_split_mode: temporal` 在每一个 FLUXNET 站按日、每一个 Chamber 站按去重后的月对有效标签排序。同一个日期或月份的记录一起进入一组。有至少三个时段时站内按连续的约 60% / 20% / 20% 划分，按有效时段数取整并保证各集合都有时段。各站时间边界独立。默认 `observation_short_sites: train_only` 将一至两个有效时段的站点留在 Training。可选 `train_validation` 对两个时段各留一个训练和验证，或设为 `exclude`。可用 `observation_split_mode: spatial` 恢复按站点末次记录时间排序的 Site Split。
- 同一标签或精确 Query 的跨集合副本保留在最严格的留出集合。单站时间支持有重叠时数据审计报错。审计记录各集合站点数、重叠站点、最近大圆距离、共享空间块和研究编号。站内划分的相同站点最近距离可为零。
- `data.split_seed` 控制划分所需的随机性，`training.common.training_seed` 控制训练随机性。默认 v2 清单在 `moe/runs/splits-v2`。CSV 记录 SHA-256，模拟 NetCDF 记录路径、大小和修改时间。检查点记录清单 SHA-256，启动时验证历史清单、模拟位图及源文件版本。旧 v1 清单保留在原路径，v2 阶段使用新的划分与 checkpoint。
- 初始微调使用 `initial_observation.json`。新 cohort 设置 `training.continual.data.observation_cohort` 和数据范围。时间模式下同站新批次须晚于已冻结记录，空间模式下历史站点沿用原集合。两种模式都保护原 Testing 标签和 Query。Replay 从历史 Training 和 Validation 原位读取，训练入口直接使用物理单位的训练侧标签。
- 离线审计按来源和划分报告原始量、有效量、正负零、缺测、过滤原因及分布。全量模拟另外记录各年精确计数、公历有效槽位、完整 pass 的 steps 和位图大小。模拟分位数使用小样本近似，其余统计和训练覆盖来自全量扫描。

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

默认 [moe.yaml](configs/moe.yaml) 选取当前文件中的全部有效观测作为初始微调数据。若要用现有文件中的部分站点或时段作为未来 cohort，先设置 `training.finetune.data.observation_site_ids` 或 `observation_date_range`，并为该实验指定单独的 `data.split_manifest_dir`。正式运行前为 continual 配置新 cohort 的 ID 和范围。用 `replay_sources` 选择历史来源，默认是 TEM / MeMo。`replay_data` 可覆盖 `embedding_mode`、`cache_path` 与 `gather_mode`。

正式训练使用以下入口。输出目录中的 `best.pt` 是最佳验证 checkpoint，`last.pt` 是最新可恢复 checkpoint。微调另保存 `best-A.pt` / `best-B.pt`。`distance_threshold` 和 `loss_threshold` 保持 `null` 时，continual 只运行增长监测。

~~~bash
python -m moe.train --config moe/configs/moe.yaml --stage pretrain
python -m moe.train --config moe/configs/moe.yaml --stage finetune \
  --checkpoint moe/runs/pretrain/best.pt
python -m moe.train --config moe/configs/moe.yaml --stage continual \
  --checkpoint moe/runs/finetune/best.pt
~~~

任一阶段中断后，用相同配置、相同 `--stage` 和相应的 `--resume <run_dir>/last.pt` 恢复。后续有一批新观测时，更新 continual 的 cohort ID 和数据范围，并通过 `--checkpoint <previous_continual_dir>/last.pt` 继承专家、优化器和历史划分。可用 `--output-dir` 指定单独的实验目录。CUDA 运行可选 FP16 或 BF16，CPU 调试可将 `device` 设为 `cpu`。学习率、梯度裁剪、调度、早停、种子、各来源权重与校准在同一个 YAML 中配置。

### Cached embeddings

小规模数据可以将用 v2 Query 生成的 embeddings 交给 `moe.data.save_embedding_cache(path, pairs, encoder.provenance, gather_mode="history")`。缓存包含校正后的坐标版本及日驱动覆盖边界，旧缓存需重新生成。训练时设置 `data.embedding_mode: cached` 和 `data.cache_path`，保持 Encoder 冻结。全量模拟、Phase B 和使用已联合微调的 Encoder 时使用 online 模式。

### 查看专家和增长记录

`moe.checkpoint.load_checkpoint` 与 `restore_model` 恢复动态 Router 和 Expert Bank，`restore_encoder` 恢复环境 Encoder。专家 ID、父 ID、创建阶段、状态、验证指标和出生 checkpoint 来自 `model.expert_records`。固定 anchors、统计和 fingerprints 来自 `model.signatures`，对比矩阵通过 `model.signatures.diagnostics()` 获取。Phase B 重新估计路由用的 latent moments，保留历史 Response Fingerprint 的固定 latent probes。

正式运行时按阶段导出 `metrics.json` 与 `metrics.csv`，记录 RMSE、MAE、R²、各来源及各任务损失、路由使用率和熵、Signature 距离、Fingerprint 相似度及增长收益。增长历史和接受时的 `birth-candidate_NNNN.pt` 可用于重现当时的专家结构。

## 测试

~~~bash
smoke_dir=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp "$smoke_dir" moe/tests
test_status=$?
rm -rf -- "$smoke_dir"
exit "$test_status"
~~~

测试覆盖数据读入、固定划分和 Replay 隔离，以及随机张量前向、梯度、签名统计、候选接受和 checkpoint 恢复。真实数据已通过只读划分审计。模型效果和训练吞吐量待正式运行评估。
