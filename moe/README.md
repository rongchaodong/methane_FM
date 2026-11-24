# Continual Mixture of Response Experts

同一个模型预测自然甲烷排放和土壤甲烷吸收。Environmental JEPA 把每个点或格点区域的日、月 Query 编为 128 维向量。`ContinualMethaneMoE` 对同一向量计算共享基础预测 \(b(z)\in\mathbb R^2\)、各专家的残差 \(r_k(z)\in\mathbb R^2\)，以及每个任务单独的路由

\[
\pi_{q,k}(z)=\operatorname{softmax}_k\!\left(\frac{g_{q,k}(z)-\beta_q d_{k,q}(z)}{\tau}\right),
\qquad \hat y_q=b_q(z)+\sum_k\pi_{q,k}(z)r_{k,q}(z).
\]

排放采用正方向，吸收采用负方向，净通量等于两者之和。输出层直接给出有符号通量。可选的符号惩罚默认权重为零。两项任务共享专家库，分别对路由权重归一化。路由可配置 dense 或 top-k，以及是否使用 Signature。标签只影响监督、统计更新和增长决策。来源只用于分组、加权和可选的观测校准。

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
2. **Observation Fine-tuning** 继承模拟 checkpoint。Phase A 冻结 JEPA 并调整 MoE，Phase B 用分开的较低学习率局部解冻。Phase B 可使用模拟 replay 及 Phase A 模型的预测蒸馏。FLUXNET 筛过的净通量暂作 emission 的近似标签，可切换为 net 监督。Chamber 使用负向土壤通量监督 uptake。
3. **Continual Growth** 先用训练窗口上的质量合格样本估计已有专家的潜在空间距离和单专家响应损失，并检测正式模型的持续高误差。两项阈值同时满足才提出候选。复制综合距离最小的父专家，连同其签名和分流能力建立独立候选模型。在父专家和其它旧参数冻结时，只训练新专家和新增路由。历史 replay 与旧模型蒸馏可加入候选损失。独立验证分别计算新数据改善和历史 replay 损失变化。增长收益为新数据改善减去正向遗忘惩罚和新增参数成本。验证样本数、遗忘上限与收益阈值共同决定 `accepted`、`rejected` 或 `pending_insufficient_data`。

增长瞬间，父专家的原路由概率在两份相同的响应间平分。用 softmax 的有效 logit 表示时，两份 logit 在温度缩放后的空间各减去 \(\log 2\)。硬 top-k 情况保留原选择单元，再在其内部拆分概率，避免增加专家时挤掉其它原本入选的专家。候选状态独立于正式模型，只有通过验证才替换正式模型并保存拓扑、专家 ID、Signatures 和历史。

物理响应约束默认关闭。启用 `physical_response_weight` 时，自定义数据接入需为 batch 提供可信的 `physical_response[task]["reference"]` 和可选 `weights`。参考张量对应 `environment["values"]` 的 `[B, token, driver]` 形状，单位为预测通量率除以原始驱动物理单位。梯度经过在线 JEPA 的有效值输入。latent \(Z\) 的导数对应不同目标，缓存无法提供原始驱动 \(X\) 的梯度。

## 本地数据与监督单位

`data/methane_data` 是本地数据链接。模拟目录名为 `simulations`，观测目录名为 `observations`。所有监督量在读入时转换为 `mg_ch4_m2_day`，即 mg CH₄ m⁻² d⁻¹。月平均通量率继续保留整月 Query。

| 产品和相对路径 | 本地字段和维度 | 时间、空间与转换 |
| --- | --- | --- |
| TEM `simulations/TEM_EMISSION/CH4_emission_intensity_{year}.nc` | `CH4_emission(latitude, longitude, month, day)`，360 × 720 × 12 × 31 | 38 个年度文件，1979–2018，0.5°。跳过月份中不存在的日期。变量缺少单位属性，需填写 `data.adapters.tem_emission.input_unit`。 |
| TEM `simulations/TEM_ch4_wetland_soilsink_1950_2020_mgm2day.nc4` | `ch4_oxidation(month, year, longitude, latitude)`，12 × 71 × 720 × 360 | 1950–2020，0.5°。`unit=mg/m2/day`，沿用文件中的通量符号。 |
| MeMo `simulations/MeMo_1990-2009_monthly_corrected.nc` | `CH4uptake(Time, Lat, Lon)`，240 × 180 × 360 | 1990–2009，1°。`Time` 为 1–240，首月按文件说明从 1990-01 起算。正吸收量 `mgCH4/m2/month` 按当月日数除后转为负向通量。 |
| FLUXNET `observations/FLUXNET-CH4/FLUXNET_T1_DD.csv` 与 `FLUXNET_CH4_2024.csv` | 日表 `Site`, `TIMESTAMP`, `FCH4_F_ANNOPTLM`。元数据 `SITE_ID`, `LOCATION_LAT`, `LOCATION_LONG`, `IGBP` | 规范化站点 ID 后连接两个表。`FCH4_F_ANNOPTLM` 为填补缺测后的净 CH₄ 通量，默认单位 `nmol_ch4_m2_s`。当前代理监督默认选 `WET` 和非负净通量。月模式聚合合格的日均速率，默认至少 20 天。 |
| Chamber `observations/chamber_processed.csv` | `site_unique_id`, `lat`, `lon`, `FCH4_sample_year/month/day`，以及 `FCH4_daily_mean` 或 `FCH4_monthly_processed` | 默认逐日，月模式按站点、研究、坐标和月使用去重后的月字段。默认使用非正值作为 uptake。CSV 缺少通量单位，需填写 `data.adapters.chamber.input_unit`。 |

FLUXNET 的 `ANNOPTLM`、缺测约定与标准通量定义见数据发布说明。 [FLUXNET-CH₄ 数据论文](https://essd.copernicus.org/articles/13/3607/2021/#appendix-B)

观测 CSV 的日历字段按名义 UTC 日界构建 Query。站点没有空间支持定义时使用点 Query，可按产品覆盖 `spatial_support`。`quality_filters` 支持指定列的 `allowed`、`min` 和 `max`，可用 `sample_weight_column` 传递权重。FLUXNET 可通过 `training.finetune.fluxnet_target_type: net` 改用净通量监督，并同步设置 `training.continual.fluxnet_target_type`。

模拟按共享空间块保留验证集。训练和验证各自对格点 × 有效时段做固定种子抽样，以 `samples_per_simulation_source` 控制每个产品的上限。数据加载时按样本读标签和 JEPA 原生窗口。默认从 1980 年起选取目标，以保证年初 14 天历史落在已处理 ERA5 覆盖内。观测按站点整体拆分训练和验证，可用 `validation_sites` 指定留出站点。开始微调前通过阶段的 `data` 覆盖保留一批独立站点或后续时段供 continual 使用。

## 配置和启动

在仓库根目录运行，需要项目的 `fm` 环境和 [encoding/requirements.txt](../encoding/requirements.txt) 中的依赖。

1. 核实 TEM 日排放和 Chamber 的原始单位，在 [moe.yaml](configs/moe.yaml) 填写两个 `input_unit`。支持 `mg_ch4_m2_day`、`mg_ch4_m2_month`、`mg_c_m2_day`、`nmol_ch4_m2_s`、`ug_ch4_m2_h`、`ug_c_m2_h`。未配置的产品会报告缺少单位。若 Chamber 文件存的是正吸收量，核实后配置 `sign: -1`。
2. 配置微调与 continual 的观测范围。在 `training.finetune.data`、`training.continual.data` 中分别指定 `observation_site_ids` 或 `observation_date_range`，为每批观测留下独立验证站点。增长的 replay 数据通过 `training.continual.replay_sources` 和 `replay_data` 配置，默认是历史 TEM / MeMo。候选接受阈值用训练配置中的 `growth` 设置，`distance_threshold` 和 `loss_threshold` 保持 `null` 时只运行监测。
3. 使用以下入口。输出目录中的 `best.pt` 是最佳验证 checkpoint，`last.pt` 是最新可恢复 checkpoint。微调另保存 `best-A.pt` / `best-B.pt`。

~~~bash
conda activate fm
python -m moe.train --config moe/configs/moe.yaml --stage pretrain
python -m moe.train --config moe/configs/moe.yaml --stage finetune \
  --checkpoint moe/runs/pretrain/best.pt
python -m moe.train --config moe/configs/moe.yaml --stage continual \
  --checkpoint moe/runs/finetune/best.pt
~~~

任一阶段中断后，用相同配置、相同 `--stage` 和相应的 `--resume <run_dir>/last.pt` 恢复。后续有一批独立观测时，更新 continual 的站点或日期范围，并通过 `--checkpoint <previous_continual_dir>/last.pt` 继承专家、优化器和增长历史。可用 `--output-dir` 指定单独的实验目录。CUDA 运行可选 FP16 或 BF16，CPU 调试可将 `device` 设为 `cpu`。学习率、梯度裁剪、调度、早停、种子、各来源权重与校准在同一个 YAML 中配置。

### Cached embeddings

将已有 embeddings 和生成它们的 `QuerySpec` 成对交给 `moe.data.save_embedding_cache(path, pairs, encoder.provenance, gather_mode="history")`。保存内容包括 encoder hash、Query 及 preprocessing 版本、采样与 gather 程序配置，以及按 Query 精确匹配的向量。训练时设置 `data.embedding_mode: cached` 和 `data.cache_path`，保持 Encoder 冻结。切换到 Phase B 或继续使用已联合微调过的 Encoder 时配置 online 模式。

### 查看专家和增长记录

`moe.checkpoint.load_checkpoint` 与 `restore_model` 恢复动态 Router 和 Expert Bank，`restore_encoder` 恢复环境 Encoder。专家 ID、父 ID、创建阶段、状态、验证指标和出生 checkpoint 来自 `model.expert_records`。固定 anchors、统计和 fingerprints 来自 `model.signatures`，对比矩阵通过 `model.signatures.diagnostics()` 获取。Phase B 重新估计路由用的 latent moments，保留历史 Response Fingerprint 的固定 latent probes。

正式运行时按阶段导出 `metrics.json` 与 `metrics.csv`，记录 RMSE、MAE、R²、各来源及各任务损失、路由使用率和熵、Signature 距离、Fingerprint 相似度及增长收益。增长历史和接受时的 `birth-candidate_NNNN.pt` 可用于重现当时的专家结构。

## 随机张量验证

~~~bash
smoke_dir=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp "$smoke_dir" moe/tests
test_status=$?
rm -rf -- "$smoke_dir"
exit "$test_status"
~~~

测试覆盖随机张量前向、梯度、标签掩码、签名统计、候选分割与接受状态、动态 checkpoint 恢复。已只读核验 JEPA 权重与真实数据元数据。真实三阶段训练、真实 anchor 选择、模型效果和吞吐量待首次正式运行验证。
