# Environmental JEPA

在 ERA5 日数据与静态环境数据的原网格上，训练一个带坐标与时间条件的环境 Encoder。Point 和 Area 共用同一模型，默认得到 `[B, 128]` 的 embedding。

## 从哪读起

1. [environmental_jepa.yaml](configs/environmental_jepa.yaml) 指定文件、变量、查询位置和预测尺度。[specs.py](specs.py) 定义 `QuerySpec` 和驱动配置。
2. [sources.py](sources.py) 从 NetCDF 和 GeoTIFF 读取原生窗口。[gather.py](gather.py) 决定时间窗口并合并各数据源的 token blocks。
3. [dataset.py](dataset.py) 为每个 anchor 构造历史与多个未来窗口，并把变长观测整理成 PyTorch batch。
4. [model.py](model.py) 将 Query 编成 Q，将同一原生格点、同一时间的有效变量聚合成 K/V token。一次 Cross-Attention 加残差、归一化和输出投影得到 embedding。
5. [jepa.py](jepa.py) 用历史 embedding 与未来 Query 预测未来 latent。EMA Target Encoder 用未来环境观测编码学习目标。[training.py](training.py) 负责更新、验证及保存。

## 如何选择原生格点

`delta_lat` 和 `delta_lon` 分别是矩形的完整纬向与经向宽度，单位为度。两轴均为零表示点查询，每个空间来源读取包含该点的一个原生像元。NetCDF 内部共享边界归入偏北、偏东的单元，北向上的 GeoTIFF 使用原生行列的半开区间。矩形两轴均为正，各自最多 2°，包含恰好 2° 的情况。超限在创建 Query 时直接报错。

默认 `sampling.spatial_selection: intersects` 选取与查询有正面积相交的每个原生像元，纯边界接触的交面积为零。NetCDF 优先使用文件的坐标 bounds。当前文件按相邻中心推导边界，经度回绕、纬度裁至两极。token 的矩形中心和宽度来自实际边界，包括极点处的半格。以 `(40.5, -93.5)` 为中心查询 1° × 1°，ERA5、0.5° 地形和 5 km 土壤分别相交 25、9 和 414 个像元。

空间上限与 `sampling.max_tokens` 分别检查。产品默认预算为 5,120 个 tokens，覆盖上述位置 2° × 2° 的完整 2020 年 7 月窗口（4,103 个有效 tokens）。更长时间窗或更密的来源触及预算时会报错。

第 4 epoch checkpoint 训练时使用 `centers`。重放原试验时可显式指定这一策略。用旧权重执行 `intersects` 推理时，矩形的输入像元与历史训练存在差异，此时需按当前规则重新评估验证指标。

点查询限制每个源每个时刻的空间采样数。只要有多个日期或来源，Cross-Attention 仍可利用这些 tokens。查询覆盖宽度不代表新增空间观测，Area embedding 也没有面积加权或子区域一致性约束。

## 数据与预测范围

默认配置读取 `ERA5_daily/processed/` 中的 TAIR、VAPR、PREC、SOLR、SM，保留现有物理单位和 0.25° 网格。地形和类别变量仍读取 `Drivers/` 的 0.5° NetCDF。

`soil_5km` 读取 6 份 SoilGrids 0–5 cm GeoTIFF，保留原生 Interrupted Goode Homolosine 网格。矩形在日期线、赤道和投影分瓣处拆分，然后按原生 5,000 m 像元行求相交范围。跨分区的像元按原生行列去重。投影边缘像元的坐标与局部角宽根据其有效地理范围估算。YAML 中 GeoTIFF 的 `resolution` 使用米，NetCDF 使用度。土壤变量沿用文件单位和标准化初始值，缺测通过各变量的 mask 处理。[SoilGrids 单位表](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html)

每个 source 对应一套原生网格和时间频率。添加同网格变量时，在对应 `drivers` 增加文件路径和 `normalization: [mean, std]` 或 `categories`。多变量同文件时可用 `variable` 指定文件内变量名。不同网格或频率单列一个来源。配置中的标准化系数是与文件单位相符的初始尺度，正式实验可在训练时间段估计统计量。

`global_concentrations` 直接读取 `Drivers/` 的两列年度浓度文本。历史 Context 使用最近完整公历年的 CH₄、CO₂ 浓度，合成一个全球 token，并保留文件中的原始数值。这里以公历年结束作为可用日期的近似，实际发布滞后仍需核实。年度 NetCDF 的观测按完整公历年支持读取。

设 anchor 的 UTC 起点为 `t`。Context 读 `[t - history_days, t)` 内的完整日观测，以及可用静态变量和已完成年度浓度。日、月 Target 由未来 ERA5 观测编码，独立设置以下参数。

| 配置 | 作用 |
| --- | --- |
| `horizon_days` | 相对 `t` 的最早未来日期 |
| `period: day` | 从不早于该时点的第一个 UTC 日界读一天，可用 `duration_days` 改为多天 |
| `period: month` | 从不早于该日期的第一个 UTC 月初读至下月月初 |
| `spatial_support` | 未来窗口的 `[delta_lat, delta_lon]`，省略则继承 anchor |

月目标由整月原生逐日 tokens 编码，闰年二月读 29 天。默认同时训练第 1 天、第 7 天的矩形区域和 1° × 1° 的未来月窗口。模型可接受配置中的不同时间和空间支持，真实泛化仍取决于训练覆盖。

## 运行与推理

从项目根目录运行以下命令。

```bash
conda activate fm
python -m pip install -r encoding/requirements.txt
python -m encoding.train_environmental_jepa --config encoding/configs/environmental_jepa.yaml
```

默认配置用 CUDA 训练 4 轮，适合四个示例地点的小规模试验。只用 CPU 时将 `training.device` 设为 `cpu`。可在 YAML 调整 `training.epochs` 和 `model.dropout`。默认训练目录为 `encoding/model_save/environmental_jepa/`，也可用 `--output-dir` 为每次实验指定目录。

目录内的 `train.log` 会追加 JSON Lines，包含启动配置、批次进度、每轮训练和验证指标。在 CUDA 上也记录每轮 PyTorch 的显存分配峰值和保留峰值。用 `training.log_every_steps` 设置批次记录频率，设为 0 时只记录启动和每轮指标。

`last.pt` 用于恢复训练。保留数据、目标、模型、优化器类型及实际混合精度设置，将 `training.epochs` 改为目标总轮数，使用 `--resume encoding/model_save/environmental_jepa/last.pt`。同目录的 `encoder.pt` 是推理权重。原 `soil_05` 训练 checkpoint 的数据配置与 `soil_5km` 不兼容，切换数据需开启新 run。

```python
import torch

from encoding import (
    EnvironmentalCollator,
    MultiResolutionGatherer,
    QuerySpec,
    build_data_sources,
    load_encoder,
)
from encoding.training import load_config

encoder = load_encoder("encoding/model_save/environmental_jepa/encoder.pt")
config = load_config("encoding/configs/environmental_jepa.yaml")
registry = encoder.registry
gatherer = MultiResolutionGatherer(registry, build_data_sources(registry), config["sampling"])

query = QuerySpec(40.5, -74.5, "2020-02-01", "2020-03-01", 0.5, 1.0)
context = gatherer.gather(query)
batch = EnvironmentalCollator(registry).collate_contexts([context])
with torch.inference_mode():
    embedding = encoder(batch)  # [1, 128]
```

上例用月窗口的 Query 和月初以前的历史做编码。若要编码当月已有观测，在 `gather()` 显式传入 `start_time=query.start_time, end_time=query.end_time`。
