# methane_FM
Foundation Model of Global Methane Flux Prediction

## ERA5 数据预处理

下载 1979–2025 ERA5 Daily Statistics，每次请求一个变量的一整年，逐个顺序提交；`tp` 用 Sum，其余用 Mean。
处理时保留原始 0.25° 网格，仅转换单位，每个变量每年输出一个 NetCDF。

默认数据根目录：`/common/users/rd1281/ERA5_daily/`。
`downloads/{mean,sum}` 存放 ZIP 或待归位的 NetCDF；`netcdf/{mean,sum}/{year}` 存放按变量命名的原始数据；`processed` 存放转换结果。直接返回的 NetCDF 校验后移入 `netcdf`，ZIP 则保留并解压。

从项目根目录运行，实际下载前需配置 `~/.cdsapirc` 并接受 CDS 数据许可：

```bash
conda activate fm
cd preprocessing/ERA5_download
python download_era5_daily.py --dry-run  # 仅查看计划
python download_era5_daily.py           # 下载并解压
python process_era5_daily.py            # 单位转换
```

单年试运行可在两个脚本后加 `--start-year 1980 --end-year 1980`。
下载器可加 `--variable swvl1` 只下载指定变量，支持 `t2m/d2m/msdwswrf/swvl1/tp`。
中断后重跑相同命令，已验证完成的数据会自动跳过。
完成检查在启动时执行；多个进程需自行分配不重叠的年份或变量。

## 环境表示学习（Environmental JEPA）

[encoding/](encoding/README.md) 直接读取已处理的 ERA5 日数据、5 km SoilGrids GeoTIFF、0.5° 地形和类别 NetCDF，以及全球年度浓度。点查询读取所属原生像元，矩形查询读取所有与其有正面积相交的原生像元。矩形的纬向和经向完整宽度各自最多为 2°。同一 Encoder 生成 `[B,128]` 表示，JEPA 可同时预测日、月和不同空间范围的未来表示。

```bash
conda activate fm
python -m pip install -r encoding/requirements.txt
python -m encoding.train_environmental_jepa --config encoding/configs/environmental_jepa.yaml
```

训练进度会写入运行目录的 `train.log`。空间选择、未来窗口配置、模型和恢复训练的入口见 [encoding 说明](encoding/README.md)。

## 甲烷响应专家（Continual MoE）

[moe/](moe/README.md) 用冻结的 Environmental JEPA 和共享专家库预测带符号的 Emission 与 Uptake。两个任务使用独立的 Dense Router 和 Training 归一化，默认在归一化空间计算 MSE。模拟预训练保留从 1979 年起的全量 TEM / MeMo 标签与按年验证。观测用单一微调阶段和站内时间划分，Replay 与蒸馏可分别开启。

运行 `python -m moe.train --config moe/configs/moe.yaml --stage pretrain --prepare-data` 核验模拟清单，初始观测将 `--stage` 改为 `finetune`。分块 Embedding Cache、按 optimizer steps 保存与验证、物理单位指标及 checkpoint 恢复见 [MoE 说明](moe/README.md)。
