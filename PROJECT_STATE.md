# 项目状态

- 已完成 ERA5 Daily 下载与单位转换脚本，属于数据准备的第一步；入口及命令见 README。下载器支持 `--variable` 指定单变量；仅在启动时检查完成状态，并发任务需按年份或变量错开。
- 关键约定：1979–2025、UTC 逐日，每次请求一个变量的一整年，顺序下载；`tp` 用 Sum，其余用 Mean。保留原始 0.25° 格点；PREC 仅乘 1000，VAPR 保持原 Magnus 公式。
- 验证：用户网页实测 cost 限额 400、两个变量 760；1979 年 t2m 已实际下载为 NetCDF4（365 天、721×1440），整场分块读取通过。下载器兼容直接 NetCDF 与 ZIP，按真实格式保存及归位。
- 当前（2026-11-14）：全部 235 个年份/变量任务通过完整日历、统一原始网格及首尾格点读取检查，处理 dry-run 全部通过；尚未生成 processed 文件。未逐格扫描全部数据。用户已停止 download2；重复下载留下的 `.part` 不影响有效原始文件。
- 当前（2026-11-20）：`ERA5_daily/processed/` 已有 1979–2025 的 5 个变量年度 NetCDF，真实配置读取 TAIR、VAPR、PREC、SOLR、SM。土壤改用 `Drivers/` 下 6 份原生 5,000 m SoilGrids GeoTIFF，按 Interrupted Goode Homolosine 网格读取窗口，保留原始值和缺测 mask。地形与分类仍用 0.5° NetCDF。全球 CH₄ 与 CO₂ 年度浓度使用最近完整年，并以公历年结束近似可用日期。
- `encoding/` 按 Query 的矩形宽度分别选择各来源原生格点，使用简化的坐标条件 Cross-Attention。JEPA Context 读取历史，Target 可配置日、月与不同矩形尺寸。时间区间采用 UTC 左闭右开，Target 排除静态和全球浓度。
- 闰年二月 1° × 1° 目标包含 464 个 ERA5 tokens，四个默认位置实测最大 Context 为 701 个 tokens。已用真实数据验证 CPU 训练、恢复和 Encoder 加载，以及 CUDA FP32 训练。四地点试验中第 4 轮验证 JEPA loss 最低，长训稳定性和下游表现仍待验证。
- `encoding/configs/environmental_jepa.yaml` 默认用 CUDA 训练 4 轮。训练目录包含 `last.pt`、`encoder.pt` 和 `train.log`，CUDA 日志记录每轮显存峰值。Git 忽略本地输入数据、checkpoint 和训练日志。
