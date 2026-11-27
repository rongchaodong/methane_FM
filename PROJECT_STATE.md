# 项目状态

- 已完成 ERA5 Daily 下载与单位转换脚本，属于数据准备的第一步；入口及命令见 README。下载器支持 `--variable` 指定单变量；仅在启动时检查完成状态，并发任务需按年份或变量错开。
- 关键约定：1979–2025、UTC 逐日，每次请求一个变量的一整年，顺序下载；`tp` 用 Sum，其余用 Mean。保留原始 0.25° 格点；PREC 仅乘 1000，VAPR 保持原 Magnus 公式。
- 验证：用户网页实测 cost 限额 400、两个变量 760；1979 年 t2m 已实际下载为 NetCDF4（365 天、721×1440），整场分块读取通过。下载器兼容直接 NetCDF 与 ZIP，按真实格式保存及归位。
- 当前（2026-11-14）：全部 235 个年份/变量任务通过完整日历、统一原始网格及首尾格点读取检查，处理 dry-run 全部通过；尚未生成 processed 文件。未逐格扫描全部数据。用户已停止 download2；重复下载留下的 `.part` 不影响有效原始文件。
- 当前（2026-11-20）：`ERA5_daily/processed/` 已有 1979–2025 的 5 个变量年度 NetCDF，真实配置读取 TAIR、VAPR、PREC、SOLR、SM。土壤改用 `Drivers/` 下 6 份原生 5,000 m SoilGrids GeoTIFF，按 Interrupted Goode Homolosine 网格读取窗口，保留原始值和缺测 mask。地形与分类仍用 0.5° NetCDF。全球 CH₄ 与 CO₂ 年度浓度使用最近完整年，并以公历年结束近似可用日期。
- `encoding/` 按 Query 的矩形宽度分别选择各来源原生格点，使用简化的坐标条件 Cross-Attention。JEPA Context 读取历史，Target 可配置日、月与不同矩形尺寸。时间区间采用 UTC 左闭右开，Target 排除静态和全球浓度。
- 原 `centers` 采样的四地点试验已验证 CPU 训练、恢复和 Encoder 加载，以及 CUDA FP32 训练。该 run 的第 4 轮验证 `jepa_loss` 最低，第 3 轮含方差惩罚的验证总 `loss` 最低。
- 2026-11-22 起默认按正面积相交读取各来源，矩形每轴完整宽度最多 2°，产品 token 预算为 5,120。在 `(40.5,-93.5)` 的 1° 区域，ERA5、粗静态和土壤分别选中 25、9 和 414 格，14 天 Context 有 773 个有效 tokens。第 4 轮 Encoder 已用真实数据验证日期线边缘像元及 2° 整月推理。新采样下的验证损失和下游精度待评估。
- `encoding/configs/environmental_jepa.yaml` 默认用 CUDA 训练 4 轮。训练目录包含 `last.pt`、`encoder.pt` 和 `train.log`，CUDA 日志记录每轮显存峰值。Git 忽略本地输入数据、checkpoint 和训练日志。
- `moe/` 使用 `soil_5km_trial_4/encoder.pt`。Task 2 的模拟、单阶段观测微调与候选适配全程冻结 Encoder。MoE 保留四个初始残差专家、独立 Dense Router 和有符号 Emission、Uptake 输出。正式 MoE 训练待启动。
- 2026-11-24 已生成 v2 模拟和初始观测清单 `moe/runs/splits-v2/`。正式模拟从 1979 年起读取全部有效标签，各产品末段整年验证，合计 Training 722,125,611、Validation 180,112,662。位图和原 NetCDF 分块支持完整 epoch 中断恢复。站内时间划分为 FLUXNET 23,954 / 7,982 / 7,982、Chamber 4,313 / 1,247 / 1,276。固定清单约束后续 cohort 与 Replay 的训练和留出角色。
- TEM 日排放使用南边界加半格，月氧化使用北边界减半格。真实 TEM 原索引、Query 和 ERA5 交集已核对。Chamber 直接读取既有月值并保留有效零。旧清单和旧 embedding cache 由版本检查隔离。数据规则和重建命令见 `moe/README.md`。
- Task 1 全量位图约 399 MB。正式 batch size 8 下有约 9,027 万训练 batch 和 2,251 万验证 batch。
- 2026-11-24 Task 2 使用全量 Training 的流式矩。Emission 的 `mean=131.356484, std=324.594291`，Uptake 的 `mean=-0.595501, std=2.062101`。微调与 continual 继承标签尺度，默认分别关闭 Replay、蒸馏、源校准与正则。默认 MSE、两任务权重 1.0、固定温度 1.0。模型选择用固定 Validation 方案下的加权归一化 MSE，完整 epoch 仍报告全部 Validation。
- TEM 日排放占 96.15% Training 标签。batch size 8 的全量批次中，排放 86,794,659 批，Uptake 3,471,644 批。任务频率维持原序列，训练日志分别记录样本、optimizer 步数和共享梯度。TEM 月氧化的旧符号统计覆盖 1950–2020 全文件，v2 覆盖 1979–2020，已用原文件重算核对。
- Task 2 增加按年分片的 FP32 / FP16 Embedding Cache 与续写游标，固定 Query 允许跨标签版本复用。checkpoint v2 保存完整拓扑、归一化、有效配置、优化器、RNG 和入口 Teacher。旧 MoE checkpoint 需按当前 schema 起训，已有 JEPA 与 Task 1 清单可复用。
- 真实 JEPA 小规模验收覆盖 CPU、RTX 5000 Ada、观测微调和候选接纳。Task 1 位图上的 192 条 Training 和 96 条 Validation 小基准测得 CPU 在线约 114、缓存约 1,104 标签/秒，GPU 在线约 117、缓存约 851 标签/秒。基准缓存位于 `/tmp`。CUDA BF16 通过，FP16 在调试数据上跳过 3 次溢出更新。全量缓存的存储与吞吐仍需验证。Growth 阈值待 Task 3 按归一化尺度标定。
