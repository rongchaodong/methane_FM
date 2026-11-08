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
中断后重跑相同命令，已验证完成的数据会自动跳过。
