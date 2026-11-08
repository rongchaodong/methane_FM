#!/usr/bin/env python3
"""Download global native-grid ERA5 Daily Statistics, one year/variable at a time."""

import argparse
from pathlib import Path
import shutil
import zipfile

from process_era5_daily import DEFAULT_ROOT, GROUP_VARIABLES, inspect_files


DATASET = "derived-era5-single-levels-daily-statistics"
CDS_VARIABLES = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "msdwswrf": "mean_surface_downward_short_wave_radiation_flux",
    "swvl1": "volumetric_soil_water_layer_1",
    "tp": "total_precipitation",
}


def build_request(year, group, variable):
    return {
        "product_type": "reanalysis",
        "variable": [CDS_VARIABLES[variable]],
        "year": str(year),
        "month": [f"{month:02d}" for month in range(1, 13)],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "daily_statistic": f"daily_{group}",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
    }


def valid_zip(path):
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None and any(
                name.endswith(".nc") for name in archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def valid_netcdf(files, year, variable):
    try:
        inspect_files(files, year, [variable])
        return True
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def prepare_netcdf(path, out_dir, year, variable):
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".nc":
        path.replace(out_dir / f"{variable}_data.nc")
        return
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.endswith(".nc"):
                continue
            target = out_dir / f"{variable}_{Path(member.filename).name}"
            temporary = target.with_suffix(".nc.part")
            with archive.open(member) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            temporary.replace(target)
    inspect_files(out_dir.glob(f"{variable}_*.nc"), year, [variable])


def plan_jobs(root, start, end, no_extract=False):
    jobs = []
    for year in range(start, end + 1):
        for group, variables in GROUP_VARIABLES.items():
            folder = root / "netcdf" / group / str(year)
            for variable in variables:
                stem = root / "downloads" / group / f"ERA5_daily_{group}_{variable}_{year}"
                if valid_netcdf(folder.glob(f"{variable}_*.nc"), year, variable):
                    print(f"[SKIP] {year} {variable}: complete NetCDF")
                    continue
                source = stem.with_suffix(".nc")
                if not valid_netcdf([source], year, variable):
                    source = stem.with_suffix(".zip")
                    if not valid_zip(source):
                        source = None
                if source is None or not no_extract:
                    jobs.append((year, group, variable, source))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.start_year > args.end_year:
        parser.error("Require start-year <= end-year")

    jobs = plan_jobs(args.output_dir, args.start_year, args.end_year, args.no_extract)
    print(f"Dataset: {DATASET}; global 0.25°; UTC+00:00; 1_hourly")
    print(f"Sequential tasks: {len(jobs)}; output: {args.output_dir}")
    if args.dry_run:
        for year, group, variable, source in jobs:
            action = "download" if source is None else "prepare"
            print(f"  {year} {group} {variable}: {action}")
        return

    client = None
    for year, group, variable, source in jobs:
        if source is None:
            if client is None:
                import cdsapi
                client = cdsapi.Client()
            stem = args.output_dir / "downloads" / group / f"ERA5_daily_{group}_{variable}_{year}"
            stem.parent.mkdir(parents=True, exist_ok=True)
            temporary = stem.with_suffix(".part")
            temporary.unlink(missing_ok=True)
            print(f"[GET] {year} {group} {variable}", flush=True)
            client.retrieve(DATASET, build_request(year, group, variable), str(temporary))
            if zipfile.is_zipfile(temporary):
                if not valid_zip(temporary):
                    raise ValueError(f"Invalid ZIP: {temporary}")
                source = stem.with_suffix(".zip")
            else:
                inspect_files([temporary], year, [variable])
                source = stem.with_suffix(".nc")
            temporary.replace(source)
        if not args.no_extract:
            prepare_netcdf(source, args.output_dir / "netcdf" / group / str(year), year, variable)
        print(f"[DONE] {year} {group} {variable}", flush=True)


if __name__ == "__main__":
    main()
