#!/usr/bin/env python3
"""Convert ERA5 daily units in small time blocks, preserving the native grid."""

import argparse
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_ROOT = Path("/common/users/rd1281/ERA5_daily")
VARIABLES = {
    "t2m": ("TAIR", "degree_Celsius"),
    "d2m": ("VAPR", "hPa"),
    "tp": ("PREC", "mm/day"),
    "msdwswrf": ("SOLR", "W m-2"),
    "swvl1": ("SM", "m3 m-3"),
}
GROUP_VARIABLES = {"mean": ("t2m", "d2m", "msdwswrf", "swvl1"), "sum": ("tp",)}


def calendar_for_year(year):
    return pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")


def _daily_data(dataset, variable):
    data = dataset[variable]
    time_dim = "valid_time" if "valid_time" in data.dims else "time"
    return data.transpose(time_dim, "latitude", "longitude"), time_dim


def inspect_files(files, year, required_variables):
    """Check one year's variables, daily calendar, grid and readable field ends."""
    records = {variable: {"pieces": []} for variable in required_variables}
    coordinates = None
    for path in sorted(files):
        with xr.open_dataset(path, engine="netcdf4", cache=False) as dataset:
            for variable, record in records.items():
                aliases = ("msdwswrf", "avg_sdswrf") if variable == "msdwswrf" else (variable,)
                actual = next((name for name in aliases if name in dataset), None)
                if actual is None:
                    continue
                data, time_dim = _daily_data(dataset, actual)
                dates = pd.DatetimeIndex(data[time_dim].to_numpy())
                if dates.empty:
                    raise ValueError(f"No daily dates: {path} {actual}")
                grid = {axis: data[axis].to_numpy() for axis in ("latitude", "longitude")}
                if coordinates is None:
                    coordinates = grid
                if not all(np.array_equal(grid[axis], coordinates[axis]) for axis in grid):
                    raise ValueError(f"Inconsistent source coordinates: {path} {actual}")
                data.isel({time_dim: [0, -1], "latitude": [0, -1], "longitude": [0, -1]}).to_numpy()
                record.update(coordinates)
                record["pieces"].append((path, actual, time_dim, dates))
    for variable, record in records.items():
        pieces = record["pieces"]
        pieces.sort(key=lambda piece: piece[3][0])
        dates = pd.DatetimeIndex([date for piece in pieces for date in piece[3]])
        if not dates.equals(calendar_for_year(year)):
            raise ValueError(f"Incomplete, duplicate or misplaced daily dates: {year} {variable}")
    return records


def convert_units(variable, values):
    values = values.astype(np.float64, copy=False)
    if variable == "t2m":
        return values - 273.15
    if variable == "tp":
        return values * 1000  # daily_sum is already a daily accumulation.
    if variable == "d2m":
        celsius = values - 273.15
        return 6.112 * np.exp(17.67 * celsius / (celsius + 243.5))
    return values


def valid_output(path, year, variable, source):
    name, units = VARIABLES[variable]
    try:
        with netCDF4.Dataset(path) as dataset:
            data = dataset[name]
            if data.dimensions != ("latitude", "longitude", "time"):
                return False
            time = dataset["time"]
            dates = netCDF4.num2date(time[:], time.units, calendar=time.calendar,
                                    only_use_cftime_datetimes=False)
            data[[0, -1], [0, -1], [0, -1]]
            return (dataset.data_model == "NETCDF4" and data.dtype == np.dtype("float32")
                    and data.units == units
                    and pd.DatetimeIndex(dates).equals(calendar_for_year(year))
                    and all(np.array_equal(dataset[axis][:], source[axis])
                            for axis in ("latitude", "longitude")))
    except (OSError, ValueError, KeyError, AttributeError, RuntimeError):
        return False


def _write_variable(path, year, variable, source, chunk_days):
    name, units = VARIABLES[variable]
    time = calendar_for_year(year)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as target:
        for axis, units_coord in (("latitude", "degrees_north"), ("longitude", "degrees_east")):
            values = source[axis]
            target.createDimension(axis, len(values))
            coordinate = target.createVariable(axis, values.dtype, (axis,))
            coordinate[:] = values
            coordinate.units = units_coord
        target.createDimension("time", len(time))
        coordinate = target.createVariable("time", "i4", ("time",))
        coordinate.units = "days since 1970-01-01 00:00:00"
        coordinate.calendar = "proleptic_gregorian"
        coordinate[:] = (time - pd.Timestamp("1970-01-01")).days.to_numpy()
        data_out = target.createVariable(
            name, "f4", ("latitude", "longitude", "time"), fill_value=np.float32(np.nan),
            zlib=True, complevel=3,
            chunksizes=(min(90, len(source["latitude"])),
                        min(180, len(source["longitude"])), min(chunk_days, len(time))))
        data_out.units = units
        offset = 0
        for file, actual, time_dim, dates in source["pieces"]:
            with xr.open_dataset(file, engine="netcdf4", cache=False) as dataset:
                data, _ = _daily_data(dataset, actual)
                for start in range(0, len(dates), chunk_days):
                    values = data.isel({time_dim: slice(start, start + chunk_days)}).to_numpy()
                    converted = convert_units(variable, values).astype("float32")
                    data_out[:, :, offset:offset + len(values)] = converted.transpose(1, 2, 0)
                    offset += len(values)


def process_year(year, folders, output, overwrite=False, chunk_days=4, dry_run=False):
    files = [file for group_files in folders.values() for file in group_files]
    records = inspect_files(files, year, VARIABLES)
    for variable, (name, _) in VARIABLES.items():
        filename = "soil" if name == "SM" else name
        path = output / f"daily_ERA5_{filename}_{year}.nc"
        if not overwrite and valid_output(path, year, variable, records[variable]):
            print(f"[SKIP] {path.name}", flush=True)
            continue
        if dry_run:
            print(f"[PLAN] {variable} -> {path}", flush=True)
            continue
        temporary = path.with_suffix(".nc.part")
        try:
            _write_variable(temporary, year, variable, records[variable], chunk_days)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"[DONE] {path.name}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT / "processed")
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--chunk-days", type=int, default=4,
                        help="Daily records loaded at once; no time aggregation")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.start_year > args.end_year or args.chunk_days < 1:
        parser.error("Require start-year <= end-year and chunk-days >= 1")
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
    for year in range(args.start_year, args.end_year + 1):
        folders = {group: (args.input_root / "netcdf" / group / str(year)).glob("*.nc")
                   for group in GROUP_VARIABLES}
        process_year(year, folders, args.output_root, args.overwrite, args.chunk_days, args.dry_run)


if __name__ == "__main__":
    main()
