"""Adapt simulation grids and observation tables to the existing JEPA queries."""

from bisect import bisect_right
from calendar import monthrange
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
from hashlib import sha256
import json
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from encoding.dataset import EnvironmentalCollator
from encoding.specs import QuerySpec, as_datetime, normalize_longitude


FLUX_UNIT = "mg_ch4_m2_day"
TARGET_TYPES = {"emission": 0, "uptake": 1, "net": 2}
MOLAR_MASS_CH4 = 16.0425
MOLAR_MASS_C = 12.011
ADAPTERS = {
    "tem_emission": dict(
        source="TEM", target_type="emission", period="day",
        path="TEM_EMISSION/CH4_emission_intensity_{year}.nc", variable="CH4_emission",
        axes=("latitude", "longitude", "month", "day"), spatial_support=(0.5, 0.5),
        input_unit=FLUX_UNIT, nonnegative_emission=True,
    ),
    "tem_uptake": dict(
        source="TEM", target_type="uptake", period="month",
        path="TEM_ch4_wetland_soilsink_1950_2020_mgm2day.nc4", variable="ch4_oxidation",
        axes=("month", "year", "longitude", "latitude"), spatial_support=(0.5, 0.5),
        input_unit=FLUX_UNIT,
    ),
    "memo_uptake": dict(
        source="MeMo", target_type="uptake", period="month",
        path="MeMo_1990-2009_monthly_corrected.nc", variable="CH4uptake",
        axes=("Time", "Lat", "Lon"), spatial_support=(1.0, 1.0),
        input_unit="mg_ch4_m2_month", sign=-1, month_origin="1990-01-01",
    ),
    "fluxnet": dict(
        source="FLUXNET", period="day", path="FLUXNET-CH4/FLUXNET_T1_DD.csv",
        metadata_path="FLUXNET-CH4/FLUXNET_CH4_2024.csv", target_column="FCH4_F_ANNOPTLM",
        site_column="Site", time_column="TIMESTAMP", metadata_site_column="SITE_ID",
        latitude_column="LOCATION_LAT", longitude_column="LOCATION_LONG",
        input_unit="nmol_ch4_m2_s", spatial_support=(0.0, 0.0),
        site_classes=("WET",), minimum_month_days=20, nonnegative_emission=True,
    ),
    "chamber": dict(
        source="Chamber", target_type="uptake", period="month", path="chamber_processed.csv",
        daily_column="FCH4_daily_mean", monthly_column="FCH4_monthly_processed",
        site_column="site_unique_id", latitude_column="lat", longitude_column="lon",
        year_column="FCH4_sample_year", month_column="FCH4_sample_month",
        day_column="FCH4_sample_day", study_column="dataset_no", spatial_support=(0.0, 0.0),
        input_unit=FLUX_UNIT, nonpositive_uptake=False,
    ),
}
GRID_PRODUCTS = ("tem_emission", "tem_uptake", "memo_uptake")
SPLITS = ("train", "validation", "testing")
CACHE_PROVENANCE_KEYS = ("checkpoint_sha256", "query_version", "preprocessing_version", "sampling")
DATA_QUERY_VERSION = "moe-native-grid-centers-v2"


def _period_end(start, period):
    return start + timedelta(days=1 if period == "day" else monthrange(start.year, start.month)[1])


def _date_range(value):
    start, end = (as_datetime(time) for time in value)
    if start >= end:
        raise ValueError("Data date ranges require start < end.")
    return start, end


def _conversion(unit, product):
    factors = {
        FLUX_UNIT: 1.0,
        "mg_ch4_m2_month": 1.0,
        "mg_c_m2_day": MOLAR_MASS_CH4 / MOLAR_MASS_C,
        "nmol_ch4_m2_s": MOLAR_MASS_CH4 * 86400 / 1e6,
        "ug_ch4_m2_h": 24 / 1000,
        "ug_c_m2_h": 24 / 1000 * MOLAR_MASS_CH4 / MOLAR_MASS_C,
    }
    if unit not in factors:
        raise ValueError(
            f"Set data.adapters.{product}.input_unit using verified source metadata. "
            f"Supported units: {', '.join(factors)}."
        )
    return factors[unit]


def _raw_sign(raw, options):
    if not np.isfinite(raw) or raw in options.get("missing_values", ()):
        return "missing"
    return "negative" if raw < 0 else "positive" if raw > 0 else "zero"


def _sample(query, raw_target, site, product, options, *, weight=1.0, study=""):
    factor = options["conversion"] * options.get("sign", 1)
    if options["input_unit"] == "mg_ch4_m2_month":
        factor /= (query.end_time - query.start_time).days
    target = float(raw_target) * factor
    target_type = options["target_type"]
    raw_sign = _raw_sign(raw_target, options)
    reason = None
    if raw_sign == "missing" or not np.isfinite(target):
        reason = "missing_label"
    elif not np.isfinite(weight) or weight <= 0:
        reason = "invalid_weight"
    elif product == "memo_uptake" and raw_sign == "negative":
        reason = "negative_uptake_amount"
    elif target_type == "emission" and options.get("nonnegative_emission", False) and target < 0:
        reason = "negative_emission"
    elif target_type == "uptake" and options.get("nonpositive_uptake", False) and target > 0:
        reason = "positive_uptake"
    return {
        "query": query,
        "target": target,
        "target_type": target_type,
        "source": options["source"],
        "product": product,
        "site_id": site,
        "spatial_support": (query.delta_lat, query.delta_lon),
        "temporal_support": (query.start_time, query.end_time),
        "quality_mask": reason is None,
        "raw_sign": raw_sign,
        "filter_reason": reason,
        "study_id": study,
        "sample_weight": float(weight),
    }


def _file_metadata(path, *, hash_content=False):
    stat = path.stat()
    metadata = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_content:
        metadata["sha256"] = sha256(path.read_bytes()).hexdigest()
    return metadata


def _year_files(path):
    before, after = path.name.split("{year}")
    paths = {}
    for entry in sorted(path.parent.glob(f"{before}[0-9][0-9][0-9][0-9]{after}")):
        year = int(entry.name[len(before):len(before) + 4])
        paths[year] = entry
    if not paths:
        raise FileNotFoundError(f"No yearly files matching {path}")
    return paths


class GridProduct:
    def __init__(self, product, path, options, date_range):
        from netCDF4 import Dataset as NetCDF

        self.product, self.options = product, options
        self.files = _year_files(path) if product == "tem_emission" else {None: path}
        self.dates, self.time_shapes = [], {}
        self.invalid_calendar_periods = 0
        first_path = next(iter(self.files.values()))
        with NetCDF(first_path) as file:
            variable = file[options["variable"]]
            if variable.dimensions != tuple(options["axes"]):
                raise ValueError(
                    f"{first_path}: {options['variable']} has axes {variable.dimensions}, "
                    f"expected {options['axes']}."
                )
            latitude, longitude = options["latitude_axis"], options["longitude_axis"]
            self.raw_lat = np.asarray(file[latitude][:], dtype=np.float64)
            self.raw_lon = np.asarray(file[longitude][:], dtype=np.float64)
            if self.raw_lat.ndim != 1 or self.raw_lon.ndim != 1:
                raise ValueError(f"{first_path}: expected one-dimensional latitude and longitude axes.")
            self.element_bytes = variable.dtype.itemsize
            if product == "tem_uptake":
                self.time_shapes[None] = variable.shape[:2]
                for yi, year in enumerate(map(int, file["year"][:])):
                    for mi, month in enumerate(map(int, file["month"][:])):
                        self.dates.append((datetime(year, month, 1), None, (mi, yi)))
            elif product == "memo_uptake":
                self.time_shapes[None] = variable.shape[:1]
                # The first MeMo time coordinate is 1 and represents the origin month.
                origin = as_datetime(options["month_origin"])
                for ti, offset in enumerate(map(int, file["Time"][:])):
                    years, month = divmod(origin.month + offset - 2, 12)
                    self.dates.append((datetime(origin.year + years, month + 1, 1), None, (ti,)))
        self.support = options["spatial_support"]
        self.lat, self.lon = self.raw_lat, normalize_longitude(self.raw_lon)
        if product in ("tem_emission", "tem_uptake"):
            steps = [np.diff(self.raw_lat), normalize_longitude(np.diff(self.raw_lon))]
            spacing = tuple(float(abs(step[0])) if len(step) else width
                            for step, width in zip(steps, self.support))
            if any(width <= 0 or not np.allclose(abs(step), width)
                   for step, width in zip(steps, spacing)):
                raise ValueError(f"{product}: expected regular native latitude and longitude axes.")
            self.support = spacing
            # The emission file encodes south edges. The oxidation file encodes north edges.
            lat_direction = 1 if product == "tem_emission" else -1
            self.lat = self.raw_lat + lat_direction * spacing[0] / 2
            self.lon = normalize_longitude(self.raw_lon + spacing[1] / 2)
            if np.any(np.abs(self.lat) + spacing[0] / 2 > 90 + 1e-8):
                raise ValueError(f"{product}: native grid edges extend beyond the poles.")
        start, end = date_range
        if product == "tem_emission":
            for year, year_path in self.files.items():
                if year < start.year or datetime(year, 1, 1) >= end:
                    continue
                with NetCDF(year_path) as file:
                    months, days = tuple(map(int, file["month"][:])), tuple(map(int, file["day"][:]))
                    if (file[options["variable"]].dimensions != tuple(options["axes"])
                            or not np.array_equal(file[latitude][:], self.raw_lat)
                            or not np.array_equal(file[longitude][:], self.raw_lon)):
                        raise ValueError(f"{year_path}: changed axes or grid in TEM emission series.")
                    self.time_shapes[year] = (len(months), len(days))
                for mi, month in enumerate(months):
                    for di, day in enumerate(days):
                        if 1 <= month <= 12 and 1 <= day <= monthrange(year, month)[1]:
                            self.dates.append((datetime(year, month, day), year, (mi, di)))
                        else:
                            self.invalid_calendar_periods += 1
        self.dates = [entry for entry in self.dates
                      if start <= entry[0] and _period_end(entry[0], options["period"]) <= end]
        if not self.dates:
            raise ValueError(f"{product}: no complete target periods inside the configured date range.")
        self.time_lookup = {year: np.full(shape, -1, dtype=np.int32) for year, shape in self.time_shapes.items()}
        for index, (_, year, native) in enumerate(self.dates):
            self.time_lookup[year][native] = index
        self.period_years = np.array([date.year for date, _, _ in self.dates], dtype=np.int32)

    def uptake_sign_counts(self):
        """Count the native signs over the selected dates in bounded monthly slabs."""
        from netCDF4 import Dataset as NetCDF

        counts = Counter()
        with NetCDF(self.files[None]) as file:
            variable = file[self.options["variable"]]
            for tile in self.tiles(16 * 1024 * 1024):
                if not (self.tile_layout(tile)[0] >= 0).any():
                    continue
                slab = self.read_tile(variable, tile)
                valid = np.isfinite(slab)
                for missing in self.options.get("missing_values", ()):
                    valid &= slab != missing
                counts["negative"] += int(np.count_nonzero(valid & (slab < 0)))
                counts["positive"] += int(np.count_nonzero(valid & (slab > 0)))
                counts["zero"] += int(np.count_nonzero(valid & (slab == 0)))
                counts["missing"] += slab.size - int(np.count_nonzero(valid))
        return counts

    def sample(self, time_index, grid_index, variable):
        _, _, time_indices = self.dates[time_index]
        yi, xi = divmod(grid_index, len(self.lon))
        time_axes = {"tem_emission": ("month", "day"), "tem_uptake": ("month", "year"),
                     "memo_uptake": ("Time",)}[self.product]
        selectors = dict(zip(time_axes, time_indices))
        selectors[self.options["latitude_axis"]] = yi
        selectors[self.options["longitude_axis"]] = xi
        indices = tuple(selectors[axis] for axis in self.options["axes"])
        value = variable[indices]
        raw = np.nan if np.ma.is_masked(value) else float(value)
        return self.sample_value(time_index, grid_index, raw)

    def sample_value(self, time_index, grid_index, raw):
        start = self.dates[time_index][0]
        yi, xi = divmod(grid_index, len(self.lon))
        lat, lon = self.lat[yi], self.lon[xi]
        query = QuerySpec(lat, lon, start, _period_end(start, self.options["period"]),
                          *self.support)
        return _sample(query, raw, f"grid:{lat:g}:{lon:g}", self.product, self.options)

    def tiles(self, max_bytes):
        """Describe contiguous native reads in file order."""
        cells = len(self.lat) * len(self.lon)
        for year, shape in self.time_shapes.items():
            if self.product == "tem_emission":
                size = len(self.lon) * np.prod(shape) * self.element_bytes
                step = max(1, int(max_bytes // size))
                for start in range(0, len(self.lat), step):
                    yield year, None, start, min(start + step, len(self.lat))
            else:
                step = max(1, max_bytes // (cells * self.element_bytes))
                months = range(shape[0]) if self.product == "tem_uptake" else (None,)
                for month in months:
                    for start in range(0, shape[-1], step):
                        yield year, month, start, min(start + step, shape[-1])

    def tile_layout(self, tile):
        year, month, start, stop = tile
        native = self.time_lookup[year]
        if self.product == "tem_emission":
            return native.ravel(), start * len(self.lon), (stop - start) * len(self.lon)
        periods = native[month, start:stop] if month is not None else native[start:stop]
        return periods, 0, len(self.lat) * len(self.lon)

    def read_tile(self, variable, tile):
        _, month, start, stop = tile
        periods, _, cells = self.tile_layout(tile)
        if self.product == "tem_emission":
            native = variable[start:stop]
            values = np.ma.asarray(native, dtype=variable.dtype).filled(np.nan).reshape(cells, -1).T
        elif self.product == "tem_uptake":
            native = variable[month, start:stop]
            values = np.ma.asarray(native, dtype=variable.dtype).filled(np.nan).transpose(0, 2, 1).reshape(-1, cells)
        else:
            native = variable[start:stop]
            values = np.ma.asarray(native, dtype=variable.dtype).filled(np.nan).reshape(-1, cells)
        return np.ascontiguousarray(values[periods >= 0])

    def draw(self, cells, budget, seed, periods=None):
        """Draw grid-cell/period candidates and read each native file in a single pass."""
        from netCDF4 import Dataset as NetCDF

        periods = range(len(self.dates)) if periods is None else periods
        population = len(cells) * len(periods)
        if not population or budget < 1:
            raise ValueError(f"{self.product}: each simulation split needs cells and a positive budget.")
        indices = np.random.default_rng(seed).choice(
            population, size=min(budget, population), replace=False, shuffle=False,
        )
        pairs = [(self.dates[periods[time]][1], int(cells[cell]), periods[time])
                 for time, cell in zip(*np.divmod(indices, len(cells)))]
        for year, pairs_in_file in groupby(sorted(pairs), key=lambda entry: entry[0]):
            with NetCDF(self.files[year]) as file:
                variable = file[self.options["variable"]]
                for _, cell, time in pairs_in_file:
                    yield self.sample(time, cell, variable)


def _number(value):
    if value is None or str(value).strip().lower() in {"", "na", "nan", "-9999", "-9999.0"}:
        return float("nan")
    try:
        number = float(value)
    except ValueError:
        return float("nan")
    return number if number != -9999 else float("nan")


def _valid_quality(row, filters):
    for column, rule in filters.items():
        value = row[column]
        if "allowed" in rule and value not in {str(item) for item in rule["allowed"]}:
            return False
        if "min" in rule and not _number(value) >= rule["min"]:
            return False
        if "max" in rule and not _number(value) <= rule["max"]:
            return False
    return True


def _csv_rows(path, required):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}.")
        yield from reader


def _site_key(site):
    return (site or "").strip().replace(".", "-").casefold()


def _coordinates(row, options):
    lat, lon = (_number(row[options[f"{axis}_column"]]) for axis in ("latitude", "longitude"))
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 360 else None


def _observation_rows(product, path, options, date_range, site_filter, profile, site_counts):
    start_bound, end_bound = date_range
    period = options["period"]
    filters = options.get("quality_filters", {})
    weight_column = options.get("sample_weight_column")
    required = [options["site_column"], *filters]
    if weight_column:
        required.append(weight_column)
    if product == "fluxnet":
        column = options["target_column"]
        required.extend([column, options["time_column"]])
        metadata = {
            _site_key(row[options["metadata_site_column"]]): row
            for row in _csv_rows(options["metadata_path"], [
                options["metadata_site_column"], options["latitude_column"],
                options["longitude_column"], "IGBP",
            ])
        }
    else:
        column = options["monthly_column" if period == "month" else "daily_column"]
        required.extend([column, options["latitude_column"], options["longitude_column"],
                         options["year_column"], options["month_column"]])
        if period == "day":
            required.append(options["day_column"])
        else:
            required.append(options["study_column"])
    counts = profile["input_rows"]
    for index, row in enumerate(_csv_rows(path, required)):
        if index == 0:
            profile["qc_columns"] = [key for key in row if key.startswith("FCH4")
                                     and key.endswith(("_QC", "_UNC", "_RANDUNC"))]
            profile["comparison_column"] = "FCH4_F" if product == "fluxnet" and "FCH4_F" in row else None
            profile["wetland_flag_present"] = "wetland" in row if product == "chamber" else None
        raw_site = row[options["site_column"]]
        site_key = _site_key(raw_site)
        target = _number(row[column])
        sign = _raw_sign(target, options)
        qc = Counter()
        for flag in profile["qc_columns"]:
            value = row[flag].strip() if np.isfinite(_number(row[flag])) else "missing"
            qc[f"qc:{flag}:{value}"] += 1
        if profile["comparison_column"]:
            available = np.isfinite(_number(row[profile["comparison_column"]]))
            qc["fch4_f_available" if available else "fch4_f_missing"] += 1
            if not available and sign != "missing":
                qc["annoptlm_available_when_fch4_f_missing"] += 1
        counts.update({"raw": 1, f"raw_{sign}": 1})
        counts.update(qc)
        if not site_key:
            counts["excluded_missing_site"] += 1
            continue
        if site_filter is not None and site_key not in site_filter:
            counts["excluded_site_selection"] += 1
            continue
        if product == "fluxnet":
            site_metadata = metadata.get(site_key)
            if not site_metadata:
                counts["excluded_unmatched_metadata"] += 1
                continue
            if options["site_classes"] and site_metadata["IGBP"] not in options["site_classes"]:
                counts["excluded_site_class"] += 1
                continue
            coords = _coordinates(site_metadata, options)
            try:
                date = as_datetime(row[options["time_column"]])
            except (TypeError, ValueError):
                counts["excluded_invalid_date"] += 1
                continue
            site = site_metadata[options["metadata_site_column"]]
            study = ""
        else:
            if profile["wetland_flag_present"] and row["wetland"].strip().casefold() != "no":
                counts["excluded_not_upland"] += 1
                continue
            coords = _coordinates(row, options)
            year, month = (_number(row[options[f"{part}_column"]]) for part in ("year", "month"))
            day = _number(row[options["day_column"]]) if period == "day" else 1
            if not all(np.isfinite(value) and float(value).is_integer() for value in (year, month, day)):
                counts["excluded_invalid_date"] += 1
                continue
            try:
                date = datetime(int(year), int(month), int(day))
            except ValueError:
                counts["excluded_invalid_date"] += 1
                continue
            site = raw_site.strip()
            study = (row.get(options["study_column"]) or "").strip()
        if coords is None:
            counts["excluded_invalid_coordinates"] += 1
            continue
        period_start = date.replace(day=1) if period == "month" else date
        if not (start_bound <= period_start and _period_end(period_start, period) <= end_bound):
            counts["excluded_date_range"] += 1
            continue
        site_count = site_counts[site_key, period_start.isoformat()]
        counts["scoped"] += 1
        site_count.update({"raw": 1, f"raw_{sign}": 1})
        site_count.update(qc)
        weight = _number(row[weight_column]) if weight_column else 1.0
        if sign == "missing":
            site_count["filtered_missing_label"] += 1
        elif not _valid_quality(row, filters):
            site_count["filtered_quality_rule"] += 1
        elif not np.isfinite(weight) or weight <= 0:
            site_count["filtered_invalid_weight"] += 1
        else:
            yield site, study, coords, date, target, weight


def _observation_samples(product, path, options, date_range, site_filter):
    profile = {"input_rows": Counter(), "qc_columns": [], "comparison_column": None}
    site_counts = defaultdict(Counter)
    rows = _observation_rows(product, path, options, date_range, site_filter, profile, site_counts)
    period = options["period"]
    if period == "month":
        groups = defaultdict(dict)
        for site, study, coords, date, target, weight in rows:
            month = date.replace(day=1)
            group = groups[site, study, coords, month]
            if date in group:
                existing_target, _ = group[date]
                if not np.isclose(existing_target, target):
                    raise ValueError(f"{product}: conflicting observations at {site}, {date:%Y-%m-%d}.")
                site_counts[_site_key(site), month.isoformat()]["repeated_rows"] += 1
                continue
            group[date] = target, weight
        monthly = []
        for (site, study, coords, month), days in groups.items():
            site_count = site_counts[_site_key(site), month.isoformat()]
            if product == "fluxnet" and len(days) < options["minimum_month_days"]:
                site_count["filtered_incomplete_month"] += len(days)
                continue
            targets, weights = zip(*days.values())
            site_count["aggregated_rows"] += len(days) - 1
            # FLUXNET averages daily rates. Chamber uses the supplied monthly rate.
            monthly.append((site, study, coords, month, np.mean(targets), np.mean(weights)))
        rows = monthly
    samples = []
    for site, study, coords, date, target, weight in rows:
        query = QuerySpec(*coords, date, _period_end(date, period), *options["spatial_support"])
        sample = _sample(query, target, site, product, options, weight=weight, study=study)
        site_count = site_counts[_site_key(site), date.isoformat()]
        site_count["label_candidates"] += 1
        if sample["quality_mask"]:
            samples.append(sample)
        else:
            site_count[f'filtered_{sample["filter_reason"]}'] += 1
    return samples, profile, site_counts


class MoEDatasetAdapter(Dataset):
    def __init__(self, segments, provenance):
        self.segments = segments
        self.ends = np.cumsum([len(segment) for segment in segments]).tolist()
        self.provenance = provenance
        self.streaming = any(hasattr(segment, "shard_lengths") for segment in segments)
        self.shard_lengths = [length for segment in segments
                              for length in getattr(segment, "shard_lengths", [len(segment)]) if length]

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        segment = bisect_right(self.ends, index)
        offset = self.ends[segment - 1] if segment else 0
        return self.segments[segment][index - offset]

    def close(self):
        for segment in self.segments:
            if hasattr(segment, "close"):
                segment.close()


class ShardBatchSampler(Sampler):
    """Shuffle native shards and contiguous batch runs, then resume by batch ordinal."""

    shuffle_run_batches = 128

    def __init__(self, lengths, batch_size, *, shuffle, seed):
        self.lengths = lengths
        self.starts = np.cumsum([0, *lengths[:-1]], dtype=np.int64).tolist()
        self.batches = [(length + batch_size - 1) // batch_size for length in lengths]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_batches = sum(self.batches)
        self.set_progress(seed)

    def set_progress(self, seed, completed_batches=0):
        if not 0 <= completed_batches <= self.total_batches:
            raise ValueError("Completed batches are outside the full data pass.")
        self.seed, self.completed_batches = seed, completed_batches

    def __len__(self):
        return self.total_batches - self.completed_batches

    def __iter__(self):
        seed, remaining = self.seed, self.completed_batches
        shards = range(len(self.lengths))
        if self.shuffle:
            shards = np.random.default_rng(seed).permutation(len(self.lengths))
        for shard in shards:
            count = self.batches[shard]
            if remaining >= count:
                remaining -= count
                continue
            run_size = self.shuffle_run_batches if self.shuffle else count
            runs = range((count + run_size - 1) // run_size)
            if self.shuffle:
                rng = np.random.default_rng(np.random.SeedSequence([seed, int(shard), 1]))
                runs = rng.permutation(len(runs))
            start, stop = self.starts[shard], self.starts[shard] + self.lengths[shard]
            for run in runs:
                first_batch = int(run) * run_size
                last_batch = min(count, first_batch + run_size)
                if remaining >= last_batch - first_batch:
                    remaining -= last_batch - first_batch
                    continue
                for batch in range(first_batch + remaining, last_batch):
                    offset = start + batch * self.batch_size
                    yield list(range(offset, min(stop, offset + self.batch_size)))
                remaining = 0


def cache_query_key(query):
    """Create a lossless cache key for a QuerySpec, independent of the supervision source."""
    return json.dumps([
        float(query.lat), float(query.lon), query.start_time.isoformat(), query.end_time.isoformat(),
        float(query.delta_lat), float(query.delta_lon),
    ], separators=(",", ":"))


def save_embedding_cache(path, entries, encoder_metadata, *, gather_mode="history", simulation_daily_start="1979-01-01"):
    """Save supplied (QuerySpec, latent) pairs for use in cached mode."""
    if gather_mode not in {"history", "query_window"}:
        raise ValueError("Embedding cache gather_mode must be history or query_window.")
    metadata = {key: encoder_metadata[key] for key in CACHE_PROVENANCE_KEYS}
    metadata["gather_mode"] = gather_mode
    metadata["data_query_version"] = DATA_QUERY_VERSION
    metadata["simulation_daily_start"] = as_datetime(simulation_daily_start).isoformat()
    rows = {cache_query_key(query): latent.detach().cpu().clone() for query, latent in entries}
    torch.save({"metadata": metadata, "rows": rows}, path)


class MoECollator:
    def __init__(self, *, gatherer=None, gather_mode="history", cache=None, latent_dim=128,
                 simulation_daily_start="1979-01-01"):
        self.gatherer, self.gather_mode = gatherer, gather_mode
        self.cache, self.latent_dim = cache, latent_dim
        self.environmental_collator = EnvironmentalCollator(gatherer.registry) if gatherer else None
        self.simulation_daily_start = as_datetime(simulation_daily_start)

    def __call__(self, samples):
        samples = [sample for sample in samples if sample["quality_mask"]]
        if not samples:
            return None
        batch = {}
        if self.cache is not None:
            latents = []
            for sample in samples:
                key = cache_query_key(sample["query"])
                if key not in self.cache:
                    raise KeyError(f"Missing cached embedding for query {key}.")
                latent = self.cache[key]
                if latent.shape != (self.latent_dim,) or not latent.is_floating_point() or not torch.isfinite(latent).all():
                    raise ValueError(f"Invalid cached embedding for query {key}.")
                latents.append(latent)
            batch["latent"] = torch.stack(latents).float()
        else:
            kept, contexts = [], []
            for sample in samples:
                query = sample["query"]
                window = ({"start_time": query.start_time, "end_time": query.end_time}
                          if self.gather_mode == "query_window" else {})
                if sample["product"] in GRID_PRODUCTS:
                    context = self.gatherer.gather(query, daily_start=self.simulation_daily_start, **window)
                    if not context.blocks:
                        raise ValueError(f"No environmental tokens for valid simulation label {query}.")
                    contexts.append(context)
                    kept.append(sample)
                    continue
                context = self.gatherer.gather(query, **window)
                if any(self.gatherer.registry.sources[block.source].frequency == "daily" for block in context.blocks):
                    contexts.append(context)
                    kept.append(sample)
            if not contexts:
                return None
            samples = kept
            batch["environment"] = self.environmental_collator.collate_contexts(contexts)
        for key in ("query", "source", "product", "site_id", "spatial_support", "temporal_support"):
            batch[key] = [sample[key] for sample in samples]
        batch["target"] = torch.tensor([sample["target"] for sample in samples], dtype=torch.float32)
        batch["target_type"] = torch.tensor([TARGET_TYPES[sample["target_type"]] for sample in samples])
        batch["quality_mask"] = torch.tensor([sample["quality_mask"] for sample in samples])
        batch["sample_weight"] = torch.tensor([sample["sample_weight"] for sample in samples], dtype=torch.float32)
        return batch


def _collator(config, gatherer, encoder_metadata):
    mode = config.get("embedding_mode", "online")
    gather_mode = config.get("gather_mode", "history")
    if gather_mode not in {"history", "query_window"}:
        raise ValueError("data.gather_mode must be history or query_window.")
    if mode == "online" and gatherer is not None:
        return MoECollator(gatherer=gatherer, gather_mode=gather_mode,
                           simulation_daily_start=config.get("simulation_daily_start", "1979-01-01"))
    if mode != "cached" or not config.get("cache_path") or not encoder_metadata:
        raise ValueError(
            "Online mode requires a JEPA gatherer, and cached mode requires cache_path and encoder provenance."
        )
    cache = torch.load(config["cache_path"], map_location="cpu", weights_only=True)
    metadata = cache["metadata"]
    for key in CACHE_PROVENANCE_KEYS:
        if key not in encoder_metadata or key not in metadata or metadata[key] != encoder_metadata[key]:
            raise ValueError(f"Cached embedding provenance differs from the configured JEPA encoder: {key}.")
    if metadata.get("gather_mode") != gather_mode:
        raise ValueError("Cached embedding gather_mode differs from the configured JEPA query mode.")
    if metadata.get("data_query_version") != DATA_QUERY_VERSION:
        raise ValueError("Embedding cache predates the corrected grid geometry. Rebuild it using the current queries.")
    if metadata.get("simulation_daily_start") != as_datetime(config.get("simulation_daily_start", "1979-01-01")).isoformat():
        raise ValueError("Cached embedding daily driver coverage differs from the configured simulation context.")
    return MoECollator(cache=cache["rows"], latent_dim=encoder_metadata["model_config"]["latent_dim"])


def product_names(sources, *, grid=None):
    sources = set(sources)
    known = set(ADAPTERS) | {options["source"] for options in ADAPTERS.values()}
    if not sources or sources - known:
        raise ValueError(f"Unrecognized data sources: {sorted(sources - known)}.")
    names = [name for name, options in ADAPTERS.items() if sources.intersection({name, options["source"]})]
    if grid is not None and any((name in GRID_PRODUCTS) != grid for name in names):
        raise ValueError("Pretraining requires simulation products. Observation cohorts require observation products.")
    return names


def product_inputs(config, names):
    products = {}
    for product in names:
        defaults = ADAPTERS[product]
        options = {**defaults, **config.get("adapters", {}).get(product, {})}
        options["conversion"] = _conversion(options.get("input_unit"), product)
        if options["period"] not in {"day", "month"}:
            raise ValueError(f"{product}: period must be day or month.")
        if options["input_unit"] == "mg_ch4_m2_month" and options["period"] != "month":
            raise ValueError(f"{product}: monthly totals require monthly supervision.")
        if options.get("sign", 1) not in {-1, 1}:
            raise ValueError(f"{product}: sign must be 1 or -1.")
        if product == "tem_emission":
            if options["input_unit"] != FLUX_UNIT or options.get("sign", 1) != 1:
                raise ValueError("TEM daily emission requires its native mgCH4/m2/day and sign.")
            options["nonnegative_emission"] = True
        if product == "tem_uptake" and options["input_unit"] != FLUX_UNIT:
            raise ValueError("TEM monthly uptake requires its native mgCH4/m2/day.")
        if product == "memo_uptake" and (options["input_unit"] != "mg_ch4_m2_month" or options["sign"] != -1):
            raise ValueError("MeMo requires monthly CH4 totals with positive uptake amounts.")
        if product in GRID_PRODUCTS:
            base = Path(config.get("simulation_dir", "data/methane_data/simulations"))
            path = base / options["path"]
            date_bounds = (options.get("date_range") or config.get("simulation_date_range")
                           or ["1979-01-01", "9999-01-01"])
            days = _date_range(date_bounds)
            options.setdefault("latitude_axis", "Lat" if product == "memo_uptake" else "latitude")
            options.setdefault("longitude_axis", "Lon" if product == "memo_uptake" else "longitude")
            if options["period"] != defaults["period"]:
                raise ValueError(f"{product}: use the native simulation period {defaults['period']}.")
            files = _year_files(path).values() if product == "tem_emission" else [path]
            version = [_file_metadata(entry) for entry in files]
        else:
            base = Path(config.get("observations_dir", "data/methane_data/observations"))
            path = base / options["path"]
            days = _date_range(options.get("date_range", config.get(
                "observation_date_range", ["1980-01-01", "2026-01-01"],
            )))
            version = [_file_metadata(path, hash_content=True)]
            if product == "fluxnet":
                metadata_path = base / options["metadata_path"]
                version.append(_file_metadata(metadata_path, hash_content=True))
                options["metadata_path"] = str(metadata_path.resolve())
                options["target_type"] = config.get("fluxnet_target_type", "emission")
                if options["target_type"] not in {"emission", "net"}:
                    raise ValueError("fluxnet_target_type must be emission or net.")
        products[product] = {
            "options": options, "files": version, "path": str(path.resolve()),
            "date_range": [day.isoformat() for day in days],
        }
    return products


def build_loaders(config, partitions, *, batch_size, num_workers=0, training_seed=42,
                  gatherer=None, encoder_metadata=None):
    """Build only training and validation loaders from frozen supervision records."""
    if config.get("embedding_mode") == "cached" and any(
        hasattr(segment, "shard_lengths") for segments, _ in partitions for segment in segments
    ):
        raise ValueError("Full simulation uses streamed online inputs; the in-memory cache is for small datasets.")
    collator = _collator(config, gatherer, encoder_metadata)
    loaders = []
    for index, (split, (segments, provenance)) in enumerate(zip(SPLITS[:2], partitions, strict=True)):
        if provenance["split"] != split:
            raise ValueError(f"Fitting requires the {split} partition.")
        dataset = MoEDatasetAdapter(segments, provenance)
        generator = torch.Generator().manual_seed(training_seed + index)
        common = dict(num_workers=num_workers, collate_fn=collator, generator=generator)
        if dataset.streaming:
            sampler = ShardBatchSampler(dataset.shard_lengths, batch_size, shuffle=index == 0,
                                        seed=training_seed + index)
            dataset.provenance["traversal"] = {"version": "native_shards_v1", "batch_size": batch_size,
                                               "shuffle_run_batches": sampler.shuffle_run_batches}
            loaders.append(DataLoader(dataset, batch_sampler=sampler, persistent_workers=num_workers > 0, **common))
        else:
            loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=index == 0 and len(dataset) > 0,
                                      **common))
    return tuple(loaders)
