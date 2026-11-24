"""Adapt simulation grids and observation tables to the existing JEPA queries."""

from bisect import bisect_right
from calendar import monthrange
from collections import OrderedDict, defaultdict
import csv
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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
        source="Chamber", target_type="uptake", period="day", path="chamber_processed.csv",
        daily_column="FCH4_daily_mean", monthly_column="FCH4_monthly_processed",
        site_column="site_unique_id", latitude_column="lat", longitude_column="lon",
        year_column="FCH4_sample_year", month_column="FCH4_sample_month",
        day_column="FCH4_sample_day", study_column="dataset_no", spatial_support=(0.0, 0.0),
        nonpositive_uptake=True,
    ),
}
GRID_PRODUCTS = ("tem_emission", "tem_uptake", "memo_uptake")
CACHE_PROVENANCE_KEYS = ("checkpoint_sha256", "query_version", "preprocessing_version", "sampling")


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


def _sample(query, raw_target, site, product, options, *, quality=True, weight=1.0):
    factor = options["conversion"] * options.get("sign", 1)
    if options["input_unit"] == "mg_ch4_m2_month":
        factor /= (query.end_time - query.start_time).days
    target = float(raw_target) * factor
    target_type = options["target_type"]
    quality = (quality and np.isfinite(target) and np.isfinite(weight) and weight > 0
               and raw_target not in options.get("missing_values", ()))
    if target_type == "emission" and options.get("nonnegative_emission", False):
        quality = quality and target >= 0
    if target_type == "uptake" and options.get("nonpositive_uptake", False):
        quality = quality and target <= 0
    return {
        "query": query,
        "target": target,
        "target_type": target_type,
        "source": options["source"],
        "product": product,
        "site_id": site,
        "spatial_support": (query.delta_lat, query.delta_lon),
        "temporal_support": (query.start_time, query.end_time),
        "quality_mask": bool(quality),
        "sample_weight": float(weight),
    }


def _file_metadata(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


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
        self.provenance = [_file_metadata(path) for path in self.files.values()]
        self.dates = []
        first_path = next(iter(self.files.values()))
        with NetCDF(first_path) as file:
            variable = file[options["variable"]]
            if variable.dimensions != tuple(options["axes"]):
                raise ValueError(
                    f"{first_path}: {options['variable']} has axes {variable.dimensions}, "
                    f"expected {options['axes']}."
                )
            latitude, longitude = options["latitude_axis"], options["longitude_axis"]
            self.lat = np.asarray(file[latitude][:], dtype=np.float64)
            self.lon = normalize_longitude(np.asarray(file[longitude][:], dtype=np.float64))
            if self.lat.ndim != 1 or self.lon.ndim != 1:
                raise ValueError(f"{first_path}: expected one-dimensional latitude and longitude axes.")
            if product == "tem_emission":
                months, days = tuple(map(int, file["month"][:])), tuple(map(int, file["day"][:]))
                for year in self.files:
                    for mi, month in enumerate(months):
                        for di, day in enumerate(days):
                            if day <= monthrange(year, month)[1]:
                                self.dates.append((datetime(year, month, day), year, (mi, di)))
            elif product == "tem_uptake":
                for yi, year in enumerate(map(int, file["year"][:])):
                    for mi, month in enumerate(map(int, file["month"][:])):
                        self.dates.append((datetime(year, month, 1), None, (mi, yi)))
            else:
                # MeMo 的 Time 取值从 1 开始，第一项对应 1990 年 1 月。
                origin = as_datetime(options["month_origin"])
                for ti, offset in enumerate(map(int, file["Time"][:])):
                    years, month = divmod(origin.month + offset - 2, 12)
                    self.dates.append((datetime(origin.year + years, month + 1, 1), None, (ti,)))
        start, end = date_range
        self.dates = [entry for entry in self.dates
                      if start <= entry[0] and _period_end(entry[0], options["period"]) <= end]
        if not self.dates:
            raise ValueError(f"{product}: no complete target periods inside the configured date range.")

    def sample(self, time_index, grid_index, reader):
        start, year, time_indices = self.dates[time_index]
        yi, xi = divmod(grid_index, len(self.lon))
        time_axes = {"tem_emission": ("month", "day"), "tem_uptake": ("month", "year"),
                     "memo_uptake": ("Time",)}[self.product]
        selectors = dict(zip(time_axes, time_indices))
        selectors[self.options["latitude_axis"]] = yi
        selectors[self.options["longitude_axis"]] = xi
        indices = tuple(selectors[axis] for axis in self.options["axes"])
        value = reader(self.files[year])[self.options["variable"]][indices]
        raw = np.nan if np.ma.is_masked(value) else float(value)
        lat, lon = self.lat[yi], self.lon[xi]
        query = QuerySpec(lat, lon, start, _period_end(start, self.options["period"]),
                          *self.options["spatial_support"])
        return _sample(query, raw, f"grid:{lat:g}:{lon:g}", self.product, self.options)


def _grid_split(grid, *, block_degrees, seed, validation_fraction):
    if not 0.5 <= block_degrees < 180:
        raise ValueError("simulation_spatial_block_degrees must be at least 0.5 and below 180.")
    ny, nx = int(np.ceil(180 / block_degrees)), int(np.ceil(360 / block_degrees))
    held_out = np.random.default_rng(seed).random((ny, nx)) < validation_fraction
    yi = np.minimum(((grid.lat + 90) / block_degrees).astype(int), ny - 1)
    xi = ((grid.lon + 180) / block_degrees).astype(int)
    is_validation = held_out[yi[:, None], xi[None, :]].ravel()
    return np.flatnonzero(~is_validation), np.flatnonzero(is_validation)


class GridSamples(Dataset):
    def __init__(self, grid, cells, budget, seed):
        self.grid, self.cells = grid, cells
        population = len(cells) * len(grid.dates)
        if population == 0 or budget < 1:
            raise ValueError(f"{grid.product}: each simulation split needs cells and a positive sample budget.")
        self.indices = np.random.default_rng(seed).choice(
            population, size=min(budget, population), replace=False, shuffle=False,
        )
        self.handles = OrderedDict()
        self.pid = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        time_index, cell_index = divmod(int(self.indices[index]), len(self.cells))
        return self.grid.sample(time_index, int(self.cells[cell_index]), self._open)

    def _open(self, path):
        from netCDF4 import Dataset as NetCDF

        # 每个 worker 单独管理文件句柄，监督标签按样本读取。
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        if path not in self.handles:
            self.handles[path] = NetCDF(path)
            if len(self.handles) > 4:
                self.handles.popitem(last=False)[1].close()
        self.handles.move_to_end(path)
        return self.handles[path]

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


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


def _observation_rows(product, path, options, date_range, site_filter):
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
    for row in _csv_rows(path, required):
        raw_site = row[options["site_column"]]
        site_key = _site_key(raw_site)
        if not site_key or (site_filter is not None and site_key not in site_filter):
            continue
        if product == "fluxnet":
            site_metadata = metadata.get(site_key)
            if not site_metadata:
                continue
            if options["site_classes"] and site_metadata["IGBP"] not in options["site_classes"]:
                continue
            coords = _coordinates(site_metadata, options)
            date = row[options["time_column"]]
            if not date or date == "NA":
                continue
            date = as_datetime(date)
            site = site_metadata[options["metadata_site_column"]]
            study = ""
        else:
            coords = _coordinates(row, options)
            year, month = (_number(row[options[f"{part}_column"]]) for part in ("year", "month"))
            day = _number(row[options["day_column"]]) if period == "day" else 1
            if not all(np.isfinite(value) and float(value).is_integer() for value in (year, month, day)):
                continue
            try:
                date = datetime(int(year), int(month), int(day))
            except ValueError:
                continue
            site = raw_site.strip()
            study = row[options["study_column"]] if period == "month" else ""
        if coords is None:
            continue
        if period == "month" and product == "chamber":
            date = date.replace(day=1)
        if not start_bound <= date < end_bound:
            continue
        target = _number(row[column])
        weight = _number(row[weight_column]) if weight_column else 1.0
        quality = (_valid_quality(row, filters) and np.isfinite(target) and np.isfinite(weight)
                   and weight > 0 and target not in options.get("missing_values", ()))
        yield site, study, coords, date, target, weight, quality


def _observation_samples(product, path, options, date_range, site_filter):
    rows = _observation_rows(product, path, options, date_range, site_filter)
    period = options["period"]
    end_bound = date_range[1]
    if period == "month":
        groups = defaultdict(dict)
        for site, study, coords, date, target, weight, quality in rows:
            if not quality:
                continue
            month = date.replace(day=1)
            if month < date_range[0] or _period_end(month, period) > end_bound:
                continue
            group = groups[site, study, coords, month]
            if date in group:
                existing_target, _ = group[date]
                if not np.isclose(existing_target, target):
                    raise ValueError(f"{product}: conflicting observations at {site}, {date:%Y-%m-%d}.")
                continue
            group[date] = target, weight
        monthly = []
        for (site, study, coords, month), days in groups.items():
            if product == "fluxnet" and len(days) < options["minimum_month_days"]:
                continue
            targets, weights = zip(*days.values())
            # FLUXNET 日平均等时长，按日平均为月通量率。Chamber 使用文件中的月聚合字段。
            monthly.append((site, study, coords, month, np.mean(targets), np.mean(weights), True))
        rows = monthly
    samples = []
    for site, _, coords, date, target, weight, quality in rows:
        end = _period_end(date, period)
        if end <= end_bound:
            query = QuerySpec(*coords, date, end, *options["spatial_support"])
            samples.append(_sample(query, target, site, product, options, quality=quality, weight=weight))
    return samples


class MoEDatasetAdapter(Dataset):
    def __init__(self, segments, provenance):
        self.segments = segments
        self.ends = np.cumsum([len(segment) for segment in segments]).tolist()
        self.provenance = provenance

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        segment = bisect_right(self.ends, index)
        offset = self.ends[segment - 1] if segment else 0
        return self.segments[segment][index - offset]


def cache_query_key(query):
    """Create a lossless cache key for a QuerySpec, independent of the supervision source."""
    return json.dumps([
        float(query.lat), float(query.lon), query.start_time.isoformat(), query.end_time.isoformat(),
        float(query.delta_lat), float(query.delta_lon),
    ], separators=(",", ":"))


def save_embedding_cache(path, entries, encoder_metadata, *, gather_mode="history"):
    """Save supplied (QuerySpec, latent) pairs for use in cached mode."""
    if gather_mode not in {"history", "query_window"}:
        raise ValueError("Embedding cache gather_mode must be history or query_window.")
    metadata = {key: encoder_metadata[key] for key in CACHE_PROVENANCE_KEYS}
    metadata["gather_mode"] = gather_mode
    rows = {cache_query_key(query): latent.detach().cpu().clone() for query, latent in entries}
    torch.save({"metadata": metadata, "rows": rows}, path)


class MoECollator:
    def __init__(self, *, gatherer=None, gather_mode="history", cache=None, latent_dim=128):
        self.gatherer, self.gather_mode = gatherer, gather_mode
        self.cache, self.latent_dim = cache, latent_dim
        self.environmental_collator = EnvironmentalCollator(gatherer.registry) if gatherer else None

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
                context = self.gatherer.gather(query, **window)
                if any(self.gatherer.registry.sources[block.source].frequency == "daily"
                       for block in context.blocks):
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


def _observation_split(rows, validation_sites, validation_fraction, seed):
    sites_by_product = [
        {_site_key(row["site_id"]) for row in samples if row["quality_mask"]}
        for samples in rows
    ]
    all_sites = set().union(*sites_by_product)
    if validation_sites is not None:
        held_out = {_site_key(site) for site in validation_sites}
        unknown = held_out - all_sites
        if unknown:
            raise ValueError(f"Observation validation sites have no valid labels: {sorted(unknown)}.")
        return held_out
    if len(all_sites) < 2:
        return set()
    ordered = sorted(all_sites)
    target_count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    rng = np.random.default_rng(seed)
    best_sites, best_score = set(), -1
    for _ in range(32):
        held_out = set(rng.choice(ordered, target_count, replace=False).tolist())
        score = sum(bool(sites & held_out) + bool(sites - held_out) for sites in sites_by_product)
        if score > best_score:
            best_sites, best_score = held_out, score
        if score == 2 * len(sites_by_product):
            break
    return best_sites


def _collator(config, gatherer, encoder_metadata):
    mode = config.get("embedding_mode", "online")
    gather_mode = config.get("gather_mode", "history")
    if gather_mode not in {"history", "query_window"}:
        raise ValueError("data.gather_mode must be history or query_window.")
    if mode == "online" and gatherer is not None:
        return MoECollator(gatherer=gatherer, gather_mode=gather_mode)
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
    return MoECollator(cache=cache["rows"], latent_dim=encoder_metadata["model_config"]["latent_dim"])


def build_loaders(data_config, sources, *, batch_size, num_workers=0, seed=42,
                  gatherer=None, encoder_metadata=None, stage=None):
    """Construct lazy simulation or site-grouped observation DataLoaders."""
    config = data_config
    sources = set(sources)
    known = set(ADAPTERS) | {options["source"] for options in ADAPTERS.values()}
    if not sources or sources - known:
        raise ValueError(f"Unrecognized data sources: {sorted(sources - known)}.")
    validation_fraction = float(config.get("validation_fraction", 0.2))
    if not 0 < validation_fraction < 1:
        raise ValueError("data.validation_fraction must be between zero and one.")
    train, validation, observations, products = [], [], [], {}
    observation_names = []
    site_filter = config.get("observation_site_ids")
    site_filter = {_site_key(site) for site in site_filter} if site_filter is not None else None
    for index, (product, defaults) in enumerate(ADAPTERS.items()):
        if not sources.intersection({product, defaults["source"]}):
            continue
        options = {**defaults, **config.get("adapters", {}).get(product, {})}
        options["conversion"] = _conversion(options.get("input_unit"), product)
        if options["period"] not in {"day", "month"}:
            raise ValueError(f"{product}: period must be day or month.")
        if options["input_unit"] == "mg_ch4_m2_month" and options["period"] != "month":
            raise ValueError(f"{product}: monthly totals require monthly supervision.")
        if options.get("sign", 1) not in {-1, 1}:
            raise ValueError(f"{product}: sign must be 1 or -1.")
        if product in GRID_PRODUCTS:
            base = Path(config.get("simulation_dir", "data/methane_data/simulations"))
            path = base / options["path"]
            days = _date_range(options.get("date_range", config.get(
                "simulation_date_range", ["1980-01-01", "2021-01-01"],
            )))
            options.setdefault("latitude_axis", "Lat" if product == "memo_uptake" else "latitude")
            options.setdefault("longitude_axis", "Lon" if product == "memo_uptake" else "longitude")
            if options["period"] != defaults["period"]:
                raise ValueError(f"{product}: use the native simulation period {defaults['period']}.")
            grid = GridProduct(product, path, options, days)
            train_cells, validation_cells = _grid_split(
                grid, block_degrees=float(config.get("simulation_spatial_block_degrees", 5.0)),
                seed=seed, validation_fraction=validation_fraction,
            )
            budgets = config.get("samples_per_simulation_source", {})
            train.append(GridSamples(grid, train_cells, int(budgets.get("train", 12000)), seed + 2 * index))
            validation.append(GridSamples(grid, validation_cells, int(budgets.get("validation", 3000)),
                                          seed + 2 * index + 1))
            products[product] = {"files": grid.provenance,
                                 "sampling": {"train": len(train[-1]), "validation": len(validation[-1])}}
        else:
            base = Path(config.get("observations_dir", "data/methane_data/observations"))
            path = base / options["path"]
            days = _date_range(options.get("date_range", config.get(
                "observation_date_range", ["1980-01-01", "2026-01-01"],
            )))
            products[product] = {"files": [_file_metadata(path)]}
            if product == "fluxnet":
                options["metadata_path"] = str(base / options["metadata_path"])
                products[product]["files"].append(_file_metadata(Path(options["metadata_path"])))
                options["metadata_path"] = Path(options["metadata_path"])
                options["target_type"] = config.get("fluxnet_target_type", "emission")
                if options["target_type"] not in {"emission", "net"}:
                    raise ValueError("fluxnet_target_type must be emission or net.")
                products[product]["emission_proxy"] = options["target_type"] == "emission"
            observations.append(_observation_samples(product, path, options, days, site_filter))
            observation_names.append(product)
        products[product].update({
            "target_type": options["target_type"], "input_unit": options["input_unit"],
            "period": options["period"], "start": days[0].isoformat(), "end": days[1].isoformat(),
        })
    validation_sites = set()
    dataset_sites = {"train": {}, "validation": {}}
    if observations:
        validation_sites = _observation_split(observations, config.get("validation_sites"),
                                              validation_fraction, seed)
        for product, rows in zip(observation_names, observations):
            train_rows = []
            val_rows = []
            for row in rows:
                rows_for_site = val_rows if _site_key(row["site_id"]) in validation_sites else train_rows
                rows_for_site.append(row)
            for split, split_rows in (("train", train_rows), ("validation", val_rows)):
                dataset_sites[split][product] = sorted(
                    {_site_key(row["site_id"]) for row in split_rows if row["quality_mask"]}
                )
            if stage != "continual" and not all(dataset_sites[split][product] for split in dataset_sites):
                raise ValueError(f"{product}: train and validation each need valid labels from distinct sites.")
            train.append(train_rows)
            validation.append(val_rows)
    provenance = {
        "stage": stage, "flux_unit": FLUX_UNIT, "products": products, "seed": seed,
        "validation_fraction": validation_fraction, "validation_sites": sorted(validation_sites),
        "gather_mode": config.get("gather_mode", "history"),
        "embedding_mode": config.get("embedding_mode", "online"),
        "adapters": config.get("adapters", {}),
        "observation_time_reference": "nominal_date_as_utc",
        "simulation_spatial_block_degrees": config.get("simulation_spatial_block_degrees", 5),
        "observation_site_ids": sorted(site_filter) if site_filter is not None else None,
    }
    collator = _collator(config, gatherer, encoder_metadata)
    train_ds = MoEDatasetAdapter(train, dict(provenance, split="train", sites_by_product=dataset_sites["train"]))
    val_ds = MoEDatasetAdapter(validation, dict(
        provenance, split="validation", sites_by_product=dataset_sites["validation"],
    ))
    loader_options = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collator)
    train_loader = DataLoader(train_ds, shuffle=bool(len(train_ds)),
                              generator=torch.Generator().manual_seed(seed), **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, generator=torch.Generator().manual_seed(seed + 1),
                            **loader_options)
    return train_loader, val_loader
