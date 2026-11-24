"""Full-grid validity indices and bounded native simulation reads."""

from bisect import bisect_right
from calendar import monthrange
from collections import Counter, OrderedDict, defaultdict
from hashlib import sha256
from itertools import groupby
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

from netCDF4 import Dataset as NetCDF
import numpy as np
from torch.utils.data import Dataset

from .data import GridProduct, _date_range


FIT_SPLITS = ("train", "validation")


def block_ids(lat, lon, width):
    ny, nx = int(np.ceil(180 / width)), int(np.ceil(360 / width))
    row = np.minimum(np.floor((np.asarray(lat) + 90) / width).astype(int), ny - 1)
    col = np.minimum(np.floor((np.asarray(lon) + 180) / width).astype(int), nx - 1)
    return row * nx + col


def open_grid(product, inputs, input_stats=None):
    options = dict(inputs["options"])
    if product == "tem_uptake" and input_stats is not None:
        options["sign"] = input_stats["applied_sign"]
    return GridProduct(product, Path(inputs["path"]), options, _date_range(inputs["date_range"]))


def year_partition(grid):
    years = np.unique(grid.period_years).tolist()
    holdout = max(1, round(len(years) * 0.2)) if len(years) > 1 else 0
    cut = len(years) - holdout
    return {"train": years[:cut], "validation": years[cut:]}


def grid_input_stats(grid):
    raw = {"invalid_calendar_periods": grid.invalid_calendar_periods,
           "native_periods_in_range": len(grid.dates), "native_grid_cells": len(grid.lat) * len(grid.lon),
           "grid_geometry": {"raw_lat_edges": [float(grid.raw_lat[0]), float(grid.raw_lat[-1])],
                             "raw_lon_edges": [float(grid.raw_lon[0]), float(grid.raw_lon[-1])],
                             "lat_centers": [float(grid.lat[0]), float(grid.lat[-1])],
                             "lon_centers": [float(grid.lon[0]), float(grid.lon[-1])],
                             "spatial_support": list(grid.support)}}
    if grid.product == "tem_uptake":
        signs = grid.uptake_sign_counts()
        sign = 1 if signs["negative"] > signs["positive"] else -1
        if grid.options.get("sign", sign) != sign:
            raise ValueError("TEM uptake sign override disagrees with the native oxidation field.")
        grid.options["sign"] = sign
        raw.update(native_signs=dict(signs), applied_sign=sign)
    return raw


def _raw_masks(grid, values):
    missing = ~np.isfinite(values)
    for marker in grid.options.get("missing_values", ()):
        missing |= values == marker
    positive = (values > 0) & ~missing
    negative = (values < 0) & ~missing
    zero = (values == 0) & ~missing
    if grid.product in ("tem_emission", "memo_uptake"):
        filtered = negative
    elif grid.options.get("nonpositive_uptake", False):
        filtered = positive if grid.options["sign"] > 0 else negative
    else:
        filtered = False
    valid = ~(missing | filtered)
    return {"positive": positive, "negative": negative, "zero": zero, "missing": missing}, valid


def _distribution(values, totals, quantile_sample, rng):
    if not len(values):
        return
    totals["count"] += len(values)
    totals["sum"] += float(values.sum(dtype=np.float64))
    totals["sum_sq"] += float(np.square(values, dtype=np.float64).sum())
    totals["min"] = min(totals.get("min", math.inf), float(values.min()))
    totals["max"] = max(totals.get("max", -math.inf), float(values.max()))
    totals["positive"] += int(np.count_nonzero(values > 0))
    totals["negative"] += int(np.count_nonzero(values < 0))
    totals["zero"] += int(np.count_nonzero(values == 0))
    n = rng.binomial(len(values), 0.0002)
    if n:
        quantile_sample.append(values[rng.choice(len(values), n, replace=False)])


def _summary(totals, sample):
    n = totals["count"]
    if not n:
        return None
    mean = totals["sum"] / n
    values = {"mean": mean, "std": math.sqrt(max(0, totals["sum_sq"] / n - mean**2)),
              "min": totals["min"], "max": totals["max"]}
    if sample:
        values["sample_quantiles"] = {
            f"p{q:02d}": float(value)
            for q, value in zip((1, 5, 25, 50, 75, 95, 99),
                                np.percentile(np.concatenate(sample), (1, 5, 25, 50, 75, 95, 99)))
        }
        values["quantile_sample_size"] = sum(map(len, sample))
    return values


def _bitmap_file(grid, settings, directory, block_partition, years, stats):
    directory.mkdir(parents=True, exist_ok=True)
    width = settings["block_degrees"]
    all_blocks = block_ids(grid.lat[:, None], grid.lon[None, :], width).ravel()
    val_blocks = block_partition.get("validation", [])
    spatial = settings["simulation_split_mode"] == "spatial"
    year_stats, roles = defaultdict(Counter), {split: Counter() for split in FIT_SPLITS}
    descriptors = {split: [] for split in FIT_SPLITS}
    totals, samples = {split: Counter() for split in FIT_SPLITS}, {split: [] for split in FIT_SPLITS}
    occupied = {split: set() for split in FIT_SPLITS}
    rng = np.random.default_rng(settings["split_seed"])
    digest = sha256()
    with NamedTemporaryFile(dir=directory, prefix=f".{grid.product}-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for file_year, tiles in groupby(grid.tiles(settings["simulation_tile_bytes"]), key=lambda tile: tile[0]):
                with NetCDF(grid.files[file_year]) as file:
                    variable = file[grid.options["variable"]]
                    for tile in tiles:
                        periods, first_cell, n_cells = grid.tile_layout(tile)
                        periods = periods[periods >= 0]
                        if not len(periods):
                            continue
                        labels = grid.read_tile(variable, tile)
                        signs, valid = _raw_masks(grid, labels)
                        period_years = grid.period_years[periods]
                        tile_blocks = all_blocks[first_cell:first_cell + n_cells]
                        if spatial:
                            is_validation = np.isin(tile_blocks, val_blocks)[None, :]
                        else:
                            is_validation = np.isin(period_years, years["validation"])[:, None]
                        masks = {split: valid & selector
                                 for split, selector in zip(FIT_SPLITS, (~is_validation, is_validation))}
                        _record_years(year_stats, period_years, signs, valid, masks)
                        for split, selector in zip(FIT_SPLITS, (~is_validation, is_validation)):
                            mask = masks[split]
                            count = int(mask.sum())
                            role = roles[split]
                            scope = np.broadcast_to(selector, mask.shape)
                            role["raw"] += int(scope.sum())
                            for sign, raw_mask in signs.items():
                                role[sign] += int(np.count_nonzero(raw_mask & scope))
                            if not count:
                                continue
                            positions = np.flatnonzero(mask.any(axis=0))
                            occupied[split].update(np.unique(tile_blocks[positions]).tolist())
                            converted = labels[mask] * grid.options.get("sign", 1)
                            if grid.product == "memo_uptake":
                                dates = [grid.dates[i][0] for i in periods]
                                days = np.array([monthrange(date.year, date.month)[1] for date in dates])
                                converted = converted / np.repeat(days, mask.sum(axis=1))
                            _distribution(converted, totals[split], samples[split], rng)
                            packed = np.packbits(mask, bitorder="little").tobytes()
                            offset = stream.tell()
                            stream.write(packed)
                            digest.update(packed)
                            descriptors[split].append({"tile": list(tile), "offset": offset, "labels": count})
                print(f"Indexed {grid.product}: {grid.files[file_year].name}", flush=True, file=sys.stderr)
            stream.flush()
            os.fsync(stream.fileno())
            size = stream.tell()
            fingerprint = digest.hexdigest()
            final = directory / f"{grid.product}-{fingerprint}.bits"
            try:
                os.link(temporary, final)
            except FileExistsError:
                if final.stat().st_size != size:
                    raise ValueError(f"Invalid existing simulation bitmap: {final}")
        finally:
            temporary.unlink()
    stats["by_year"] = {str(year): dict(data) for year, data in sorted(year_stats.items())}
    stats["raw_calendar_labels"] = sum(data["raw"] for data in year_stats.values())
    stats["raw_native_slots"] = stats["raw_calendar_labels"] + grid.invalid_calendar_periods * stats["native_grid_cells"]
    stats["valid_labels"] = sum(data["count"] for data in totals.values())
    stats["bitmap_bytes"] = size
    stats["raw_signs"] = {sign: sum(year[sign] for year in year_stats.values())
                          for sign in ("positive", "negative", "zero", "missing")}
    stats["filtered"] = stats["raw_calendar_labels"] - stats["valid_labels"]
    return {
        "source": grid.options["source"], "task": grid.options["target_type"], "input_stats": stats,
        "bitmap": {"file": final.name, "bytes": size, "sha256": fingerprint},
        "years": years if not spatial else None,
        "splits": {split: {
            "shards": descriptors[split],
            "stats": {"raw": roles[split]["raw"], "valid_labels": totals[split]["count"],
                      "filtered": roles[split]["raw"] - totals[split]["count"],
                      "raw_signs": {sign: roles[split][sign] for sign in ("positive", "negative", "zero", "missing")},
                      "valid_signs": {sign: totals[split][sign] for sign in ("positive", "negative", "zero")},
                      "spatial_blocks": len(occupied[split]), "sites": 0, "shards": len(descriptors[split]),
                      "distribution": _summary(totals[split], samples[split])},
        } for split in FIT_SPLITS},
    }


def _record_years(year_stats, years, signs, valid, masks):
    unique = np.unique(years)
    for year in unique:
        selected = slice(None) if len(unique) == 1 else years == year
        row = year_stats[int(year)]
        raw = valid[selected].size
        counts = {split: int(mask[selected].sum()) for split, mask in masks.items()}
        if sum(counts.values()) != int(valid[selected].sum()):
            raise ValueError("Simulation splits must partition every valid native label.")
        row.update({"raw": raw, "valid": sum(counts.values()), "filtered": raw - sum(counts.values()), **counts})
        row.update({sign: int(mask[selected].sum()) for sign, mask in signs.items()})


def prepare_full_simulation(settings, directory, block_partition):
    products = {}
    for product, inputs in settings["inputs"].items():
        grid = open_grid(product, inputs)
        products[product] = _bitmap_file(grid, settings, directory, block_partition,
                                         year_partition(grid), grid_input_stats(grid))
    return products


def verify_bitmap(path, metadata):
    if path.stat().st_size != metadata["bytes"]:
        raise ValueError(f"Simulation bitmap size changed: {path}")
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != metadata["sha256"]:
        raise ValueError(f"Simulation bitmap content changed: {path}")


class SimulationSplit(Dataset):
    """Map rank within a split to one native tile, with a bounded per-worker cache."""

    def __init__(self, grid, bitmap, shards):
        self.grid, self.bitmap, self.shards = grid, bitmap, shards
        self.shard_lengths = [shard["labels"] for shard in shards]
        self.ends = np.cumsum(self.shard_lengths, dtype=np.int64).tolist()
        self.pid = None
        self.files = OrderedDict()
        self.cache = None
        self.index_stream = None

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def _load(self, index):
        pid = os.getpid()
        if self.pid != pid:
            self.close()
            self.index_stream = self.bitmap.open("rb")
            self.pid = pid
        if self.cache is not None and self.cache[0] == index:
            return self.cache[1:]
        shard = self.shards[index]
        tile = shard["tile"]
        year = tile[0]
        if year not in self.files:
            self.files[year] = NetCDF(self.grid.files[year])
            if len(self.files) > 2:
                self.files.popitem(last=False)[1].close()
        self.files.move_to_end(year)
        periods, cell_offset, cells = self.grid.tile_layout(tile)
        periods = periods[periods >= 0]
        shape = (len(periods), cells)
        self.index_stream.seek(shard["offset"])
        data = self.index_stream.read((shape[0] * shape[1] + 7) // 8)
        offsets = np.flatnonzero(np.unpackbits(np.frombuffer(data, dtype=np.uint8),
                                               count=shape[0] * shape[1], bitorder="little"))
        if len(offsets) != shard["labels"]:
            raise ValueError("Simulation bitmap has an unexpected valid label count.")
        labels = self.grid.read_tile(self.files[year][self.grid.options["variable"]], tile)
        self.cache = (index, periods, cell_offset, cells, offsets, labels.ravel())
        return self.cache[1:]

    def __getitem__(self, index):
        shard = bisect_right(self.ends, index)
        rank = index - (self.ends[shard - 1] if shard else 0)
        periods, cell_offset, cells, offsets, labels = self._load(shard)
        offset = int(offsets[rank])
        row, cell = divmod(offset, cells)
        sample = self.grid.sample_value(periods[row], cell_offset + cell, labels[offset])
        if not sample["quality_mask"]:
            raise ValueError("Indexed simulation label is now invalid. Check the frozen source files.")
        return sample

    def close(self):
        for file in self.files.values():
            file.close()
        self.files.clear()
        if self.index_stream is not None:
            self.index_stream.close()
        self.index_stream = self.cache = self.pid = None
