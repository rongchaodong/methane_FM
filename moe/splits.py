"""Create and reuse immutable supervision partitions for the MoE stages."""

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import numpy as np

from encoding.specs import QuerySpec, as_datetime

from .data import (
    ADAPTERS, DATA_QUERY_VERSION, FLUX_UNIT, GRID_PRODUCTS, SPLITS, _date_range, _file_metadata,
    _observation_samples, _site_key, product_inputs, product_names,
)
from .simulation import (
    FIT_SPLITS, SimulationSplit, block_ids as _block_ids, grid_input_stats,
    open_grid, prepare_full_simulation, verify_bitmap, year_partition,
)
from .model import TARGET_TYPES


FORMAT_VERSION = 2
SIGN_COUNTS = ("positive", "negative", "zero", "missing")


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode()


def _read_snapshot(path, expected=None):
    data = path.read_bytes()
    manifest = json.loads(data)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported split manifest version: {path}. Prepare splits in a new manifest directory.")
    ref = {"id": manifest["settings"]["id"], "path": str(path.resolve()),
           "sha256": sha256(data).hexdigest()}
    if expected is not None and ref != expected:
        raise ValueError(f"Split manifest changed since checkpoint creation: {path}")
    for product, data in manifest["products"].items():
        if "bitmap" in data:
            bitmap = data["bitmap"]
            verify_bitmap(path.parent / "indices" / bitmap["file"], bitmap)
            if any(_file_metadata(Path(file["path"])) != file for file in manifest["settings"]["inputs"][product]["files"]):
                raise ValueError(f"Frozen simulation input changed: {product}")
    return ref, manifest


def _publish(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(manifest)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(f"A different split manifest already exists: {path}")
        finally:
            temporary.unlink()
    return _read_snapshot(path)


def _member(sample, width, *, shared_monthly_support=False):
    query = sample["query"]
    row = {"site_id": sample["site_id"], "study_id": sample["study_id"],
           "query": [float(query.lat), float(query.lon), query.start_time.isoformat(),
                     query.end_time.isoformat(), float(query.delta_lat), float(query.delta_lon)],
           "spatial_block": int(_block_ids(query.lat, query.lon, width)),
           "target": sample["target"], "sample_weight": sample["sample_weight"]}
    if shared_monthly_support and row["study_id"]:
        row["shared_monthly_support"] = True
    return row


def _split_stats(members, counts, *, sites=None):
    targets = np.array([row["target"] for row in members])
    filtered = sum(value for key, value in counts.items() if key.startswith("filtered_"))
    stats = {
        "raw": counts["raw"], "valid_labels": len(members), "filtered": filtered,
        "raw_signs": {sign: counts[f"raw_{sign}"] for sign in SIGN_COUNTS},
        "valid_signs": {"positive": int((targets > 0).sum()), "negative": int((targets < 0).sum()),
                        "zero": int((targets == 0).sum())},
        "sites": len(sites or ()),
        "grid_cells": len({tuple(row["query"][:2]) for row in members}) if sites is None else None,
        "spatial_blocks": len({row["spatial_block"] for row in members}),
        "row_counts": dict(counts),
        "distribution": None,
    }
    if len(targets):
        percentiles = (1, 5, 25, 50, 75, 95, 99)
        stats["distribution"] = {
            "mean": float(targets.mean()), "std": float(targets.std()),
            "min": float(targets.min()), "max": float(targets.max()),
            **{f"p{q:02d}": float(value) for q, value in zip(percentiles, np.percentile(targets, percentiles))},
        }
    if sites:
        stats["site_equal_mean"] = float(np.mean([site["target_sum"] / site["labels"] for site in sites.values()]))
    return stats


def _block_partition(settings):
    if settings["simulation_split_mode"] != "spatial":
        return {}
    width, seed = settings["block_degrees"], settings["split_seed"]
    n_blocks = int(np.ceil(180 / width)) * int(np.ceil(360 / width))
    validation = np.random.default_rng(seed).choice(n_blocks, round(0.2 * n_blocks), replace=False)
    return {"train": np.setdiff1d(np.arange(n_blocks), validation).tolist(),
            "validation": np.sort(validation).tolist()}


def _debug_simulation(settings, block_lists):
    width, seed = settings["block_degrees"], settings["split_seed"]
    products = {}
    for product, inputs in settings["inputs"].items():
        grid = open_grid(product, inputs)
        options = grid.options
        blocks = _block_ids(grid.lat[:, None], grid.lon[None, :], width).ravel()
        held_out = np.isin(blocks, block_lists.get("validation", []))
        raw_counts = grid_input_stats(grid)
        year_roles = year_partition(grid) if not block_lists else None
        partitions = {}
        for index, split in enumerate(FIT_SPLITS):
            cells = np.flatnonzero((~held_out, held_out)[index]) if block_lists else np.arange(len(blocks))
            periods = np.flatnonzero(np.isin(grid.period_years, year_roles[split])) if year_roles else None
            product_seed = np.random.SeedSequence([seed, GRID_PRODUCTS.index(product), index])
            members, counts = [], Counter()
            for sample in grid.draw(cells, settings["simulation_budgets"][split], product_seed, periods):
                counts.update({"raw": 1, f'raw_{sample["raw_sign"]}': 1})
                if sample["quality_mask"]:
                    members.append(_member(sample, width))
                else:
                    counts[f'filtered_{sample["filter_reason"]}'] += 1
            stats = _split_stats(members, counts)
            stats.update(native_candidates=len(cells) * (len(grid.dates) if periods is None else len(periods)),
                         assigned_blocks=len(np.unique(blocks[cells])))
            partitions[split] = {"members": members, "stats": stats}
        products[product] = {"source": options["source"], "task": options["target_type"],
                             "splits": partitions, "input_stats": raw_counts, "years": year_roles}
    return products


def _site_profiles(members):
    sites = {}
    for row in members:
        key, query = _site_key(row["site_id"]), row["query"]
        site = sites.setdefault(key, {"start": query[2], "end": query[3], "coordinates": set(),
                                      "studies": set(), "labels": 0, "target_sum": 0.0})
        site["start"], site["end"] = min(site["start"], query[2]), max(site["end"], query[3])
        site["coordinates"].add(tuple(query[:2]))
        if row["study_id"]:
            site["studies"].add(row["study_id"])
        site["labels"] += 1
        site["target_sum"] += row["target"]
    for site in sites.values():
        site["coordinates"], site["studies"] = sorted(site["coordinates"]), sorted(site["studies"])
    return sites


def _site_assignments(snapshots):
    assigned = {}
    for _, manifest in snapshots:
        for product in manifest["products"].values():
            for site, split in product.get("site_roles", {}).items():
                if assigned.setdefault(site, split) != split:
                    raise ValueError(f"Observation site crosses frozen partitions: {site}")
    return assigned


def _assign_sites(sites, assigned, seed):
    n_sites = len(sites)
    new_sites = sorted(sites.keys() - assigned.keys())
    if not new_sites:
        return
    held_out = max(1, round(n_sites * 0.2))
    targets = (n_sites - 2 * held_out, held_out, held_out) if n_sites >= 3 else (1, n_sites - 1, 0)
    fixed = Counter(assigned[site] for site in sites if site in assigned)
    deficits = np.array([max(0, target - fixed[split]) for split, target in zip(SPLITS, targets)])
    # Inherited site roles take priority when they exceed a cohort's target share.
    quotas = deficits * len(new_sites) / deficits.sum()
    counts = np.floor(quotas).astype(int)
    rest = len(new_sites) - int(counts.sum())
    counts[np.argsort(counts - quotas, kind="stable")[:rest]] += 1
    tie_break = dict(zip(new_sites, np.random.default_rng(seed).permutation(len(new_sites))))
    ordered = sorted(new_sites, key=lambda key: (sites[key]["end"], sites[key]["start"], tie_break[key]))
    cursor = 0
    for split, count in zip(SPLITS, counts):
        for site in ordered[cursor:cursor + count]:
            assigned[site] = split
        cursor += count


def _period_key(row):
    return _site_key(row["site_id"]), row["query"][2]


def _temporal_roles(members, short_rule):
    dates, roles, short = defaultdict(Counter), {}, Counter()
    for row in members:
        site, start = _period_key(row)
        dates[site][start] += 1
    for site, site_dates in dates.items():
        starts = sorted(site_dates)
        n = len(starts)
        if n < 3:
            short.update({"sites": 1, "labels": sum(site_dates.values()), f"sites_with_{n}_periods": 1})
            validation_start = 1 if short_rule == "train_validation" and n == 2 else n
            testing_start = n
            if short_rule == "exclude":
                for start in starts:
                    roles[site, start] = None
                continue
        else:
            held_out = max(1, round(n * 0.2))
            validation_start, testing_start = n - 2 * held_out, n - held_out
        for index, start in enumerate(starts):
            split = "train" if index < validation_start else "validation" if index < testing_start else "testing"
            roles[site, start] = split
    short["sites"] += 0
    short["labels"] += 0
    return roles, {"rule": short_rule, **short}


def _exclude_held_out_copies(partitions):
    # Keep shared support in its most held-out role without moving a site's time boundary.
    held_labels, held_queries = set(), set()
    for split in reversed(SPLITS):
        added_labels, added_queries = set(), set()
        for product, (members_by_split, counts) in partitions.items():
            kept = []
            for row in members_by_split[split]:
                key, query = _label_key(product, row), tuple(row["query"])
                if key in held_labels or query in held_queries:
                    counts[split]["filtered_duplicate_held_out_support"] += 1
                else:
                    kept.append(row)
                    added_labels.add(key)
                    added_queries.add(query)
            members_by_split[split] = kept
        held_labels.update(added_labels)
        held_queries.update(added_queries)


def _observations(settings, history):
    spatial = settings["observation_split_mode"] == "spatial"
    assigned = _site_assignments(history) if spatial else {}
    products, partitions = {}, {}
    for product in (name for name in ADAPTERS if name in settings["inputs"]):
        inputs = settings["inputs"][product]
        options = dict(inputs["options"])
        if product == "fluxnet":
            options["metadata_path"] = Path(options["metadata_path"])
        samples, profile, site_counts = _observation_samples(
            product, Path(inputs["path"]), options, _date_range(inputs["date_range"]), settings["site_ids"],
        )
        chamber_monthly = product == "chamber" and options["period"] == "month"
        members = [_member(sample, settings["block_degrees"], shared_monthly_support=chamber_monthly)
                   for sample in samples]
        if spatial:
            sites = _site_profiles(members)
            _assign_sites(sites, assigned, settings["split_seed"])
            site_roles = {site: assigned[site] for site in sites}
            roles = {key: assigned.get(key[0]) for key in site_counts}
        else:
            site_roles = {}
            roles, short = _temporal_roles(members, settings["observation_short_sites"])
            profile["short_sites"] = short
        by_split, counts = defaultdict(list), defaultdict(Counter)
        for row in members:
            split = roles.get(_period_key(row))
            if split is None:
                counts["unassigned"]["filtered_short_site"] += 1
            else:
                by_split[split].append(row)
        for key, row_counts in site_counts.items():
            counts[roles.get(key) or "unassigned"].update(row_counts)
        profile["candidate_labels"] = len(members)
        products[product] = {"source": options["source"], "task": options["target_type"],
                             "input_stats": profile, "site_roles": site_roles}
        partitions[product] = (by_split, counts)

    _exclude_held_out_copies(partitions)
    for product, (by_split, counts) in partitions.items():
        profile = products[product]["input_stats"]
        profile["scoped_unassigned_rows"] = dict(counts["unassigned"])
        profile["scoped_row_counts"] = dict(sum(counts.values(), Counter()))
        profiles = {split: _site_profiles(by_split[split]) for split in SPLITS}
        valid_sites = set().union(*(set(sites) for sites in profiles.values()))
        label_count = sum(map(len, by_split.values()))
        split_data = {}
        for split in SPLITS:
            split_sites = profiles[split]
            stats = _split_stats(by_split[split], counts[split], sites=split_sites)
            stats["site_fraction"] = len(split_sites) / len(valid_sites) if valid_sites else None
            stats["label_fraction"] = len(by_split[split]) / label_count if label_count else None
            split_data[split] = {"members": by_split[split], "stats": stats}
        products[product].update(splits=split_data, sites=profiles)
    return products


def _rows(manifest, split, names=None):
    for product, data in manifest["products"].items():
        if names is None or product in names:
            for row in data["splits"].get(split, {}).get("members", ()):
                yield product, row


def _label_key(product, row):
    # This native identity is independent of the task, value, and configured Query widths.
    site = "" if row.get("shared_monthly_support") else _site_key(row["site_id"])
    return product, site, row["study_id"], *row["query"][:4]


def _local_intersections(manifest):
    simulation = manifest["settings"]["kind"] == "simulation"
    splits = FIT_SPLITS if simulation else SPLITS
    raw = {split: {_label_key(p, row) for p, row in _rows(manifest, split)} for split in splits}
    query = {split: {tuple(row["query"]) for _, row in _rows(manifest, split)} for split in splits}
    if simulation:
        grouping = manifest["spatial_blocks"]
        if not grouping:
            return {"train/validation": {
                "raw_labels": len(raw["train"] & raw["validation"]),
                "exact_queries": len(query["train"] & query["validation"]),
                "shared_product_years": sum(len(set(data["years"]["train"]) & set(data["years"]["validation"]))
                                            for data in manifest["products"].values()),
            }}
        group_key = "shared_blocks"
    else:
        grouping = {split: {_site_key(row["site_id"]) for _, row in _rows(manifest, split)} for split in splits}
        group_key = "shared_sites"
    checks = {
        f"{left}/{right}": {"raw_labels": len(raw[left] & raw[right]),
                            "exact_queries": len(query[left] & query[right]),
                            group_key: len(set(grouping[left]).intersection(grouping[right]))}
        for left, right in combinations(splits, 2)
    }
    if not simulation and manifest["settings"]["observation_split_mode"] == "temporal":
        for left, right in combinations(splits, 2):
            invalid = 0
            for data in manifest["products"].values():
                sites_left, sites_right = data["sites"][left], data["sites"][right]
                invalid += sum(sites_left[site]["end"] > sites_right[site]["start"]
                               for site in sites_left.keys() & sites_right.keys())
            checks[f"{left}/{right}"]["nonchronological_sites"] = invalid
    return checks


def _site_distances(profiles, width):
    points, blocks, studies, time = {}, {}, {}, {}
    for split in SPLITS:
        sites = [(key, site) for key, site in profiles if site["split"] == split]
        points[split] = sorted({(key, *coord) for key, site in sites for coord in site["coordinates"]})
        blocks[split] = {int(_block_ids(lat, lon, width)) for _, lat, lon in points[split]}
        studies[split] = {study for _, site in sites for study in site["studies"]}
        if sites:
            time[split] = [min(site["start"] for _, site in sites), max(site["end"] for _, site in sites)]
    report = {}
    for left, right in combinations(SPLITS, 2):
        a, b = points[left], points[right]
        diagnostics = {"nearest_km": None, "nearest_sites": None, "site_pairs_within_km": {},
                       "shared_blocks": len(blocks[left] & blocks[right]),
                       "shared_study_ids": sorted(studies[left] & studies[right]),
                       "time_ranges": [time.get(left), time.get(right)], "time_range_overlap_days": None}
        report[f"{left}/{right}"] = diagnostics
        if not a or not b:
            continue
        alat, alon = np.radians([(lat, lon) for _, lat, lon in a]).T[:, :, None]
        blat, blon = np.radians([(lat, lon) for _, lat, lon in b]).T[:, None, :]
        half_chord = np.sin((alat - blat) / 2) ** 2 + np.cos(alat) * np.cos(blat) * np.sin((alon - blon) / 2) ** 2
        distance = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(half_chord, 0, 1)))
        i, j = np.unravel_index(distance.argmin(), distance.shape)
        diagnostics["nearest_km"], diagnostics["nearest_sites"] = float(distance[i, j]), [a[i][0], b[j][0]]
        diagnostics["site_pairs_within_km"] = {
            str(km): len({(a[i][0], b[j][0]) for i, j in zip(*np.where(distance <= km))})
            for km in (0, 1, 10, 50)
        }
        start, end = max(time[left][0], time[right][0]), min(time[left][1], time[right][1])
        diagnostics["time_range_overlap_days"] = max(0, (as_datetime(end) - as_datetime(start)).days)
    return report


def prepare_manifest(config, sources, stage, history=(), expected=None):
    """Reuse an existing manifest or publish the first split for this dataset and cohort."""
    simulation = stage == "pretrain"
    kind = {"pretrain": "simulation", "finetune": "initial_observation",
            "continual": "continual_observation"}[stage]
    if stage == "continual":
        cohort = config.get("observation_cohort")
        if not isinstance(cohort, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", cohort):
            raise ValueError("Set a unique data.observation_cohort for the new continual cohort.")
        name = f"continual_{cohort}"
    else:
        name = kind
    path = Path(expected["path"]) if expected else Path(config.get("split_manifest_dir", "moe/runs/splits-v2")) / f"{name}.json"
    seed, width = int(config.get("split_seed", 42)), float(config.get("simulation_spatial_block_degrees", 5))
    if seed < 0 or not 0.5 <= width < 180:
        raise ValueError("Expected split_seed >= 0 and spatial block degrees in [0.5, 180).")
    if any(manifest["settings"]["split_seed"] != seed for _, manifest in history):
        raise ValueError("The split_seed differs from the historical split manifests.")
    sim_split = config.get("simulation_split_mode", "temporal")
    sim_mode = config.get("simulation_mode", "full")
    if simulation and (sim_split not in {"spatial", "temporal"} or sim_mode not in {"full", "debug"}):
        raise ValueError("Expected simulation_split_mode temporal/spatial and simulation_mode full/debug.")
    budgets = {"train": 12000, "validation": 3000} | config.get("samples_per_simulation_source", {})
    tile_bytes = int(config.get("simulation_tile_mib", 16) * 1024 * 1024)
    if simulation and tile_bytes < 1:
        raise ValueError("simulation_tile_mib must be positive.")
    observation_mode = config.get("observation_split_mode", "temporal")
    short_sites = config.get("observation_short_sites", "train_only")
    if not simulation and (observation_mode not in {"temporal", "spatial"} or
                           short_sites not in {"train_only", "train_validation", "exclude"}):
        raise ValueError("Invalid observation_split_mode or observation_short_sites.")
    if not simulation and any(
        manifest["settings"].get("observation_split_mode") not in (None, observation_mode)
        for _, manifest in history
    ):
        raise ValueError("Observation split mode differs from the historical lineage.")
    site_ids = config.get("observation_site_ids")
    names = product_names(sources, grid=simulation)
    settings = {
        "id": name, "kind": kind, "split_seed": seed, "block_degrees": width,
        "data_query_version": DATA_QUERY_VERSION,
        "algorithm": (f"simulation_{sim_split}_80_20" if simulation else
                      "within_site_time_groups_60_20_20" if observation_mode == "temporal" else
                      "site_latest_end_then_start_60_20_20"),
        "simulation_mode": sim_mode if simulation else None,
        "simulation_split_mode": sim_split if simulation else None,
        "simulation_tile_bytes": tile_bytes if simulation and sim_mode == "full" else None,
        "simulation_budgets": {key: int(budgets[key]) for key in FIT_SPLITS} if simulation and sim_mode == "debug" else None,
        "site_ids": sorted({_site_key(site) for site in site_ids}) if not simulation and site_ids is not None else None,
        "observation_parents": [ref["sha256"] for ref, manifest in history
                                if manifest["settings"]["kind"] != "simulation"],
        "inputs": product_inputs(config, names),
    }
    if not simulation:
        settings.update(observation_split_mode=observation_mode,
                        observation_short_sites=short_sites if observation_mode == "temporal" else None)
    settings = json.loads(_json_bytes(settings))
    if expected or path.exists():
        ref, manifest = _read_snapshot(path, expected)
        if manifest["settings"] != settings:
            raise ValueError(f"Data, cohort settings, or split seed differ from the frozen manifest: {path}")
        return ref, manifest

    manifest = {"format_version": FORMAT_VERSION, "settings": settings, "flux_unit": FLUX_UNIT}
    if simulation:
        blocks = manifest["spatial_blocks"] = _block_partition(settings)
        if sim_mode == "full":
            manifest["products"] = prepare_full_simulation(settings, path.parent / "indices", blocks)
        else:
            manifest["products"] = _debug_simulation(settings, blocks)
    else:
        manifest["products"] = _observations(settings, history)
        profiles = {name: [(site, dict(metadata, split=split))
                           for split, group in product["sites"].items() for site, metadata in group.items()]
                    for name, product in manifest["products"].items()}
        profiles["all"] = [item for group in profiles.values() for item in group]
        manifest["location_diagnostics"] = {key: _site_distances(items, width) for key, items in profiles.items()}
    manifest["split_intersections"] = _local_intersections(manifest)
    if any(counts["raw_labels"] or counts["exact_queries"] or counts.get("shared_blocks")
           or counts.get("shared_product_years") or counts.get("nonchronological_sites")
           or (not simulation and observation_mode == "spatial" and counts.get("shared_sites"))
           for counts in manifest["split_intersections"].values()):
        raise ValueError("Label, Query, or site-time independence check failed.")
    if settings["inputs"] != json.loads(_json_bytes(product_inputs(config, names))):
        raise ValueError("Input files changed while creating the split manifest.")
    _check_new_cohort(history, manifest)
    return _publish(path, manifest)


def _check_new_cohort(history, manifest):
    if manifest["settings"]["kind"] != "continual_observation":
        return
    old_periods, new_periods = defaultdict(set), defaultdict(set)
    old_queries, new_queries = set(), set()
    for snapshot, index, queries in [(m, old_periods, old_queries) for _, m in history] + [
        (manifest, new_periods, new_queries)
    ]:
        if snapshot["settings"]["kind"] != "simulation":
            for split in SPLITS:
                for product, row in _rows(snapshot, split):
                    queries.add(tuple(row["query"]))
                    period = tuple(row["query"][2:4])
                    index[product, _site_key(row["site_id"])].add(period)
                    if product == "chamber" and row["study_id"]:
                        index[product, row["study_id"], *row["query"][:2]].add(period)
    if old_queries & new_queries:
        raise ValueError("Continual observation periods overlap prior labels by exact Query identity.")
    temporal = manifest["settings"]["observation_split_mode"] == "temporal"
    for key in old_periods.keys() & new_periods.keys():
        if temporal and len(key) == 2 and min(start for start, _ in new_periods[key]) < max(
            end for _, end in old_periods[key]
        ):
            raise ValueError(f"New temporal cohort must follow prior labels at the same site: {key}")
        old = iter(sorted(old_periods[key]))
        start, end = next(old)
        for new_start, new_end in sorted(new_periods[key]):
            while end <= new_start:
                start, end = next(old, (None, None))
                if end is None:
                    break
            if end is None:
                break
            if start < new_end:
                raise ValueError(f"Continual observation periods overlap prior labels: {key}")


def stage_manifests(config, sources, stage, past_data, *, resume, replay_sources):
    if past_data is None:
        past_data = {}
    elif not past_data.get("active_manifest"):
        raise ValueError("The checkpoint predates frozen splits. Start a new lineage from a split-aware pretrain run.")
    active_ref = past_data.get("active_manifest")
    refs = past_data.get("historical_manifests", []) + ([active_ref] if active_ref and not resume else [])
    history = [_read_snapshot(Path(ref["path"]), ref) for ref in refs]
    active = prepare_manifest(config, sources, stage, history, active_ref if resume else None)
    if active[1]["settings"].get("observation_split_mode") == "spatial":
        _site_assignments([*history, active])
    _check_new_cohort(history, active[1])
    replay_names = product_names(replay_sources) if replay_sources else []
    available = {name for _, manifest in history for name in manifest["products"]}
    if missing := set(replay_names) - available:
        raise ValueError(f"No frozen historical partitions for replay products: {sorted(missing)}.")

    historical_train = {_label_key(p, row) for _, m in history for p, row in _rows(m, "train")}
    replay_validation = {_label_key(p, row) for _, m in history for p, row in _rows(m, "validation", replay_names)}
    testing = {_label_key(p, row) for _, m in [*history, active] for p, row in _rows(m, "testing")}
    checks = {"historical_train/replay_validation": len(historical_train & replay_validation),
              "historical_train/testing": len(historical_train & testing)}
    if any(checks.values()):
        raise ValueError(f"Historical training data intersect a held-out partition: {checks}")
    return active, history, replay_names, checks


def loader_partitions(snapshots, names=None):
    """Project frozen records onto the fitting inputs and their split-local metadata."""
    partitions = []
    grids = {}
    for split in FIT_SPLITS:
        segments = []
        provenance = {"split": split, "flux_unit": FLUX_UNIT, "manifests": [], "products": {}}
        for ref, manifest in snapshots:
            selected = set(manifest["products"]) if names is None else set(names) & manifest["products"].keys()
            if not selected:
                continue
            provenance["manifests"].append(ref)
            for product in sorted(selected):
                data = manifest["products"][product]
                partition = data["splits"][split]
                metadata = provenance["products"].setdefault(product, {
                    "target_type": data["task"], "valid_labels": 0,
                })
                if "bitmap" in data:
                    name = (ref["id"], product)
                    if name not in grids:
                        grids[name] = open_grid(product, manifest["settings"]["inputs"][product], data["input_stats"])
                    bitmap = Path(ref["path"]).parent / "indices" / data["bitmap"]["file"]
                    segment = SimulationSplit(grids[name], bitmap, partition["shards"])
                    metadata["valid_labels"] += len(segment)
                    segments.append(segment)
                    continue
                rows = partition["members"]
                metadata["valid_labels"] += len(rows)
                segment = []
                for row in rows:
                    query = QuerySpec(*row["query"])
                    segment.append({
                        "query": query, "target": row["target"], "sample_weight": row["sample_weight"],
                        "site_id": row["site_id"], "product": product, "source": data["source"],
                        "target_type": data["task"], "quality_mask": True,
                        "spatial_support": (query.delta_lat, query.delta_lon),
                        "temporal_support": (query.start_time, query.end_time),
                    })
                segments.append(segment)
        partitions.append((segments, provenance))
    return partitions


def training_label_statistics(snapshot):
    """Merge full-coverage, unweighted Training moments independently by task."""
    moments = {}
    for product in snapshot[1]["products"].values():
        partition = product["splits"]["train"]
        count = partition["stats"]["valid_labels"]
        if not count:
            continue
        if "bitmap" in product:
            # The full-grid index already scanned every selected label in bounded tiles.
            distribution = partition["stats"]["distribution"]
            mean, variance = distribution["mean"], distribution["std"] ** 2
        else:
            # Frozen observation and debug manifests contain the Training values directly.
            mean, variance = 0.0, 0.0
            for index, row in enumerate(partition["members"], 1):
                delta = row["target"] - mean
                mean += delta / index
                variance += delta * (row["target"] - mean)
            variance /= count
        task = product["task"]
        previous_count, previous_mean, previous_m2 = moments.get(task, (0, 0.0, 0.0))
        total = previous_count + count
        delta = mean - previous_mean
        moments[task] = (total, previous_mean + delta * count / total,
                         previous_m2 + variance * count + delta**2 * previous_count * count / total)
    if missing := set(TARGET_TYPES) - moments.keys():
        raise ValueError(f"Label normalization requires valid Training labels for: {sorted(missing)}")
    return {task: {"count": count, "mean": mean, "std": math.sqrt(m2 / count)}
            for task, (count, mean, m2) in moments.items()}


def audit_summary(snapshot, checks, batch_size=8):
    ref, manifest = snapshot
    result = {
        "manifest": ref, "flux_unit": manifest["flux_unit"], "split_seed": manifest["settings"]["split_seed"],
        "algorithm": manifest["settings"]["algorithm"],
        "spatial_blocks": {key: len(ids) for key, ids in manifest.get("spatial_blocks", {}).items()},
        "split_intersections": manifest["split_intersections"], "lineage_intersections": checks,
        "location_diagnostics": manifest.get("location_diagnostics"),
        "products": {name: {
            "source": product["source"], "task": product["task"], "input": product["input_stats"],
            "splits": {split: data["stats"] for split, data in product["splits"].items()},
        } for name, product in manifest["products"].items()},
    }
    if manifest["settings"].get("simulation_mode") == "full":
        result["simulation_steps"] = {
            split: sum((shard["labels"] + batch_size - 1) // batch_size
                       for product in manifest["products"].values()
                       for shard in product["splits"][split]["shards"])
            for split in FIT_SPLITS
        }
        result["batch_size"] = batch_size
        result["simulation_index_bytes"] = sum(product["bitmap"]["bytes"] for product in manifest["products"].values())
    return result
