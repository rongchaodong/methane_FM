import csv
from copy import deepcopy
from datetime import datetime
from itertools import chain
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from encoding import DriverRegistry, MultiResolutionGatherer, build_data_sources
from encoding.training import load_config
from moe import data
from moe.checkpoint import load_checkpoint, restore_model
from moe.model import ContinualMethaneMoE, MoELoss
from moe.simulation import block_ids
from moe.splits import _read_snapshot, loader_partitions, prepare_manifest, stage_manifests
from moe import training
from moe.training import run_training


def _grid_file(path, axes, variable, values):
    with Dataset(path, "w") as file:
        for name, coordinates in axes.items():
            file.createDimension(name, len(coordinates))
            file.createVariable(name, "f8", (name,))[:] = coordinates
        file.createVariable(variable, "f8", tuple(axes))[:] = values


def _grid_config(path):
    return {"simulation_dir": path, "simulation_date_range": ["1979-01-01", "2021-01-01"]}


def test_tem_daily_filters_without_clipping_and_skips_invalid_dates(tmp_path):
    path = tmp_path / "TEM_EMISSION"
    path.mkdir()
    axes = {"latitude": [0], "longitude": [0, 1, 2, 3], "month": [2], "day": [28, 29, 30]}
    raw = np.full((1, 4, 1, 3), 3.5)
    raw[0, :, 0, 0] = [-1.25, 0.0, np.nan, np.inf]
    for year in (1979, 1980):
        _grid_file(path / f"CH4_emission_intensity_{year}.nc", axes, "CH4_emission", raw)
    inputs = data.product_inputs(_grid_config(tmp_path), ["tem_emission"])["tem_emission"]
    grid = data.GridProduct("tem_emission", Path(inputs["path"]), inputs["options"],
                            data._date_range(inputs["date_range"]))
    assert [date for date, _, _ in grid.dates] == [datetime(1979, 2, 28), datetime(1980, 2, 28),
                                                  datetime(1980, 2, 29)]
    assert grid.invalid_calendar_periods == 3
    with Dataset(grid.files[1979]) as file:
        samples = [grid.sample(0, cell, file["CH4_emission"]) for cell in range(4)]
    assert [row["quality_mask"] for row in samples] == [False, True, False, False]
    assert [row["filter_reason"] for row in samples] == ["negative_emission", None, "missing_label", "missing_label"]
    assert [row["target"] for row in samples[:2]] == [-1.25, 0.0]


def test_memo_uses_calendar_days_and_negative_uptake(tmp_path):
    path = tmp_path / data.ADAPTERS["memo_uptake"]["path"]
    # February 1999 and February 2000 in the native 1-based month index.
    _grid_file(path, {"Time": [110, 122, 123], "Lat": [0], "Lon": [0]}, "CH4uptake", [[[28]], [[29]], [[-1]]])
    inputs = data.product_inputs(_grid_config(tmp_path), ["memo_uptake"])["memo_uptake"]
    grid = data.GridProduct("memo_uptake", path, inputs["options"], data._date_range(inputs["date_range"]))
    with Dataset(path) as file:
        samples = [grid.sample(index, 0, file["CH4uptake"]) for index in range(3)]
    assert [(row["query"].end_time - row["query"].start_time).days for row in samples] == [28, 29, 31]
    assert [row["target"] for row in samples[:2]] == [-1.0, -1.0]
    assert samples[2]["filter_reason"] == "negative_uptake_amount"


def test_tem_uptake_orients_the_field_and_keeps_both_signs(tmp_path):
    path = tmp_path / data.ADAPTERS["tem_uptake"]["path"]
    axes = {"month": [1, 2, 3], "year": [1978, 2000, 2001], "longitude": np.arange(-180, -178, .5),
            "latitude": [90, 89.5, 89, 88.5]}
    values = np.ones(tuple(map(len, axes.values())))
    values[0] *= -2
    values[1, :, 0] = 0
    values[:, 0] = -100
    _grid_file(path, axes, "ch4_oxidation", values)
    config = {**_grid_config(tmp_path), "split_manifest_dir": tmp_path / "splits"}
    snapshot = prepare_manifest(config, ["tem_uptake"], "pretrain")
    _, manifest = snapshot
    product = manifest["products"]["tem_uptake"]
    assert product["input_stats"]["applied_sign"] == -1
    assert set(product["input_stats"]["by_year"]) == {"2000", "2001"}
    for segments, _ in loader_partitions([snapshot]):
        for segment in segments:
            try:
                assert {segment[index]["target"] for index in range(len(segment))} == {-1, 0, 2}
                assert segment[0]["query"].latitude_bounds == (89.5, 90.0)
            finally:
                segment.close()
    assert set(product["splits"]) == {"train", "validation"}


def _chamber(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("site_unique_id", "lat", "lon", "dataset_no", "wetland", "FCH4_sample_year",
                         "FCH4_sample_month", "FCH4_sample_day", "FCH4_daily_mean", "FCH4_monthly_processed"))
        for site, lat, year, month, target in rows:
            writer.writerow((site, lat, 0, 1, "no", year, month, 1, -500, target))


def _observation_config(path):
    return {"observations_dir": path, "split_manifest_dir": path / "splits", "split_seed": 171}


@pytest.mark.parametrize(("unit", "factor"), [("nmol_ch4_m2_s", 1.386072), (data.FLUX_UNIT, 1.0)])
def test_fluxnet_units_qc_and_screening(tmp_path, unit, factor):
    root = tmp_path / "FLUXNET-CH4"
    root.mkdir()
    (root / "FLUXNET_CH4_2024.csv").write_text(
        "SITE_ID,LOCATION_LAT,LOCATION_LONG,IGBP\nUS-Wet,40,-90,WET\nUS-Crop,41,-90,CRO\n"
    )
    (root / "FLUXNET_T1_DD.csv").write_text(
        "Site,TIMESTAMP,FCH4_F_ANNOPTLM,FCH4_F,FCH4_F_ANNOPTLM_QC\n"
        "US.Wet,2020-01-01,2,-9999,3\nUS.Wet,2020-01-02,0,0,1\n"
        "US.Wet,2020-01-03,-2,-2,1\nUS.Wet,2020-01-04,-9999,-9999,-9999\nUS.Crop,2020-01-01,3,3,1\n"
    )
    config = {**_observation_config(tmp_path), "adapters": {"fluxnet": {"input_unit": unit}}}
    _, manifest = prepare_manifest(config, ["FLUXNET"], "finetune")
    product = manifest["products"]["fluxnet"]
    assert product["input_stats"]["input_rows"]["excluded_site_class"] == 1
    assert product["input_stats"]["qc_columns"] == ["FCH4_F_ANNOPTLM_QC"]
    assert [row["target"] for row in product["splits"]["train"]["members"]] == pytest.approx([2 * factor, 0])
    rejected = product["input_stats"]["scoped_unassigned_rows"]
    assert rejected["filtered_missing_label"] == rejected["filtered_negative_emission"] == 1
    counts = product["splits"]["train"]["stats"]["row_counts"]
    assert counts["annoptlm_available_when_fch4_f_missing"] == counts["qc:FCH4_F_ANNOPTLM_QC:3"] == 1


def test_spatial_chamber_keeps_sites_and_excludes_heldout_copies(tmp_path):
    table = tmp_path / "chamber_processed.csv"
    _chamber(table, [("A", 10, 2012, 1, 0), ("A", 10, 2012, 2, 2),
                     ("A", 10, 2012, 2, 2), ("B", 20, 2013, 1, -1),
                     ("C", 30, 2014, 1, 1), ("D", 40, 2016, 1, -3),
                     ("E", 10, 2012, 2, 2), ("E", 10, 2020, 1, 99)])
    config = {**_observation_config(tmp_path), "observation_split_mode": "spatial"}
    _, manifest = prepare_manifest(config, ["Chamber"], "finetune")
    chamber = manifest["products"]["chamber"]
    assert chamber["site_roles"] == {
        "a": "train", "b": "train", "c": "train", "d": "validation", "e": "testing",
    }
    splits = chamber["splits"]
    assert {row["target"] for row in splits["train"]["members"]} == {-1, 0, 1}
    assert {row["target"] for row in splits["testing"]["members"]} == {2, 99}
    assert splits["train"]["stats"]["row_counts"]["repeated_rows"] == 1
    assert splits["train"]["stats"]["row_counts"]["filtered_duplicate_held_out_support"] == 1
    assert all(not any(metrics.values()) for metrics in manifest["split_intersections"].values())
    assert manifest["location_diagnostics"]["chamber"]["train/testing"]["nearest_km"] == 0

    restricted = {**config, "split_manifest_dir": tmp_path / "nonpositive",
                  "adapters": {"chamber": {"nonpositive_uptake": True}}}
    _, filtered = prepare_manifest(restricted, ["Chamber"], "finetune")
    assert all(row["target"] <= 0 for split in filtered["products"]["chamber"]["splits"].values()
               for row in split["members"])


def test_training_seed_and_replay_use_the_frozen_fitting_partitions(tmp_path, monkeypatch):
    _chamber(tmp_path / "chamber_processed.csv",
             [("S", 1, 2010 + i, 1, float(i)) for i in range(5)])
    config = {
        "data": _observation_config(tmp_path),
        "training": {"common": {"training_seed": 3}, "finetune": {"sources": ["Chamber"]}},
    }
    before = run_training(config, "finetune", prepare_data=True)
    different_seed = deepcopy(config)
    different_seed["training"]["common"]["training_seed"] = 131
    assert run_training(different_seed, "finetune", prepare_data=True) == before
    active, _, _, _ = stage_manifests(config["data"], ["Chamber"], "finetune", None, resume=False, replay_sources=[])
    past_data = {"active_manifest": active[0], "historical_manifests": []}
    checkpoint = tmp_path / "finetune.pt"
    torch.save({"format_version": 1, "stage": "finetune", "progress": {"data_provenance": past_data}}, checkpoint)
    _chamber(tmp_path / "future.csv", [("S", 1, 2021, 5, 10), ("S", 1, 2022, 4, 11),
                                       ("S", 1, 2023, 2, 200)])
    future = {**config["data"], "observation_cohort": "future",
              "adapters": {"chamber": {"path": "future.csv"}}}
    different_seed["training"]["continual"] = {"sources": ["Chamber"], "replay_sources": ["Chamber"],
                                                "data": future}
    summary = run_training(different_seed, "continual", checkpoint=checkpoint, prepare_data=True)
    assert summary["lineage_intersections"] == {
        "historical_train/replay_validation": 0, "historical_train/testing": 0,
    }
    new, history, names, _ = stage_manifests(future, ["Chamber"], "continual", past_data, resume=False,
                                             replay_sources=["Chamber"])
    assert new[1]["products"]["chamber"]["sites"]["testing"]["s"]["labels"] == 1
    for (segments, provenance), target in zip(loader_partitions([new]), (10, 11)):
        assert [row["target"] for segment in segments for row in segment] == [target]
        assert provenance["products"]["chamber"]["valid_labels"] == 1
    monkeypatch.setattr(data, "_collator", lambda *_: list)
    replay = loader_partitions(history, names)
    for seed in (3, 131):
        train, validation = data.build_loaders({}, replay, batch_size=4, training_seed=seed)
        assert train.generator.initial_seed() == seed
        assert {record["target"] for record in next(iter(train))} == {0, 1, 2}
        assert {record["target"] for record in next(iter(validation))} == {3}
        assert set(train.dataset.provenance["products"]["chamber"]) == {"target_type", "valid_labels"}

    duplicate = {**config["data"], "observation_cohort": "duplicate"}
    with pytest.raises(ValueError, match="overlap prior labels"):
        stage_manifests(duplicate, ["Chamber"], "continual", past_data, resume=False, replay_sources=["Chamber"])


def test_full_simulation_ignores_debug_caps_and_resumes_exactly(tmp_path):
    path = tmp_path / data.ADAPTERS["memo_uptake"]["path"]
    axes = {"Time": list(range(1, 61)), "Lat": np.arange(20.5, 42.5), "Lon": np.arange(-110.5, -88.5)}
    values = np.ones((60, 22, 22))
    values[0, 0, 0], values[0, 0, 1], values[12, 0, 0] = np.nan, 0, -1
    _grid_file(path, axes, "CH4uptake", values)
    config = {**_grid_config(tmp_path), "split_manifest_dir": tmp_path / "splits",
              "simulation_tile_mib": .02, "samples_per_simulation_source": {"train": 1, "validation": 1}}
    snapshot = prepare_manifest(config, ["MeMo"], "pretrain")
    regenerated = run_training({
        "data": {**config, "split_manifest_dir": tmp_path / "regenerated"},
        "training": {"pretrain": {"sources": ["MeMo"], "training_seed": 4321}},
    }, "pretrain", prepare_data=True)
    assert regenerated["manifest"]["sha256"] == snapshot[0]["sha256"]
    partition = snapshot[1]["products"]["memo_uptake"]["splits"]
    assert [partition[split]["stats"]["valid_labels"] for split in ("train", "validation")] == [23230, 5808]
    assert all("members" not in item for item in partition.values())
    for index, (segments, _) in enumerate(loader_partitions([snapshot])):
        segment = segments[0]
        try:
            sampler = data.ShardBatchSampler(segment.shard_lengths, 8, shuffle=index == 0, seed=3)
            batches = list(sampler)
            assert len(segment.shard_lengths) > 1
            assert sorted(chain.from_iterable(batches)) == list(range(len(segment)))
            sampler.set_progress(3, 157)
            assert list(sampler) == batches[157:]
            seen = {data.cache_query_key(segment[i]["query"]) for i in chain.from_iterable(batches)}
            assert len(seen) == len(segment)
            expected = 1994 if index else 1990
            assert min(segment[i]["query"].start_time.year for i in range(len(segment))) == expected
            assert segment[0]["query"].delta_lat == 1
            assert segment[0]["query"].lat in axes["Lat"]
        finally:
            segment.close()


def test_full_spatial_partition_uses_corrected_tem_cells(tmp_path):
    path = tmp_path / data.ADAPTERS["tem_uptake"]["path"]
    axes = {"month": [1], "year": [2000], "longitude": np.arange(-110, -86, .5),
            "latitude": np.arange(54.5, 34.5, -.5)}
    values = -np.arange(1, 1921, dtype=float).reshape(1, 1, 48, 40)
    _grid_file(path, axes, "ch4_oxidation", values)
    config = {**_grid_config(tmp_path), "split_manifest_dir": tmp_path / "splits",
              "simulation_split_mode": "spatial"}
    snapshot = prepare_manifest(config, ["tem_uptake"], "pretrain")
    expected = snapshot[1]["spatial_blocks"]
    count = 0
    for segments, provenance in loader_partitions([snapshot]):
        segment = segments[0]
        assert len(segment)
        try:
            for i in range(len(segment)):
                sample = segment[i]
                query = sample["query"]
                assert int(block_ids(query.lat, query.lon, 5)) in expected[provenance["split"]]
                yi = int(np.flatnonzero(axes["latitude"] == query.lat + .25)[0])
                xi = int(np.flatnonzero(axes["longitude"] == query.lon - .25)[0])
                assert sample["target"] == values[0, 0, xi, yi]
            count += len(segment)
        finally:
            segment.close()
    assert count == values.size


def test_temporal_groups_sort_by_period_and_keep_chamber_zero(tmp_path):
    rows = [("A", 41, 2020, m, value) for m, value in zip((5, 1, 4, 3, 2), (5, 0, 4, 3, 2))]
    rows.extend([("A", 41, 2020, 1, 0), ("Short", 42, 2020, 2, 22)])
    _chamber(tmp_path / "chamber_processed.csv", rows)
    config = _observation_config(tmp_path)
    _, manifest = prepare_manifest(config, ["Chamber"], "finetune")
    product = manifest["products"]["chamber"]
    assert product["input_stats"]["short_sites"]["labels"] == 1
    assert product["splits"]["train"]["stats"]["row_counts"]["repeated_rows"] == 1
    for split, months in {"train": [1, 2, 3], "validation": [4], "testing": [5]}.items():
        saved = product["splits"][split]["members"]
        assert sorted(int(row["query"][2][5:7]) for row in saved if row["site_id"] == "A") == months
    assert 0 in [row["target"] for row in product["splits"]["train"]["members"]]
    assert manifest["split_intersections"]["train/testing"]["shared_sites"] == 1
    assert manifest["split_intersections"]["train/testing"]["nonchronological_sites"] == 0

    restricted = {**config, "split_manifest_dir": tmp_path / "exclude", "observation_short_sites": "exclude"}
    _, excluded = prepare_manifest(restricted, ["Chamber"], "finetune")
    assert excluded["products"]["chamber"]["input_stats"]["scoped_unassigned_rows"]["filtered_short_site"] == 1


def test_fluxnet_temporal_daily_duplicate_is_a_single_group(tmp_path):
    root = tmp_path / "FLUXNET-CH4"
    root.mkdir()
    (root / "FLUXNET_CH4_2024.csv").write_text("SITE_ID,LOCATION_LAT,LOCATION_LONG,IGBP\nW,40,-90,WET\n")
    (root / "FLUXNET_T1_DD.csv").write_text(
        "Site,TIMESTAMP,FCH4_F_ANNOPTLM\n" +
        "".join(f"W,2020-01-0{day},{day}\n" for day in (4, 2, 5, 1, 3, 4))
    )
    _, manifest = prepare_manifest(_observation_config(tmp_path), ["FLUXNET"], "finetune")
    splits = manifest["products"]["fluxnet"]["splits"]
    assert [s["stats"]["valid_labels"] for s in map(splits.get, ("train", "validation", "testing"))] == [3, 2, 1]
    assert all(part["stats"]["sites"] == 1 for part in splits.values())
    assert all(item["exact_queries"] == item["nonchronological_sites"] == 0
               for item in manifest["split_intersections"].values())


def test_old_manifest_and_old_cache_are_rejected(tmp_path):
    old_manifest = tmp_path / "simulation.json"
    old_manifest.write_text('{"format_version":1}')
    with pytest.raises(ValueError, match="Unsupported split manifest"):
        _read_snapshot(old_manifest)
    metadata = {key: "old" for key in data.CACHE_PROVENANCE_KEYS}
    cache = tmp_path / "latents.pt"
    torch.save({"metadata": {**metadata, "gather_mode": "history"}, "rows": {}}, cache)
    with pytest.raises(ValueError, match="predates the corrected grid geometry"):
        data._collator({"embedding_mode": "cached", "cache_path": cache}, None, metadata)


@pytest.mark.parametrize("interruption_split", ["train", "validation"])
def test_pretraining_resumes_an_incomplete_data_pass(tmp_path, monkeypatch, interruption_split):
    class FrozenEncoder(torch.nn.Module):
        provenance = {}
        adapted = trainable = False

        def accept_cached(self, latent):
            return latent

    path = tmp_path / data.ADAPTERS["memo_uptake"]["path"]
    _grid_file(path, {"Time": [1, 13, 25], "Lat": [41.5, 42.5], "Lon": [0.5, 1.5]}, "CH4uptake",
               np.arange(1, 13).reshape(3, 2, 2))
    config = {**_grid_config(tmp_path), "split_manifest_dir": tmp_path / "splits"}
    snapshot = prepare_manifest(config, ["MeMo"], "pretrain")
    partitions = loader_partitions([snapshot])
    seen = []
    cache = {}
    for segments, _ in partitions:
        segment = segments[0]
        for index in range(len(segment)):
            query = segment[index]["query"]
            cache[data.cache_query_key(query)] = torch.ones(4)
        segment.close()
    collate = data.MoECollator(cache=cache, latent_dim=4)

    def track(samples):
        seen.extend(data.cache_query_key(row["query"]) for row in samples)
        return collate(samples)

    monkeypatch.setattr(data, "_collator", lambda *_: track)
    loaders = data.build_loaders({}, partitions, batch_size=2, training_seed=9)
    options = {"training_seed": 9, "epochs": 1, "checkpoint_interval_steps": 1, "anchor_count": 2}
    encoder, criterion = FrozenEncoder(), MoELoss()
    save = training.save_checkpoint

    def interrupt(*args, **kwargs):
        save(*args, **kwargs)
        scan = kwargs["progress"].get("data_scan")
        if scan and scan["split"] == interruption_split:
            raise InterruptedError("stop after a persisted batch")

    monkeypatch.setattr(training, "save_checkpoint", interrupt)
    model = ContinualMethaneMoE(latent_dim=4, initial_num_experts=2, expert_hidden_dims=[4], shared_hidden_dims=[4])
    output = tmp_path / "run"
    with pytest.raises(InterruptedError):
        training._fit(model, encoder, criterion, loaders, None, {}, options, "pretrain", None, None, None,
                      torch.device("cpu"), None, output, [], {})
    checkpoint = load_checkpoint(output / "last.pt")
    assert checkpoint["progress"]["data_scan"]["batches"] == 1
    monkeypatch.setattr(training, "save_checkpoint", save)
    train_history = []
    try:
        training._fit(restore_model(checkpoint), encoder, criterion, loaders, None, {}, options, "pretrain", None,
                      checkpoint, None, torch.device("cpu"), None, output, train_history, {})
    finally:
        for loader in loaders:
            loader.dataset.close()
    assert train_history[-1]["train"]["samples"] == 8
    assert train_history[-1]["validation"]["samples"] == 4
    assert len(seen) == len(set(seen)) == 12


@pytest.mark.parametrize(("product", "raw_lat", "raw_lon", "value"), [
    ("tem_emission", 40.0, -100.0, 4166.9),
    ("tem_uptake", 40.5, -107.5, -0.39024143397705635),
])
def test_real_tem_query_centers_match_native_labels_and_era5(product, raw_lat, raw_lon, value):
    data_dir = Path("data/methane_data/simulations")
    native = data_dir / data.ADAPTERS[product]["path"].format(year=1980)
    if not native.is_file() or not Path("ERA5_daily/processed/daily_ERA5_TAIR_1980.nc").is_file():
        pytest.skip("Local TEM and ERA5 files are required for the real coordinate regression.")
    inputs = data.product_inputs({"simulation_dir": data_dir}, [product])[product]
    grid = data.GridProduct(product, Path(inputs["path"]), inputs["options"], data._date_range(inputs["date_range"]))
    y, x = int(np.flatnonzero(grid.raw_lat == raw_lat)[0]), int(np.flatnonzero(grid.raw_lon == raw_lon)[0])
    t = next(i for i, (date, _, _) in enumerate(grid.dates) if date == datetime(1980, 7, 1))
    _, year, time_indices = grid.dates[t]
    with Dataset(grid.files[year]) as file:
        variable = file[grid.options["variable"]]
        raw = variable[y, x, *time_indices] if product == "tem_emission" else variable[*time_indices, x, y]
        saved = grid.sample(t, y * len(grid.lon) + x, variable)
        assert saved["target"] == raw == value
    query = saved["query"]
    assert (query.lat, query.lon, query.delta_lat, query.delta_lon) == (40.25, raw_lon + .25, .5, .5)
    assert (float(grid.lat.min()), float(grid.lat.max())) == (-89.75, 89.75)
    assert (float(grid.lon.min()), float(grid.lon.max())) == (-179.75, 179.75)
    source = load_config("encoding/configs/environmental_jepa.yaml")["registry"]["sources"][0]
    registry = DriverRegistry.from_dict({"sources": [source]})
    sources = build_data_sources(registry)
    try:
        context = MultiResolutionGatherer(registry, sources, {"history_days": 2}).gather(query)
        pairs = np.unique(context.blocks[0].metadata[:, :2], axis=0)
        assert len(pairs) == 9
        assert np.array_equal(np.unique(pairs[:, 0]), [40.0, 40.25, 40.5])
        assert np.array_equal(np.unique(pairs[:, 1]), [raw_lon, raw_lon + .25, raw_lon + .5])
    finally:
        for source in sources.values():
            source.close()
