from copy import deepcopy
import os
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from encoding import DataSourceSpec, DriverRegistry, DriverSpec, EnvironmentalEncoder, QuerySpec
from encoding.model import save_encoder
from moe.checkpoint import load_checkpoint, restore_encoder, restore_model, restore_rng, save_checkpoint
from moe.data import _collator, save_embedding_cache
from moe.encoder import EnvironmentalEmbedding, driver_inventory, load_environmental_encoder, weights_fingerprint
from moe.growth import CandidateTrainer, GrowthManager
from moe.model import TARGET_TYPES, ContinualMethaneMoE, MoELoss, physical_response_loss, task_mask
from moe.training import AnchorReservoir, _set_training_temperature, _settings, _teacher_snapshot, _teacher_state, run_epoch


@pytest.fixture(autouse=True)
def random_seed():
    torch.manual_seed(19)


def small_model(**options):
    return ContinualMethaneMoE(expert_hidden_dims=(12, 8), shared_hidden_dims=(12,), **options)


def batch_for(latent, kind="emission"):
    return {
        "target": torch.randn(len(latent)),
        "target_type": [kind] * len(latent),
        "source": ["test"] * len(latent),
        "quality_mask": torch.ones(len(latent), dtype=torch.bool),
        "sample_weight": torch.ones(len(latent)),
    }


def encoder_fixture(path=""):
    registry = DriverRegistry(
        [DriverSpec("temperature", "daily", path=str(path))],
        [DataSourceSpec("daily", (0.25, 0.25), "daily", ("latitude", "longitude", "time"))],
    )
    encoder = EnvironmentalEncoder(registry, hidden_dim=16, latent_dim=128, dropout=0.2)
    provenance = {"weights_sha256": weights_fingerprint(encoder.state_dict()), "query_version": "test",
                  "preprocessing_version": "test", "sampling": {}, "drivers": driver_inventory(registry) if path else {},
                  "model_config": {"latent_dim": 128}}
    return EnvironmentalEmbedding(encoder, {}, provenance)


def test_forward_mask_and_backward():
    model = small_model()
    latent = torch.randn(8, 128)
    output = model(latent)
    assert set(output) == {*TARGET_TYPES, "routing", "latent", "expert_outputs", "base"}
    for name in TARGET_TYPES:
        assert output[name].shape == (8,)
    assert output["latent"].shape == (8, 128)
    assert output["expert_outputs"].shape == (8, 2, 4)
    assert output["routing"].shape == (8, 2, 4)
    assert all(torch.isfinite(tensor).all() for tensor in output.values())
    torch.testing.assert_close(output["routing"].sum(-1), torch.ones(8, 2))
    assert len({next(expert.parameters()).data_ptr() for expert in model.experts}) == 4

    batch = batch_for(latent)
    batch["target"][0] = torch.nan
    batch["quality_mask"][1] = False
    batch["sample_weight"][2] = 0
    result = MoELoss(kind="mse")(output, batch)
    torch.testing.assert_close(result["data_loss"], (output["emission"][3:] - batch["target"][3:]).square().mean())
    result["loss"].backward()
    for module in (model.shared, model.router, *model.experts):
        grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert any(grad.abs().sum() > 0 for grad in grads)
    assert torch.count_nonzero(model.shared[-1].weight.grad[1]) == 0
    for branch in model.router.branches:
        assert branch[0][-1].weight.grad.norm() > 0
        assert branch[1][-1].weight.grad.norm() == 0


@pytest.mark.parametrize("temperature", [0.3, 1.0, 2.3])
def test_signature_and_function_preserving_split(temperature):
    model = small_model(routing_temperature=temperature, router_hidden_dims=[8], dropout=0.2).eval()
    latent = torch.randn(8, 128)
    batch = batch_for(latent)
    batch["target_type"][4:] = ["uptake"] * 4
    mask = task_mask(batch)
    routing = model(latent)["routing"].detach()
    for idx in (slice(0, 4), slice(4, 8)):
        model.signatures.update(latent[idx], routing[idx], mask[idx])
    weights = routing * mask[..., None]
    mass = weights.sum(0)
    mean = torch.einsum("bqk,bd->qkd", weights, latent) / mass.clamp_min(1e-10)[..., None]
    var = (weights[..., None] * (latent[:, None, None, :] - mean).square()).sum(0)
    var = var / mass.clamp_min(1e-10)[..., None]
    torch.testing.assert_close(model.signatures.weight, mass)
    torch.testing.assert_close(model.signatures.mean, mean)
    torch.testing.assert_close(model.signatures.variance, var)

    model.signatures.set_anchors("emission", latent[:3], {"version": "test"})
    model.refresh_fingerprints()
    assert torch.isfinite(model.signatures.fingerprint).all()
    query_latent = torch.randn(8, 128)
    before = model(query_latent)
    model.signature_guided_routing = False
    pure = model(query_latent)
    model.signature_guided_routing = True
    assert not torch.allclose(before["routing"], pure["routing"])
    original = deepcopy(model)
    parent = int(before["routing"].sum(dim=(0, 1)).argmax())
    child = model.split_expert(parent)
    for first, second in zip(model.experts[parent].parameters(), model.experts[child].parameters()):
        torch.testing.assert_close(first, second)
        assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(model.signatures.mean[:, parent], model.signatures.mean[:, child])
    after = model(query_latent)
    for key in TARGET_TYPES:
        torch.testing.assert_close(after[key], before[key], atol=1e-6, rtol=0)
    torch.testing.assert_close(after["routing"][..., parent] + after["routing"][..., child],
                               before["routing"][..., parent], atol=1e-6, rtol=0)
    model.split_expert(child)
    for other_temperature in (temperature, 0.55):
        model.router.set_temperature(other_temperature)
        original.router.set_temperature(other_temperature)
        for key in TARGET_TYPES:
            torch.testing.assert_close(model(query_latent)[key], original(query_latent)[key], atol=1e-6, rtol=0)
    assert all(torch.isfinite(tensor).all() for tensor in model(query_latent).values())


def test_legacy_checkpoint_uses_product_coverage(tmp_path):
    driver = tmp_path / "driver.nc"
    driver.write_bytes(b"driver")
    encoder = encoder_fixture(driver).encoder
    path = tmp_path / "encoder.pt"
    save_encoder(path, encoder)
    original = {"spatial_selection": "centers", "max_tokens": 4096, "history_days": 14}
    torch.save({"epoch": 4, "history": [{"epoch": 4, "validation": {"jepa_loss": 0.1}}],
                "model": {f"online_encoder.{key}": value for key, value in encoder.state_dict().items()},
                "config": {"registry": encoder.registry.to_dict(), "model": encoder.to_config(), "sampling": original}},
               tmp_path / "last.pt")
    options = {"checkpoint": str(path)}
    product = load_environmental_encoder(options)
    assert product.sampling == {"spatial_selection": "intersects", "max_tokens": 5120, "history_days": 14}
    assert load_environmental_encoder({**options, "spatial_selection": "centers"}).sampling == original
    old = {**product.provenance, "query_version": "encoding-query-spec-v1"}
    with pytest.raises(ValueError, match="query_version"):
        load_environmental_encoder(options, expected_provenance=old)


def test_online_encoder_stays_frozen_through_optimizer_steps():
    encoder = encoder_fixture()
    values = torch.randn(2, 3, 1)
    query = torch.from_numpy(QuerySpec(42.0, -75.0, "2020-01-01", "2020-01-02").metadata())
    inputs = {
        "values": values,
        "variable_mask": torch.ones_like(values, dtype=torch.bool),
        "metadata": query.expand(2, 3, -1).clone(),
        "static_mask": torch.zeros(2, 3, dtype=torch.bool),
        "source_ids": torch.zeros(2, 3, dtype=torch.long),
        "token_mask": torch.ones(2, 3, dtype=torch.bool),
        "query": query.expand(2, -1).clone(),
    }
    frozen = encoder(inputs)
    assert frozen.shape == (2, 128)
    assert not frozen.requires_grad and torch.isfinite(frozen).all()
    torch.testing.assert_close(encoder.accept_cached(frozen), frozen)
    encoder.train()
    assert all(not module.training for module in encoder.modules())
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    before = deepcopy(encoder.state_dict())
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = {**batch_for(frozen), "environment": inputs}
    assert run_epoch(model, encoder, MoELoss(), [batch], torch.device("cpu"), None,
                     optimizer=optimizer)["optimizer_steps"] == 1
    torch.testing.assert_close(encoder.state_dict(), before, atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in encoder.parameters())
    values.requires_grad_()
    reference = torch.randn_like(values)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        output = model(encoder(inputs))
        response_loss = physical_response_loss(output["emission"], values, reference)
    response_loss.backward()
    assert torch.isfinite(response_loss)
    assert model.shared[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in encoder.parameters())
    with pytest.raises(ValueError, match="connected to raw drivers"):
        physical_response_loss(output["emission"], values.detach(), reference)


def test_checkpoint_after_growth(tmp_path):
    model = small_model().eval()
    latent = torch.randn(6, 128)
    batch = batch_for(latent)
    model.signatures.update(latent, model(latent)["routing"], task_mask(batch))
    model.signatures.set_anchors("emission", latent, {"version": "test"})
    model.refresh_fingerprints()
    model.split_expert(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = MoELoss()
    optimizer.zero_grad()
    criterion(model(latent), batch)["loss"].backward()
    optimizer.step()
    growth = {"history": [{"expert_id": "expert_0004", "status": "accepted"}]}
    encoder = encoder_fixture()
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, encoder, {}, stage="continual", criterion=criterion,
                    optimizer=optimizer, growth=growth, progress={"epoch": 1})
    payload = load_checkpoint(path)
    restored = restore_model(payload).eval()
    restore_encoder(deepcopy(encoder), payload)
    assert restored.num_experts == 5 and restored.metadata() == model.metadata()
    assert restored.signatures.anchor_metadata == model.signatures.anchor_metadata
    torch.testing.assert_close(restored.signatures.fingerprint, model.signatures.fingerprint)
    torch.testing.assert_close(restored.signatures.variance, model.signatures.variance)
    torch.testing.assert_close(restored(latent), model(latent))
    assert payload["growth"] == growth and payload["progress"]["epoch"] == 1
    resumed = torch.optim.AdamW(restored.parameters())
    resumed.load_state_dict(payload["optimizer"])
    child_weight = restored.experts[-1][0].weight
    assert any(parameter is child_weight for group in resumed.param_groups for parameter in group["params"])
    assert resumed.state
    expected = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng(payload["rng"], torch.device("cpu"))
    baseline = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng(payload["rng"], torch.device("cpu"))
    assert baseline == (random.random(), np.random.rand(), torch.rand(1))
    assert expected != baseline


def test_candidate_training_and_growth_decisions():
    model = small_model().eval()
    latent = torch.randn(8, 128)
    batch = batch_for(latent)
    criterion = MoELoss()
    original = deepcopy(model.state_dict())
    manager = GrowthManager(min_samples=4, min_validation_samples=4, min_replay_samples=4,
                            persistence=2, distance_threshold=0.0, loss_threshold=0.0)
    assert not manager.observe(model, model(latent), batch, criterion)["triggered"]
    proposed = manager.observe(model, model(latent), batch, criterion)
    assert proposed["triggered"] and proposed["old_loss"] > 0
    assert len(proposed["latent_distances"]) == len(proposed["response_losses"]) == 4

    parent = proposed["parent"]
    candidate = CandidateTrainer(model, parent, expert_id=proposed["candidate_id"])
    child = candidate.index
    before = candidate.model.experts[child][0].weight.detach().clone()
    parent_before = candidate.model.experts[parent][0].weight.detach().clone()
    optimizer = torch.optim.AdamW(candidate.trainable_parameters(), lr=0.001)
    criterion(candidate.model(latent), batch)["loss"].backward()
    assert candidate.model.router.branches[child][0][-1].weight.grad.abs().sum() > 0
    assert candidate.model.router.branches[parent][0][-1].weight.grad is None
    optimizer.step()
    assert not torch.equal(candidate.model.experts[child][0].weight, before)
    assert torch.equal(candidate.model.experts[parent][0].weight, parent_before)
    assert candidate.model.experts[parent][0].weight.grad is None

    selected, pending = manager.decide(model, candidate, new_count=3, replay_count=8, proposal=proposed)
    assert selected is model and pending["status"] == "pending_insufficient_data"
    selected, rejected = manager.decide(model, candidate, new_count=8, replay_count=8,
                                        new_old_loss=1.0, new_candidate_loss=0.5,
                                        replay_old_loss=1.0, replay_candidate_loss=1.1, proposal=proposed)
    assert selected is model and rejected["status"] == "rejected"
    torch.testing.assert_close(model.state_dict(), original)
    selected, accepted = manager.decide(model, candidate, new_count=8, replay_count=8,
                                        new_old_loss=1.0, new_candidate_loss=0.5,
                                        replay_old_loss=1.0, replay_candidate_loss=0.9, proposal=proposed)
    assert selected is candidate.model and accepted["status"] == "accepted"
    assert selected.expert_records[child]["parent_expert_id"] == model.expert_records[parent]["expert_id"]
    assert selected.num_experts == 5 and selected.experts[parent][0].weight.requires_grad
    restored = GrowthManager()
    restored.load_state_dict(manager.state_dict())
    assert restored.history == manager.history


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_cached_adapter_and_single_training_step(tmp_path, dtype):
    encoder = encoder_fixture()
    model = small_model()
    criterion = MoELoss()
    latent = torch.randn(2, 128)
    queries = [
        QuerySpec(42.0, -75.0, "2020-01-01", "2020-01-02"),
        QuerySpec(40.0, -75.0, "2020-02-01", "2020-03-01", 0.5, 0.5),
    ]
    path = tmp_path / "latents"
    save_embedding_cache(path, zip(queries, latent), encoder.provenance, dtype=dtype)
    config = {"embedding_mode": "cached", "cache_path": path, "cache_dtype": dtype}
    collate = _collator(config, None, encoder.provenance)
    for key in ("weights_sha256", "query_version", "preprocessing_version", "sampling", "drivers"):
        with pytest.raises(ValueError, match=key):
            _collator(config, None, {**encoder.provenance, key: "different"})
    with pytest.raises(ValueError, match="gather_mode"):
        _collator({**config, "gather_mode": "query_window"}, None, encoder.provenance)
    samples = [
        {
            "query": query, "source": "test", "product": "test", "site_id": str(index),
            "spatial_support": (query.delta_lat, query.delta_lon),
            "temporal_support": (query.start_time, query.end_time),
            "target": float(target), "target_type": kind, "quality_mask": True,
            "sample_weight": 1.0,
        }
        for index, (query, target, kind) in enumerate(zip(queries, torch.randn(2), ("emission", "uptake")))
    ]
    batch = collate(samples)
    expected_latent = latent.half().float() if dtype == "float16" else latent
    torch.testing.assert_close(batch["latent"], expected_latent)
    changed = collate([{**row, "target": row["target"] + 1} for row in samples])
    torch.testing.assert_close(changed["latent"], batch["latent"])
    torch.testing.assert_close(changed["target"], batch["target"] + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    anchors = AnchorReservoir(2, 19, {}, encoder.provenance)
    metrics = run_epoch(model, encoder, criterion, [batch], torch.device("cpu"), None,
                        optimizer=optimizer, scaler=scaler, anchors=anchors, max_steps=1)
    anchors.apply(model)
    assert metrics["samples"] == 2
    assert all(row["rmse"] >= row["mae"] >= 0 for row in metrics["by_task"].values())
    assert set(metrics["expert_utilization"]) == {"emission", "uptake"}
    torch.testing.assert_close(model.signatures.anchors[:, 0], expected_latent)
    assert model.signatures.anchor_metadata[1]["samples"][0]["site_id"] == "1"
    collate.cache.close()


def test_two_task_normalization_and_site_equal_metrics():
    statistics = {"emission": {"mean": 100.0, "std": 10.0, "count": 100},
                  "uptake": {"mean": -5.0, "std": 2.0, "count": 20}}
    model = small_model(label_normalization=statistics)
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.zero_()
    model.eval()
    criterion = MoELoss(label_normalization=model.label_normalization, sign_weight=1.0,
                        task_weights={"emission": 2.0, "uptake": 3.0}, source_weights={"TEM": 0.5})
    latent = torch.zeros(4, model.latent_dim)
    batch = {**batch_for(latent), "latent": latent, "source": ["TEM"] * 4,
             "target_type": ["emission", "emission", "emission", "uptake"],
             "target": torch.tensor([100.0, 110.0, 120.0, -3.0]), "site_id": ["A", "A", "B", "C"]}
    result = criterion(model(latent), batch)
    torch.testing.assert_close(result["by_source"]["emission"]["TEM"], torch.tensor(5.0 / 3.0))
    torch.testing.assert_close(result["by_source"]["uptake"]["TEM"], torch.tensor(1.0))
    torch.testing.assert_close(result["data_loss"], torch.tensor(5.0 / 3.0 + 1.5))
    assert result["sign_loss"] == 0
    physical = model.predict_physical(latent)
    torch.testing.assert_close(physical["emission"], torch.full((4,), 100.0))
    torch.testing.assert_close(physical["uptake"], torch.full((4,), -5.0))
    metrics = run_epoch(model, encoder_fixture(), criterion, [batch], torch.device("cpu"), None)
    assert metrics["by_source"]["TEM"]["emission"]["rmse"] == pytest.approx((500.0 / 3.0) ** 0.5)
    assert metrics["by_source"]["TEM"]["uptake"]["rmse"] == pytest.approx(2)
    assert metrics["by_task"]["emission"]["r2"] == pytest.approx(1 - 500 / 200)
    assert metrics["site_equal"]["by_task"]["emission"]["normalized_mse"] == pytest.approx(2.25)
    assert metrics["site_equal"]["by_task"]["emission"]["sites"] == 2
    for invalid in (0, 1e-10, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="label normalization"):
            small_model(label_normalization={**statistics, "uptake": {"mean": 0, "std": invalid}})


@pytest.mark.parametrize("precision", [None, torch.bfloat16, torch.float16])
def test_nonfinite_gradient_skips_optimizer_and_signatures(precision):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = small_model(activation="relu").to(device)
    encoder = encoder_fixture().to(device)
    latent = torch.randn(4, 128)
    batch = {**batch_for(latent), "latent": latent}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler(device.type, init_scale=64, enabled=precision == torch.float16)
    before = deepcopy(model.state_dict())
    parameter = model.shared[-1].weight
    hook = parameter.register_hook(lambda gradient: torch.full_like(gradient, torch.inf))
    criterion = MoELoss().to(device)
    metrics = run_epoch(model, encoder, criterion, [batch], device, precision,
                        optimizer=optimizer, scaler=scaler)
    assert metrics["optimizer_steps"] == 0 and metrics["skipped_nonfinite_gradients"] == 1
    assert not optimizer.state
    torch.testing.assert_close(model.state_dict(), before, atol=0, rtol=0)
    if scaler.is_enabled():
        assert scaler.get_scale() < 64
    hook.remove()
    metrics = run_epoch(model, encoder, criterion, [batch], device, precision,
                        optimizer=optimizer, scaler=scaler)
    assert metrics["optimizer_steps"] == 1 and model.signatures.weight.sum() > 0
    with pytest.raises(FloatingPointError, match="loss is not finite"):
        run_epoch(model, encoder, criterion, [{**batch, "target": torch.full((4,), 1e30)}],
                  device, precision, optimizer=optimizer, scaler=scaler)


@pytest.mark.parametrize("replay, distill", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_replay_and_fixed_teacher(replay, distill):
    model = small_model()
    teacher = _teacher_snapshot(model) if distill else None
    initial_teacher = deepcopy(_teacher_state(teacher))
    latent = torch.randn(4, 128)
    batch = {**batch_for(latent), "latent": latent}
    replay_loader = DataLoader([{**batch, "target_type": ["uptake"] * len(latent)}], collate_fn=lambda rows: rows[0])
    opts = {"replay": {"simulation": {"weight": 0.5}}, "distillation_weight": 0.2 if distill else 0.0,
            "gradient_diagnostic_interval_steps": 1, "task_gradient_cosine": True}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    metrics = run_epoch(model, encoder_fixture(), MoELoss(), [batch] * 2, torch.device("cpu"), None,
                        optimizer=optimizer, options=opts, teacher=teacher,
                        replay_loaders={"simulation": replay_loader} if replay else None)
    assert ("replay_simulation_loss" in metrics) == replay
    assert ("distillation_loss" in metrics) == distill
    assert metrics["batches_by_task"] == {"emission": 2}
    gradients = metrics["gradient_diagnostics"]
    assert gradients["shared_supervised_grad_norm_emission"] > 0
    assert bool(gradients["shared_supervised_grad_norm_uptake"]) == replay
    assert ("task_gradient_cosine" in gradients) == replay
    if replay:
        assert metrics["replay_samples"]["simulation"]["test"]["uptake"] == 8
    if distill:
        assert metrics["distillation_loss"] > 0 and not teacher.training
        model.split_expert(0)
        restored = _teacher_snapshot(model, initial_teacher)
        assert restored.num_experts == teacher.num_experts < model.num_experts
        torch.testing.assert_close(_teacher_state(teacher)["model"], initial_teacher["model"], atol=0, rtol=0)


def test_cache_invalidates_after_in_place_driver_change(tmp_path):
    driver = tmp_path / "large-driver.nc"
    with driver.open("wb") as stream:
        stream.truncate(17 * 1024**2)
    encoder = encoder_fixture(driver)
    cache = save_embedding_cache(tmp_path / "cache", [], encoder.provenance)
    previous = driver.stat()
    with driver.open("r+b") as stream:
        stream.seek(2**23)
        stream.write(b"changed")
    os.utime(driver, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    changed = driver.stat()
    assert (previous.st_size, previous.st_mtime_ns) == (changed.st_size, changed.st_mtime_ns)
    new = {**encoder.provenance, "drivers": driver_inventory(encoder.registry)}
    with pytest.raises(ValueError, match="drivers"):
        _collator({"embedding_mode": "cached", "cache_path": cache}, None, new)


def test_stage_options_and_checkpoint_format(tmp_path):
    config = {"training": {"common": {"learning_rate": 0.01, "task_weights": {"emission": 0.6}},
                           "finetune": {"learning_rate": 0.001, "task_weights": {"uptake": 2.0}}}}
    _, _, _, options = _settings(config, "finetune")
    assert options["learning_rate"] == 0.001 and options["task_weights"] == {"emission": 0.6, "uptake": 2.0}
    assert options["loss"] == "mse" and not options["distillation_weight"]
    assert all(not value["weight"] for value in options["replay"].values())
    for incorrect in ({"encoder": {"freeze": False}}, {"training": {"finetune": {"phases": []}}}):
        with pytest.raises(ValueError, match="Unsupported"):
            _settings(incorrect, "finetune")
    model = small_model()
    _set_training_temperature(model, options, 4)
    assert model.router.temperature == 1.0
    options["temperature_schedule"] = {"kind": "linear", "end": 0.25, "steps": 8}
    for step, expected in ((0, 1.0), (4, 0.625), (8, 0.25), (12, 0.25)):
        _set_training_temperature(model, options, step)
        assert model.router.temperature == expected
    legacy = tmp_path / "old.pt"
    torch.save({"format_version": 1}, legacy)
    with pytest.raises(ValueError, match="existing JEPA checkpoint and Task 1 split manifests"):
        load_checkpoint(legacy)
