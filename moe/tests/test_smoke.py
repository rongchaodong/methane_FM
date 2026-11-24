from copy import deepcopy

import pytest
import torch

from encoding import DataSourceSpec, DriverRegistry, DriverSpec, EnvironmentalEncoder, QuerySpec
from encoding.model import save_encoder
from moe.checkpoint import load_checkpoint, restore_encoder, restore_model, save_checkpoint
from moe.data import _collator, save_embedding_cache
from moe.encoder import EnvironmentalEmbedding, load_environmental_encoder
from moe.growth import CandidateTrainer, GrowthManager
from moe.model import ContinualMethaneMoE, MoELoss, physical_response_loss, task_mask
from moe.training import AnchorReservoir, run_epoch


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


def encoder_fixture():
    registry = DriverRegistry(
        [DriverSpec("temperature", "daily")],
        [DataSourceSpec("daily", (0.25, 0.25), "daily", ("latitude", "longitude", "time"))],
    )
    encoder = EnvironmentalEncoder(registry, hidden_dim=16, latent_dim=128, dropout=0.2)
    provenance = {"checkpoint_sha256": "test", "query_version": "test",
                  "preprocessing_version": "test", "sampling": {},
                  "model_config": {"latent_dim": 128}}
    return EnvironmentalEmbedding(encoder, {}, provenance)


def test_forward_mask_and_backward():
    model = small_model()
    latent = torch.randn(8, 128)
    output = model(latent)
    for name in ("emission", "uptake", "net"):
        assert output[name].shape == (8,)
    assert output["latent"].shape == (8, 128)
    assert output["expert_outputs"].shape == (8, 2, 4)
    assert output["routing"].shape == (8, 2, 4)
    assert all(torch.isfinite(tensor).all() for tensor in output.values())
    torch.testing.assert_close(output["routing"].sum(-1), torch.ones(8, 2))
    torch.testing.assert_close(output["net"], output["emission"] + output["uptake"])
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


@pytest.mark.parametrize("routing_type", ["dense", "top_k"])
def test_signature_and_function_preserving_split(routing_type):
    model = small_model(routing_type=routing_type, routing_temperature=0.7, dropout=0.2).eval()
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
    parent = int(before["routing"].sum(dim=(0, 1)).argmax())
    child = model.split_expert(parent)
    for first, second in zip(model.experts[parent].parameters(), model.experts[child].parameters()):
        torch.testing.assert_close(first, second)
        assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(model.signatures.mean[:, parent], model.signatures.mean[:, child])
    after = model(query_latent)
    for key in ("emission", "uptake", "net"):
        torch.testing.assert_close(after[key], before[key], atol=1e-6, rtol=0)
    torch.testing.assert_close(after["routing"][..., parent] + after["routing"][..., child],
                               before["routing"][..., parent], atol=1e-6, rtol=0)
    model.split_expert(child)
    torch.testing.assert_close(model(query_latent)["net"], before["net"], atol=1e-6, rtol=0)
    assert all(torch.isfinite(tensor).all() for tensor in model(query_latent).values())


def test_legacy_checkpoint_uses_product_coverage(tmp_path):
    encoder = encoder_fixture().encoder
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
    with pytest.raises(ValueError, match="preprocessing"):
        load_environmental_encoder(options, expected_provenance=old)


def test_online_encoder_gradients_and_cache():
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
    encoder.set_trainable(["final_projection.*"])
    assert encoder.encoder.final_projection.weight.requires_grad
    assert not encoder.encoder.attention[0].in_proj_weight.requires_grad
    with pytest.raises(ValueError, match="Cached embeddings require"):
        encoder.accept_cached(frozen)
    encoder(inputs).square().mean().backward()
    assert encoder.encoder.final_projection.weight.grad.abs().sum() > 0
    assert encoder.encoder.attention[0].in_proj_weight.grad is None
    encoder.set_trainable()
    inputs["values"].requires_grad_()
    model = small_model()
    reference = torch.randn_like(values)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        output = model(encoder(inputs))
        response_loss = physical_response_loss(output["emission"], values, reference)
    response_loss.backward()
    assert torch.isfinite(response_loss)
    assert model.shared[-1].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="connected to raw drivers"):
        physical_response_loss(output["emission"], values.detach(), reference)


def test_checkpoint_after_growth(tmp_path):
    model = small_model(routing_type="top_k").eval()
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
    restore_encoder(encoder_fixture(), payload)
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
    assert candidate.model.router.branches[child][-1].weight.grad.abs().sum() > 0
    assert candidate.model.router.branches[parent][-1].weight.grad is None
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


def test_cached_adapter_and_single_training_step(tmp_path):
    encoder = encoder_fixture()
    model = small_model()
    criterion = MoELoss()
    latent = torch.randn(2, 128)
    queries = [
        QuerySpec(42.0, -75.0, "2020-01-01", "2020-01-02"),
        QuerySpec(40.0, -75.0, "2020-02-01", "2020-03-01", 0.5, 0.5),
    ]
    path = tmp_path / "latents.pt"
    save_embedding_cache(path, zip(queries, latent), encoder.provenance)
    config = {"embedding_mode": "cached", "cache_path": path}
    collate = _collator(config, None, encoder.provenance)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        _collator(config, None, {**encoder.provenance, "checkpoint_sha256": "different"})
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
    torch.testing.assert_close(batch["latent"], latent)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    anchors = AnchorReservoir(2, 19, {}, encoder.provenance)
    metrics = run_epoch(model, encoder, criterion, [batch], torch.device("cpu"), None,
                        optimizer=optimizer, scaler=scaler, anchors=anchors, max_steps=1)
    anchors.apply(model)
    assert metrics["samples"] == 2 and metrics["rmse"] >= metrics["mae"] >= 0
    assert set(metrics["expert_utilization"]) == {"emission", "uptake"}
    torch.testing.assert_close(model.signatures.anchors[:, 0], latent)
    assert model.signatures.anchor_metadata[1]["samples"][0]["site_id"] == "1"
