import hashlib
import inspect
import json
import re
from pathlib import Path

import torch
from torch import nn

from encoding import DriverRegistry, EnvironmentalEncoder, MultiResolutionGatherer, build_data_sources
from encoding.dataset import EnvironmentalCollator
from encoding.specs import DEFAULT_MAX_TOKENS
from encoding.training import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_VERSION = "environmental-jepa-online-v1"
QUERY_VERSION = "encoding-query-spec-v2"
ONLINE_PREFIX = "online_encoder."
EMBEDDING_VERSION_KEYS = ("weights_sha256", "query_version", "preprocessing_version", "sampling", "drivers")
ENCODER_OPTIONS = {"checkpoint", "expected_best_epoch", "best_metric", "checkpoint_dir", "latent_dim",
                   "jepa_config", "spatial_selection", "max_tokens"}


def weights_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def driver_inventory(registry, *, hash_small=True):
    """Version the actual inputs, including in-place edits that preserve the modification time."""
    inventory = {}
    for driver in registry.drivers.values():
        pattern = Path(re.sub(r"\{year[^}]*\}", "[0-9][0-9][0-9][0-9]", driver.path))
        paths = (sorted(pattern.parent.glob(pattern.name)) if "{year" in driver.path else [pattern])
        if not paths:
            raise FileNotFoundError(f"No environmental driver files match {driver.path}")
        files = []
        for path in paths:
            stat = path.stat()
            version = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                       "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino}
            if hash_small and stat.st_size <= 16 * 1024**2:
                version["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append(version)
        inventory[driver.name] = files
    return inventory


def preprocessing_fingerprint(registry, model_config):
    from .data import MoECollator

    digest = hashlib.sha256(json.dumps([registry, model_config]).encode())
    for name in ("specs.py", "sources.py", "projected.py", "gather.py", "model.py"):
        digest.update((REPO_ROOT / "encoding" / name).read_bytes())
    for function in (EnvironmentalCollator, MoECollator.gather_context):
        digest.update(inspect.getsource(function).encode())
    return digest.hexdigest()


def _validation_record(checkpoint, metric):
    records = [record for record in checkpoint.get("history", [])
               if record.get("validation") and metric in record["validation"]]
    if not records:
        return None
    return min(records, key=lambda record: record["validation"][metric])


def _resolve_checkpoint(config):
    requested = config.get("checkpoint", "auto")
    expected = config.get("expected_best_epoch", 4)
    metric = config.get("best_metric", "jepa_loss")
    if requested != "auto":
        return Path(requested), expected, metric
    if expected is None:
        raise ValueError("Auto JEPA checkpoint selection requires expected_best_epoch")
    root = Path(config.get("checkpoint_dir", REPO_ROOT / "encoding/model_save"))
    matches = []
    for export in sorted(root.glob("*/encoder.pt")):
        training = export.with_name("last.pt")
        if not training.is_file():
            continue
        state = torch.load(training, map_location="cpu", weights_only=True)
        best = _validation_record(state, metric)
        if best and state.get("epoch") == best["epoch"] == expected:
            matches.append(export)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one exported JEPA checkpoint from best epoch {expected} "
                         f"by validation {metric} in {root}, found: {matches}. Set encoder.checkpoint explicitly.")
    return matches[0], expected, metric


def _online_weights(training):
    return {key.removeprefix(ONLINE_PREFIX): value for key, value in training["model"].items()
            if key.startswith(ONLINE_PREFIX)}


class EnvironmentalEmbedding(nn.Module):
    def __init__(self, encoder, sampling, provenance):
        super().__init__()
        self.encoder, self.sampling, self.provenance = encoder, sampling, provenance
        self.registry = encoder.registry
        self.latent_dim = encoder.latent_dim
        self.encoder.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def make_gatherer(self):
        return MultiResolutionGatherer(self.registry, build_data_sources(self.registry), self.sampling)

    def forward(self, batch):
        # Physical response supervision differentiates with respect to raw drivers.
        if batch["values"].requires_grad:
            return self.encoder(batch)
        with torch.no_grad():
            return self.encoder(batch)

    def accept_cached(self, latent):
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Cached embeddings must have shape [B, {self.latent_dim}]")
        return latent.detach()


def load_environmental_encoder(config, expected_provenance=None):
    if unknown := config.keys() - ENCODER_OPTIONS:
        raise ValueError(f"Unsupported encoder options: {sorted(unknown)}")
    path, expected, metric = _resolve_checkpoint(config)
    item = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in item and "registry" in item:
        registry, model_config, weights = item["registry"], item["model_config"], item["state_dict"]
        companion = path.with_name("last.pt")
        training = torch.load(companion, map_location="cpu", weights_only=True) if companion.is_file() else None
        if training is not None:
            online = _online_weights(training)
            if weights.keys() != online.keys() or any(not torch.equal(weights[key], online[key]) for key in weights):
                raise ValueError(f"Export does not match the online context encoder in {companion}")
            training_registry = DriverRegistry.from_dict(training["config"]["registry"]).to_dict()
            if registry != training_registry or any(model_config[key] != value
                                                   for key, value in training["config"]["model"].items()):
                raise ValueError(f"Export configuration does not match {companion}")
    elif "model" in item and "config" in item:
        training = item
        registry = DriverRegistry.from_dict(item["config"]["registry"]).to_dict()
        model_config, weights = item["config"]["model"], _online_weights(item)
        if not weights:
            raise ValueError(f"No online context encoder in JEPA training checkpoint {path}")
    else:
        raise ValueError(f"Unsupported JEPA checkpoint format: {path}")
    best = _validation_record(training, metric) if training is not None else None
    epoch = training.get("epoch") if training is not None else None
    if expected is not None and (best is None or epoch != expected or best["epoch"] != expected):
        raise ValueError(f"Cannot verify {path} as best epoch {expected} by validation {metric}")
    sampling = (training["config"].get("sampling", {}) if training is not None
                else load_config(config.get("jepa_config", REPO_ROOT / "encoding/configs/environmental_jepa.yaml"))["sampling"])
    selection = config.get("spatial_selection", "intersects")
    budget = sampling.get("max_tokens", DEFAULT_MAX_TOKENS)
    if selection != sampling.get("spatial_selection", "centers"):
        budget = DEFAULT_MAX_TOKENS
    sampling = {**sampling, "spatial_selection": selection, "max_tokens": config.get("max_tokens", budget)}
    provenance = dict(checkpoint=str(path.resolve()), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      weights_sha256=weights_fingerprint(weights),
                      role="online_context", version=EXPORT_VERSION, epoch=epoch,
                      best_metric=metric, best_metric_value=best["validation"][metric] if best else None,
                      model_config=model_config, registry=registry, sampling=sampling,
                      query_version=QUERY_VERSION, preprocessing_version=preprocessing_fingerprint(registry, model_config),
                      drivers=driver_inventory(DriverRegistry.from_dict(registry)))
    if expected_provenance:
        for key in EMBEDDING_VERSION_KEYS:
            if provenance[key] != expected_provenance.get(key):
                raise ValueError(f"JEPA embedding provenance differs from the MoE checkpoint: {key}")
    encoder = EnvironmentalEncoder(DriverRegistry.from_dict(registry), **model_config)
    if encoder.latent_dim != config.get("latent_dim", 128):
        raise ValueError("JEPA checkpoint latent dimension does not match encoder.latent_dim")
    encoder.load_state_dict(weights, strict=True)
    return EnvironmentalEmbedding(encoder, sampling, provenance)
