import hashlib
import json
from fnmatch import fnmatchcase
from pathlib import Path

import torch
from torch import nn

from encoding import DriverRegistry, EnvironmentalEncoder, MultiResolutionGatherer, build_data_sources
from encoding.specs import DEFAULT_MAX_TOKENS
from encoding.training import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_VERSION = "environmental-jepa-online-v1"
QUERY_VERSION = "encoding-query-spec-v2"
ONLINE_PREFIX = "online_encoder."


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
        self.adapted = False
        self._patterns = ()
        self.set_trainable(())

    @property
    def trainable(self):
        return bool(self._patterns)

    def set_trainable(self, patterns=()):
        if isinstance(patterns, str):
            patterns = (patterns,)
        patterns = tuple(patterns)
        names = [name for name, _ in self.encoder.named_parameters()]
        missing = [pattern for pattern in patterns if not any(fnmatchcase(name, pattern) for name in names)]
        if missing:
            raise ValueError(f"JEPA unfreeze patterns match no parameter: {missing}")
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad_(any(fnmatchcase(name, pattern) for pattern in patterns))
        self._patterns = patterns
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # 局部解冻仅改变参数梯度。保持 JEPA dropout 关闭，使固定 anchors 可比较。
        self.encoder.eval()
        return self

    def make_gatherer(self):
        return MultiResolutionGatherer(self.registry, build_data_sources(self.registry), self.sampling)

    def forward(self, batch):
        # Encoder 全冻结时仍允许对物理驱动求导。
        if self.trainable or batch["values"].requires_grad:
            return self.encoder(batch)
        with torch.no_grad():
            return self.encoder(batch)

    def accept_cached(self, latent):
        if self.trainable or self.adapted:
            raise ValueError("Cached embeddings require the frozen encoder from the original JEPA checkpoint")
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Cached embeddings must have shape [B, {self.latent_dim}]")
        return latent.detach()


def load_environmental_encoder(config, expected_provenance=None):
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
    schema = json.dumps({"registry": registry, "sampling": sampling}, sort_keys=True).encode()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = dict(checkpoint=str(path.resolve()), checkpoint_sha256=digest,
                      role="online_context", version=EXPORT_VERSION, epoch=epoch,
                      best_metric=metric, best_metric_value=best["validation"][metric] if best else None,
                      model_config=model_config, registry=registry, sampling=sampling,
                      query_version=QUERY_VERSION,
                      preprocessing_version=config.get("preprocessing_version") or hashlib.sha256(schema).hexdigest())
    if expected_provenance and any(provenance[key] != expected_provenance.get(key)
                                   for key in ("checkpoint_sha256", "role", "version", "query_version",
                                               "preprocessing_version", "sampling")):
        raise ValueError("JEPA checkpoint or preprocessing does not match the MoE checkpoint")
    encoder = EnvironmentalEncoder(DriverRegistry.from_dict(registry), **model_config)
    if encoder.latent_dim != config.get("latent_dim", 128):
        raise ValueError("JEPA checkpoint latent dimension does not match encoder.latent_dim")
    encoder.load_state_dict(weights, strict=True)
    result = EnvironmentalEmbedding(encoder, sampling, provenance)
    if not config.get("freeze", True):
        result.set_trainable(config["unfreeze_patterns"])
    return result
