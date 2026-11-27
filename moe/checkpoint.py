import os
import random
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from .encoder import EMBEDDING_VERSION_KEYS, weights_fingerprint

FORMAT_VERSION = 2


def rng_state(device=None):
    kind, state, index, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": (kind, state.tolist(), index, has_gauss, cached),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device is not None and device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    kind, values, index, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, np.array(values, dtype=np.uint32), index, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def save_checkpoint(path, model, encoder, effective_config, *, stage, criterion=None,
                    optimizer=None, scheduler=None, scaler=None, progress=None, growth=None, teacher=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "stage": stage,
        "effective_config": effective_config,
        "model_config": model.metadata(),
        "model": model.state_dict(),
        "encoder_provenance": encoder.provenance,
        "encoder": encoder.state_dict(),
        "criterion": criterion.state_dict() if criterion is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "progress": progress or {},
        "growth": growth.state_dict() if hasattr(growth, "state_dict") else growth,
        "teacher": teacher,
        "rng": rng_state(next(model.parameters()).device),
    }
    # Publish atomically on the same filesystem so an interrupted write leaves the previous checkpoint intact.
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
    return path


def load_checkpoint(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Incompatible MoE checkpoint: {path}. Start a new MoE run with the existing JEPA checkpoint "
                         "and Task 1 split manifests. This format requires two normalized tasks and independent routers.")
    return payload


def restore_model(payload):
    from .model import ContinualMethaneMoE

    model = ContinualMethaneMoE.from_metadata(payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    return model


def restore_encoder(encoder, payload):
    saved = payload["encoder_provenance"]
    if any(encoder.provenance.get(key) != saved.get(key)
           for key in EMBEDDING_VERSION_KEYS):
        raise ValueError("JEPA encoder provenance does not match the MoE checkpoint")
    if weights_fingerprint(encoder.state_dict()) != weights_fingerprint(payload["encoder"]):
        raise ValueError("MoE checkpoint encoder state differs from the frozen JEPA parameters or buffers")
    return encoder.eval()
