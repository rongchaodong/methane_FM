import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


FORMAT_VERSION = 1


def save_checkpoint(path, model, encoder, config, *, stage, phase=None, criterion=None,
                    optimizer=None, scheduler=None, scaler=None, progress=None, growth=None, teacher=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "stage": stage,
        "phase": phase,
        "config": config,
        "model_config": model.metadata(),
        "model": model.state_dict(),
        "encoder_provenance": encoder.provenance,
        "encoder_adapted": encoder.adapted or encoder.trainable,
        "encoder": encoder.state_dict(),
        "criterion": criterion.state_dict() if criterion is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "progress": progress or {},
        "growth": growth.state_dict() if hasattr(growth, "state_dict") else growth,
        "teacher": teacher,
    }
    # 同一目录原子替换，保留写入中断前的完整 checkpoint。
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
        raise ValueError(f"Unsupported MoE checkpoint format: {path}")
    return payload


def restore_model(payload):
    from .model import ContinualMethaneMoE

    model = ContinualMethaneMoE.from_metadata(payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    return model


def restore_encoder(encoder, payload):
    saved = payload["encoder_provenance"]
    if any(encoder.provenance.get(key) != saved.get(key)
           for key in ("checkpoint_sha256", "query_version", "preprocessing_version", "sampling")):
        raise ValueError("JEPA encoder provenance does not match the MoE checkpoint")
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.adapted = payload["encoder_adapted"]
    return encoder
