"""Training, chronological validation, and checkpointing for Environmental JEPA."""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import EnvironmentalCollator, EnvironmentalJEPADataset, build_queries, future_query
from .gather import MultiResolutionGatherer
from .jepa import EnvironmentalJEPA
from .model import EnvironmentalEncoder
from .sources import build_data_sources
from .specs import DriverRegistry


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream) if Path(path).suffix.lower() == ".json" else yaml.safe_load(stream)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    return value


def split_queries(queries, validation_fraction, future_targets):
    """Keep dates together and purge training targets extending into validation."""
    queries = sorted(queries, key=lambda query: query.start_time)
    if not validation_fraction:
        return queries, []
    dates = sorted({query.start_time for query in queries})
    if not 0 < validation_fraction < 1 or len(dates) < 2:
        raise ValueError("Validation requires at least two dates and a fraction in (0, 1)")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * (1 - validation_fraction))))
    validation_start = dates[split_index]
    train = [query for query in queries
             if max(future_query(query, target).end_time for target in future_targets) <= validation_start]
    validation = [query for query in queries if query.start_time >= validation_start]
    if not train:
        raise ValueError("No training queries remain after purging future targets; extend the date range")
    return train, validation


def _epoch(model, loader, device, optimizer=None, scaler=None,
           mixed_precision=False, gradient_clip=1.0, log=None, epoch=None,
           log_every_steps=0):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "jepa_loss": 0.0, "variance_loss": 0.0}
    samples = 0
    for step, batch in enumerate(loader, 1):
        if batch is None:
            continue
        batch = to_device(batch, device)
        size = batch["context"]["query"].shape[0]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=mixed_precision):
                output = model(batch)
            if not torch.isfinite(output["loss"]):
                raise FloatingPointError("JEPA loss is not finite; check data and normalization")
            if training:
                scaler.scale(output["loss"]).backward()
                scaler.unscale_(optimizer)
                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # AMP may skip an optimizer step after overflow.
                if scaler.get_scale() >= previous_scale:
                    model.update_target()
        for key in totals:
            totals[key] += output[key].detach().item() * size
        samples += size
        if training and log_every_steps and step % log_every_steps == 0:
            log({"event": "train_step", "epoch": epoch, "step": step, "steps": len(loader),
                 "samples": samples, "loss": output["loss"].detach().item(),
                 "mean_loss": totals["loss"] / samples,
                 "learning_rate": optimizer.param_groups[0]["lr"]})
    if not samples:
        raise RuntimeError("No valid samples in this epoch; check data coverage and missing values")
    return {**{key: value / samples for key, value in totals.items()}, "samples": samples}


def run_training(config, output_dir=None, resume=None):
    """Train on configured files and keep one checkpoint plus an exported encoder."""
    options = config.get("training", {})
    seed = options.get("seed", 42)
    seed_everything(seed)
    if "cpu_threads" in options:
        torch.set_num_threads(options["cpu_threads"])
    requested_device = options.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else torch.device(requested_device)
    mixed_precision = options.get("mixed_precision", False) and device.type == "cuda"
    registry = DriverRegistry.from_dict(config["registry"])
    sources = build_data_sources(registry)
    gatherer = MultiResolutionGatherer(registry, sources, config.get("sampling", {}))
    dataset_options = config["dataset"]
    future_targets = dataset_options["future_targets"]
    train_queries, validation_queries = split_queries(
        build_queries(config["queries"]), dataset_options.get("validation_fraction", 0.2), future_targets
    )
    loader_options = dict(batch_size=options.get("batch_size", 8),
                          num_workers=options.get("num_workers", 0),
                          collate_fn=EnvironmentalCollator(registry))
    generator = torch.Generator()
    train_loader = DataLoader(
        EnvironmentalJEPADataset(gatherer, train_queries, future_targets),
        shuffle=True, generator=generator, **loader_options,
    )
    validation_loader = DataLoader(
        EnvironmentalJEPADataset(gatherer, validation_queries, future_targets),
        shuffle=False, **loader_options,
    ) if validation_queries else None
    model = EnvironmentalJEPA(EnvironmentalEncoder(registry, **config.get("model", {})),
                              **config.get("jepa", {})).to(device)
    optimizer_class = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[
        options.get("optimizer", "adamw").lower()]
    optimizer = optimizer_class(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=options.get("learning_rate", 1e-3), weight_decay=options.get("weight_decay", 1e-4),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    start_epoch, history = 0, []
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        saved = checkpoint["config"]
        if any(saved.get(key) != config.get(key)
               for key in ("registry", "sampling", "queries", "dataset", "model", "jepa")):
            raise ValueError("Resume requires the same data, targets, and model configuration")
        saved_optimizer = saved.get("training", {}).get("optimizer", "adamw").lower()
        if saved_optimizer != options.get("optimizer", "adamw").lower() or bool(checkpoint["scaler"]) != mixed_precision:
            raise ValueError("Resume requires the same optimizer and active mixed precision mode")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, history = checkpoint["epoch"], checkpoint["history"]
    destination = Path(output_dir or options.get("output_dir", "encoding/runs/environmental_jepa"))
    if output_dir is None and not destination.is_absolute():
        destination = Path(__file__).resolve().parent.parent / destination
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path, encoder_path = destination / "last.pt", destination / "encoder.pt"
    log_path = destination / "train.log"

    def log(record):
        line = json.dumps({"time": datetime.now(timezone.utc).isoformat(), **record})
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    try:
        log({"event": "startup", "device": str(device),
             "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
             "resume": str(resume) if resume else None, "completed_epochs": start_epoch,
             "epochs": options.get("epochs", 1), "batch_size": loader_options["batch_size"],
             "optimizer": options.get("optimizer", "adamw"),
             "learning_rate": optimizer.param_groups[0]["lr"], "seed": seed,
             "mixed_precision": mixed_precision,
             "train_queries": len(train_queries), "validation_queries": len(validation_queries),
             "model": model.online_encoder.to_config(), "sampling": config.get("sampling", {}),
             "ema_momentum": model.ema_momentum, "variance_weight": model.variance_weight,
             "future_targets": future_targets, "log_every_steps": options.get("log_every_steps", 0)})
        for epoch in range(start_epoch, options.get("epochs", 1)):
            # Epoch-local seeds make epoch-boundary resume reproducible without RNG snapshots.
            seed_everything(seed + epoch)
            generator.manual_seed(seed + epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_metrics = _epoch(model, train_loader, device, optimizer, scaler,
                                   mixed_precision, options.get("gradient_clip", 1.0),
                                   log, epoch + 1, options.get("log_every_steps", 0))
            validation_metrics = _epoch(model, validation_loader, device,
                                        mixed_precision=mixed_precision) if validation_loader else None
            record = {"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics}
            history.append(record)
            log_record = {"event": "epoch", **record}
            if device.type == "cuda":
                log_record.update(cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                                  cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
            log(log_record)
            torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "config": config, "history": history}, checkpoint_path)
        model.save_encoder(encoder_path)
    finally:
        for source in sources.values():
            source.close()
    return {"history": history, "encoder_path": str(encoder_path),
            "log_path": str(log_path),
            "checkpoint_path": str(checkpoint_path if checkpoint_path.exists() else resume),
            "train_queries": len(train_queries), "validation_queries": len(validation_queries)}
