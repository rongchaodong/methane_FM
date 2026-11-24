"""Staged optimization, evaluation, and continual growth for methane MoE."""

import csv
import json
import math
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from itertools import islice
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from encoding.training import seed_everything, to_device

from .checkpoint import load_checkpoint, restore_encoder, restore_model, save_checkpoint
from .data import build_loaders
from .encoder import load_environmental_encoder
from .growth import CandidateTrainer, GrowthManager
from .model import (
    TARGET_TYPES,
    ContinualMethaneMoE,
    MoELoss,
    physical_response_loss,
    supervision,
    task_mask,
)
from .splits import audit_summary, loader_partitions, stage_manifests


def _device_and_amp(options):
    requested = options.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    amp = options.get("mixed_precision", False)
    if amp not in (False, True, "fp16", "bf16"):
        raise ValueError("mixed_precision must be false, true, fp16, or bf16")
    amp_dtype = (torch.bfloat16 if amp == "bf16" else torch.float16) if amp and device.type == "cuda" else None
    return device, amp_dtype


def _optimizer(model, criterion, encoder, options):
    optimizer = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}[
        options.get("optimizer", "adamw").lower()
    ]
    learning_rate = options.get("learning_rate", options.get("lr", 1e-3))
    trainable = [parameter for module in (model, criterion) for parameter in module.parameters()
                 if parameter.requires_grad]
    groups = [{"params": trainable, "lr": learning_rate}]
    encoder_parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    if encoder_parameters:
        encoder_lr = options["encoder_lr"]
        if not 0 < encoder_lr < learning_rate:
            raise ValueError("encoder_lr must be positive and lower than the MoE learning rate")
        groups.append({"params": encoder_parameters, "lr": encoder_lr})
    return optimizer(groups, lr=learning_rate, weight_decay=options.get("weight_decay", 1e-4))


def _scheduler(optimizer, options, epochs):
    kind = options.get("scheduler", "none").lower()
    if kind == "none":
        return None
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=options.get("scheduler_factor", 0.5),
            patience=options.get("scheduler_patience", 2),
        )
    raise ValueError("scheduler must be none, cosine, or plateau")


def _make_criterion(options, device, initial=None):
    criterion = MoELoss(
        kind=options.get("loss", "huber"),
        huber_delta=options.get("huber_delta", 1.0),
        source_weights=options.get("source_weights", {}),
        calibration_sources=options.get("calibration_sources", ()),
        balance_weight=options.get("balance_weight", 0.0),
        sign_weight=options.get("sign_weight", 0.0),
        expert_weight=options.get("expert_weight", 0.0),
        min_utilization=options.get("min_utilization", 0.05),
    ).to(device)
    if initial is not None:
        criterion.load_state_dict(initial, strict=False)
    return criterion


class EpochMetrics:
    """Accumulate weighted errors and routing statistics across batches."""

    def __init__(self, source_weights):
        self.source_weights = source_weights
        self.groups = defaultdict(lambda: torch.zeros(7, dtype=torch.float64))
        self.routing_mass = None
        self.routing_entropy = torch.zeros(2, dtype=torch.float64)
        self.routing_weight = torch.zeros(2, dtype=torch.float64)
        self.weighted_objective = 0.0
        self.weighted_auxiliary = defaultdict(float)
        self.objective_steps = 0

    def state_dict(self):
        return {"groups": dict(self.groups), "routing_mass": self.routing_mass, "routing_entropy": self.routing_entropy,
                "routing_weight": self.routing_weight, "weighted_objective": self.weighted_objective,
                "weighted_auxiliary": dict(self.weighted_auxiliary), "objective_steps": self.objective_steps}

    def load_state_dict(self, state):
        self.groups.update(state["groups"])
        self.weighted_auxiliary.update(state["weighted_auxiliary"])
        for name in ("routing_mass", "routing_entropy", "routing_weight", "weighted_objective", "objective_steps"):
            setattr(self, name, state[name])

    @torch.no_grad()
    def update(self, output, batch, criterion, objective, auxiliary=None):
        target, types, weights, valid = supervision(batch, output["latent"].device)
        if not valid.any():
            return
        prediction = criterion.predicted_target(output, batch).double()
        target = target.double().masked_fill(~valid, 0)
        weights = weights.double()
        delta = (prediction - target).masked_fill(~valid, 0)
        losses, _ = criterion.sample_losses(output, batch)
        terms = torch.stack((weights, weights * target, weights * target.square(),
                             weights * delta.square(), weights * delta.abs(),
                             weights * losses.double().masked_fill(~valid, 0), valid.double()))
        self.groups["all", "all"] += terms.sum(dim=1).cpu()
        for task, name in enumerate(TARGET_TYPES):
            self.groups["task", name] += terms[:, types == task].sum(dim=1).cpu()

        # 逐来源保留损失分母，同时报告 TEM 的日排放和月吸收产品。
        for kind, labels in (("source", criterion.sources(batch, len(valid))), ("product", batch.get("product", ()))):
            for label in dict.fromkeys(labels):
                rows = [index for index, value in enumerate(labels) if value == label]
                self.groups[kind, label] += terms[:, rows].sum(dim=1).cpu()

        routing = output["routing"].detach().double()
        route_weights = task_mask(batch, weights.device).to(weights)
        if self.routing_mass is None:
            self.routing_mass = torch.zeros(routing.shape[1:], dtype=torch.float64)
        self.routing_mass += (routing * route_weights[:, :, None]).sum(dim=0).cpu()
        entropy = -(routing * routing.clamp_min(1e-15).log()).sum(-1)
        self.routing_entropy += (entropy * route_weights).sum(dim=0).cpu()
        self.routing_weight += route_weights.sum(dim=0).cpu()

        self.weighted_objective += float(objective.detach())
        for key, value in (auxiliary or {}).items():
            self.weighted_auxiliary[key] += float(value.detach())
        self.objective_steps += 1

    def result(self, num_experts):
        result = {"samples": 0, "by_task": {}, "by_source": {}, "by_product": {}, "expert_count": num_experts}
        result["expert_utilization"] = {}
        result["router_entropy"] = {}
        for (kind, key), values in self.groups.items():
            weight, sum_y, sum_y2, squared, absolute, loss, count = values.tolist()
            if weight == 0:
                continue
            total_variation = max(0.0, sum_y2 - sum_y * sum_y / weight)
            metrics = {
                "samples": int(count), "loss": loss / weight, "rmse": math.sqrt(squared / weight),
                "mae": absolute / weight, "r2": 1.0 - squared / total_variation if total_variation else None,
            }
            if kind == "all":
                result.update(metrics)
            else:
                result[f"by_{kind}"][key] = metrics
        source_results = result["by_source"]
        result["data_loss"] = (
            sum(self.source_weights.get(source, 1.0) * values["loss"]
                for source, values in source_results.items())
            if source_results else None
        )
        result["objective"] = self.weighted_objective / self.objective_steps if self.objective_steps else None
        result.update({name: value / self.objective_steps for name, value in self.weighted_auxiliary.items()})
        for index, task in enumerate(TARGET_TYPES[:2]):
            weight = float(self.routing_weight[index])
            if weight:
                result["expert_utilization"][task] = (self.routing_mass[index] / weight).tolist()
                result["router_entropy"][task] = float(self.routing_entropy[index] / weight)
        return result


class AnchorReservoir:
    """Select a uniform fixed reservoir over valid training latents for each task."""

    def __init__(self, count, seed, dataset_metadata, encoder_metadata):
        self.count = count
        self.generator = torch.Generator().manual_seed(seed)
        self.metadata = {"dataset": dataset_metadata, "encoder": encoder_metadata,
                         "selection": "first_training_epoch_uniform_without_replacement", "seed": seed}
        self.selected = {}

    @torch.no_grad()
    def update(self, latents, batch):
        finite = torch.isfinite(latents).all(dim=-1)
        masks = (task_mask(batch, latents.device).bool() & finite[:, None]).cpu()
        for task in range(2):
            indices = masks[:, task].nonzero().flatten()
            if not len(indices):
                continue
            incoming = torch.rand(len(indices), generator=self.generator)
            rows = indices.tolist()
            metadata = [{"query": str(batch["query"][row]), "source": batch["source"][row],
                         "site_id": batch["site_id"][row]} for row in rows]
            values = latents.detach()[indices.to(latents.device)].float().cpu()
            if task in self.selected:
                priorities, previous, prior_metadata = self.selected[task]
                incoming = torch.cat((priorities, incoming))
                values = torch.cat((previous, values))
                metadata = prior_metadata + metadata
            priorities, kept = incoming.topk(min(self.count, len(incoming)), sorted=True)
            self.selected[task] = (priorities, values[kept], [metadata[row] for row in kept.tolist()])

    def apply(self, model):
        for task, (_, latents, samples) in self.selected.items():
            model.signatures.set_anchors(task, latents, {**self.metadata, "samples": samples})


def _forward(model, encoder, batch, device, amp_dtype, *, response_supervision=False):
    if "latent" in batch:
        if response_supervision:
            raise ValueError("Physical response supervision requires online raw drivers, not cached embeddings")
        latent = encoder.accept_cached(batch["latent"])
    else:
        environment = batch["environment"]
        if response_supervision:
            environment = {**environment, "values": environment["values"].detach().requires_grad_(True)}
            batch = {**batch, "environment": environment}
        attention = (
            torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
            if response_supervision else nullcontext()
        )
        with attention, torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None and not response_supervision):
            latent = encoder(environment)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None and not response_supervision):
        output = model(latent)
    return {key: value.float() if value.is_floating_point() else value for key, value in output.items()}, batch


def _physical_loss(output, batch):
    # 可信参考值按任务提供，维度和单位对应原始 driver token。
    references = batch.get("physical_response")
    if not references:
        raise ValueError("physical_response_weight requires trusted physical_response tensors in training batches")
    if references.keys() - set(TARGET_TYPES):
        raise ValueError("Physical response targets must be emission, uptake, or net")
    values = batch["environment"]["values"]
    mask = batch["environment"]["variable_mask"]
    losses = [
        physical_response_loss(output[task], values, reference["reference"],
                               mask if "weights" not in reference else reference["weights"] * mask)
        for task, reference in references.items()
    ]
    return torch.stack(losses).sum()


def _distill(student, teacher):
    return F.mse_loss(
        torch.stack([student[task] for task in TARGET_TYPES[:2]], dim=-1),
        torch.stack([teacher[task] for task in TARGET_TYPES[:2]], dim=-1),
    )


def _usable(batch):
    return batch is not None and bool(supervision(batch)[-1].any())


def _replay_batches(loader):
    while True:
        found = False
        for batch in loader:
            if _usable(batch):
                found = True
                yield batch
        if not found:
            raise RuntimeError("Training replay contains no valid supervision")


def run_epoch(
    model, encoder, criterion, loader, device, amp_dtype, *,
    optimizer=None, scaler=None, options=None, replay_loader=None,
    teacher=None, anchors=None, candidate=None, max_steps=None, require_samples=True,
    stats=None, on_step=None,
):
    options = options or {}
    training = optimizer is not None
    if candidate:
        candidate.enable_training_mode()
    else:
        model.train(training)
    encoder.train(training and candidate is None)
    criterion.train(training and candidate is None)
    if stats is None:
        stats = EpochMetrics(criterion.source_weights)
    replay_weight = options.get("replay_weight", 0.0)
    distillation_weight = options.get("distillation_weight", 0.0)
    physical_weight = options.get("physical_response_weight", 0.0) if training else 0.0
    replay_iter = _replay_batches(replay_loader) if replay_loader is not None and (replay_weight or distillation_weight) else None
    steps = 0
    for raw_batch in loader:
        if not _usable(raw_batch):
            if on_step is not None:
                raise RuntimeError("A full simulation batch lost valid indexed supervision.")
            continue
        batch = to_device(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output, batch = _forward(model, encoder, batch, device, amp_dtype,
                                     response_supervision=bool(physical_weight))
            terms = criterion(output, batch)
            objective = terms["loss"]
            auxiliary = {name: terms[name] for name in ("balance_loss", "sign_loss", "expert_loss")}
            if replay_iter is not None:
                replay_batch = to_device(next(replay_iter), device)
                replay_output, _ = _forward(model, encoder, replay_batch, device, amp_dtype)
                if replay_weight:
                    auxiliary["replay_loss"] = criterion(replay_output, replay_batch)["data_loss"]
                    objective = objective + replay_weight * auxiliary["replay_loss"]
            else:
                replay_batch, replay_output = batch, output
            if distillation_weight:
                if teacher is None:
                    raise ValueError("Distillation requires a frozen old model")
                with torch.no_grad():
                    old_output, _ = _forward(*teacher, replay_batch, device, amp_dtype)
                auxiliary["distillation_loss"] = _distill(replay_output, old_output)
                objective = objective + distillation_weight * auxiliary["distillation_loss"]
            if physical_weight:
                auxiliary["physical_response_loss"] = _physical_loss(output, batch)
                objective = objective + physical_weight * auxiliary["physical_response_loss"]
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("MoE objective is not finite")
            if training:
                previous_scale = scaler.get_scale()
                scaler.scale(objective).backward()
                scaler.unscale_(optimizer)
                gradient_clip = options.get("gradient_clip", options.get("grad_clip", 1.0))
                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer.param_groups for p in group["params"]],
                        gradient_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    selected = [candidate.index] if candidate else None
                    model.signatures.update(output["latent"], output["routing"], task_mask(batch, device),
                                            expert_indices=selected)
                    if replay_iter is not None:
                        model.signatures.update(replay_output["latent"], replay_output["routing"],
                                                task_mask(replay_batch, device), expert_indices=selected)
                    if anchors is not None:
                        anchors.update(output["latent"], batch)
        stats.update(output, batch, criterion, objective, auxiliary)
        steps += 1
        if on_step is not None:
            on_step(steps, stats)
        if max_steps is not None and steps >= max_steps:
            break
    metrics = stats.result(model.num_experts)
    if require_samples and not metrics["samples"]:
        raise RuntimeError("No valid methane supervision in this split")
    return metrics


def _loader_seed(loader, seed):
    if loader is not None and loader.generator is not None:
        loader.generator.manual_seed(seed)
        if hasattr(loader.batch_sampler, "set_progress"):
            loader.batch_sampler.set_progress(seed)


def _export_metrics(output, history):
    columns = ("stage", "phase", "cohort", "epoch", "batch", "event")

    def flatten(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from flatten(child, f"{path}.{key}" if path else key)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from flatten(child, f"{path}[{index}]")
        elif value is not None:
            yield path, value

    def write_json(stream):
        json.dump(history, stream, indent=2, allow_nan=False)
        stream.write("\n")

    def write_csv(stream):
        writer = csv.writer(stream)
        writer.writerow((*columns, "metric", "value"))
        for row in history:
            base = [row.get(key) for key in columns]
            writer.writerows((*base, metric, value) for key, entry in row.items()
                             if key not in columns
                             for metric, value in flatten(entry, key))

    for name, write in (("metrics.json", write_json), ("metrics.csv", write_csv)):
        temporary = output / f".{name}.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            write(stream)
        temporary.replace(output / name)


def _checkpoint_progress(progress):
    return {**progress, "history": list(progress["history"])}


def _teacher_snapshot(model, encoder, state=None):
    old_model, old_encoder = deepcopy(model).requires_grad_(False), deepcopy(encoder)
    if state is not None:
        old_model.load_state_dict(state["model"])
        old_encoder.load_state_dict(state["encoder"])
    old_model.eval()
    old_encoder.set_trainable(())
    old_encoder.eval()
    return old_model, old_encoder


def _teacher_state(teacher):
    return {key: module.state_dict() for key, module in zip(("model", "encoder"), teacher)} if teacher else None


def _diagnostics(model):
    model.refresh_fingerprints()
    return {"expert_ids": [record["expert_id"] for record in model.expert_records],
            **{key: value.detach().cpu().tolist() for key, value in model.signatures.diagnostics().items()}}


def _fit(
    model, encoder, criterion, loaders, replay, config, options, stage, phase,
    resume, teacher, device, amp_dtype, output, history, provenance,
):
    epochs = int(options.get("epochs", 1))
    if epochs < 1:
        raise ValueError("Each active phase requires at least one epoch")
    optimizer = _optimizer(model, criterion, encoder, options)
    scheduler = _scheduler(optimizer, options, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    progress = {"epoch": 0, "best_loss": None, "stale_epochs": 0, "phase_completed": False}
    if resume:
        progress.update(resume["progress"])
        optimizer.load_state_dict(resume["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
    progress["history"] = history
    progress["data_provenance"] = provenance

    train, validation = loaders
    best = output / (f"best-{phase}.pt" if phase else "best.pt")
    replay_train = replay[0] if replay is not None else None
    seed = options["training_seed"]
    patience = int(options.get("patience", 0))
    streaming = train.dataset.streaming
    checkpoint_interval = int(options.get("checkpoint_interval_steps", 10000))
    if streaming and checkpoint_interval < 1:
        raise ValueError("checkpoint_interval_steps must be positive for full simulation.")

    def persist_progress():
        save_checkpoint(output / "last.pt", model, encoder, config, stage=stage, phase=phase,
                        criterion=criterion, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        progress=_checkpoint_progress(progress), teacher=_teacher_state(teacher))

    def scan(loader, split, epoch, *, anchors=None, **kwargs):
        saved = progress.get("data_scan")
        sampler = loader.batch_sampler
        offset = 0
        metrics = EpochMetrics(criterion.source_weights)
        if saved is not None and saved["split"] == split:
            offset = saved["batches"]
            metrics.load_state_dict(saved["metrics"])
            torch.set_rng_state(saved["torch_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(saved["cuda_rng"], device)
            if anchors is not None:
                anchors.selected = saved["anchor_samples"]
                anchors.generator.set_state(saved["anchor_rng"])
        sampler.set_progress(seed + epoch, offset)

        def checkpoint_step(steps, stats):
            batches = offset + steps
            if batches % checkpoint_interval:
                return
            progress["data_scan"] = {"split": split, "batches": batches, "metrics": stats.state_dict(),
                                     "torch_rng": torch.get_rng_state()}
            if device.type == "cuda":
                progress["data_scan"]["cuda_rng"] = torch.cuda.get_rng_state(device)
            if anchors is not None:
                progress["data_scan"].update(anchor_samples=anchors.selected, anchor_rng=anchors.generator.get_state())
            persist_progress()
            print(json.dumps({"stage": stage, "event": "data_progress", "epoch": epoch + 1,
                              "split": split, "batches": batches, "total_batches": sampler.total_batches,
                              "checkpoint": str(output / "last.pt")}), flush=True)

        return run_epoch(model, encoder, criterion, loader, device, amp_dtype, stats=metrics,
                         on_step=checkpoint_step, anchors=anchors, **kwargs)

    for epoch in range(int(progress["epoch"]), epochs):
        if progress["phase_completed"]:
            break
        seed_everything(seed + epoch)
        _loader_seed(train, seed + epoch)
        _loader_seed(replay_train, seed + epoch)
        count = min(int(options.get("anchor_count", 0)), model.signatures.anchors.shape[1])
        anchors = (
            AnchorReservoir(count, seed, getattr(train.dataset, "provenance", {}), encoder.provenance)
            if count > 0 and epoch == 0 and stage == "pretrain" else None
        )
        fit_options = dict(optimizer=optimizer, scaler=scaler, options=options,
                           replay_loader=replay_train, teacher=teacher, anchors=anchors)
        if streaming:
            if "completed_train_pass" not in progress:
                progress["completed_train_pass"] = scan(train, "train", epoch, **fit_options)
                if anchors is not None:
                    anchors.apply(model)
                progress.pop("data_scan", None)
                persist_progress()
            train_metrics = progress["completed_train_pass"]
            validation_metrics = scan(validation, "validation", epoch)
            progress.pop("completed_train_pass")
            progress.pop("data_scan", None)
        else:
            train_metrics = run_epoch(model, encoder, criterion, train, device, amp_dtype, **fit_options)
            if anchors is not None:
                anchors.apply(model)
            validation_metrics = run_epoch(model, encoder, criterion, validation, device, amp_dtype)
        score = validation_metrics["data_loss"]
        progress["epoch"] = epoch + 1
        improved = progress["best_loss"] is None or score < progress["best_loss"] - options.get("min_delta", 0.0)
        if improved:
            progress["best_loss"], progress["stale_epochs"] = score, 0
        else:
            progress["stale_epochs"] += 1
        progress["phase_completed"] = epoch + 1 == epochs or (patience > 0 and progress["stale_epochs"] >= patience)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()
        record = {"event": "epoch", "stage": stage, "phase": phase, "epoch": epoch + 1,
                  "learning_rates": [group["lr"] for group in optimizer.param_groups],
                  "train": train_metrics, "validation": validation_metrics, "signatures": _diagnostics(model)}
        history.append(record)
        kwargs = dict(stage=stage, phase=phase, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                      scaler=scaler, progress=_checkpoint_progress(progress), teacher=_teacher_state(teacher))
        if improved:
            save_checkpoint(best, model, encoder, config, **kwargs)
            if phase:
                save_checkpoint(output / "best.pt", model, encoder, config, **kwargs)
        save_checkpoint(output / "last.pt", model, encoder, config, **kwargs)
        _export_metrics(output, history)
        print(json.dumps({"stage": stage, "phase": phase, "epoch": epoch + 1,
                          "validation_data_loss": score, "experts": model.num_experts,
                          "checkpoint": str(output / "last.pt")}), flush=True)
    return best, progress


def _matching_window(pairs):
    predictions, batches = zip(*pairs)
    output = {key: torch.cat([prediction[key] for prediction in predictions]) for key in predictions[0]}
    batch = {key: torch.cat([part[key] for part in batches])
             for key in ("target", "target_type", "sample_weight", "quality_mask")}
    batch["source"] = [source for part in batches for source in part["source"]]
    return output, batch


def _continual(
    model, encoder, criterion, loaders, replay, config, options,
    resume, previous, device, amp_dtype, output, history, provenance, cohort,
):
    candidate_training = options.get("candidate_training", True)
    if candidate_training and not all(options.get(key, True)
                                     for key in ("function_preserving_split", "validate_before_accept")):
        raise ValueError("Continual growth requires function-preserving splits and held-out acceptance")
    growth_keys = {"max_experts", "min_samples", "min_validation_samples", "min_replay_samples",
                   "persistence", "latent_weight", "response_weight", "distance_threshold", "loss_threshold",
                   "gain_threshold", "forgetting_weight", "parameter_weight", "max_forgetting"}
    growth = GrowthManager(**({key: value for key, value in options.items() if key in growth_keys}
                              | options.get("growth", {})))
    encoder.eval()
    criterion.requires_grad_(False).eval()
    train, validation = loaders
    replay_train, replay_validation = replay if replay is not None else (None, None)
    if candidate_training and options.get("replay_weight", 1.0) and replay_train is None:
        raise ValueError("Candidate replay requires historical replay sources")
    growth_optimizer = _optimizer(model, criterion, encoder, options)
    progress = {"epoch": 0, "batch": 0, "monitor": {}}
    inherited = resume or previous
    if inherited:
        growth.load_state_dict(inherited["growth"])
        growth_optimizer.load_state_dict(inherited["optimizer"])
    if resume:
        progress.update(resume["progress"])
    else:
        growth.streak = 0
    progress["history"] = history
    progress["data_provenance"] = provenance

    def persist(path):
        save_checkpoint(path, model, encoder, config, stage="continual", criterion=criterion,
                        optimizer=growth_optimizer, progress=_checkpoint_progress(progress), growth=growth)

    epochs = int(options.get("epochs", 1))
    candidate_epochs = int(options.get("candidate_epochs", 1))
    candidate_steps = options.get("candidate_steps")
    if min(epochs, candidate_epochs) < 1 or (candidate_steps is not None and candidate_steps < 1):
        raise ValueError("Continual/candidate epochs and any candidate_steps must be positive")
    seed = options["training_seed"]
    # 嵌套优化使用单独的 shuffle RNG，恢复扫描可以从相同顺序继续。
    candidate_loader = DataLoader(
        train.dataset, batch_size=train.batch_size, num_workers=train.num_workers,
        collate_fn=train.collate_fn, shuffle=bool(len(train.dataset)), generator=torch.Generator(),
    )
    for epoch in range(progress["epoch"], epochs):
        seed_everything(seed + epoch)
        _loader_seed(train, seed + epoch)
        _loader_seed(replay_train, seed + epoch)
        skip = progress["batch"]
        stats = progress["monitor"] or {"batches": 0, "valid_samples": 0, "ready": 0, "triggered": 0}
        progress["monitor"] = stats
        matching, matching_count = [], 0
        for batch_index, raw_batch in enumerate(islice(iter(train), skip, None), start=skip + 1):
            progress["batch"] = batch_index
            if not _usable(raw_batch):
                continue
            batch = to_device(raw_batch, device)
            model.eval()
            with torch.no_grad():
                prediction, _ = _forward(model, encoder, batch, device, amp_dtype)
            count = int(supervision(batch)[-1].sum())
            stats["batches"] += 1
            stats["valid_samples"] += count
            # 仅缓冲匹配需要的输出与标签，避免保留大型环境 token 批次。
            matching.append((
                {key: prediction[key] for key in ("base", "expert_outputs", "latent", "emission", "uptake")},
                {key: batch[key] for key in ("target", "target_type", "sample_weight", "quality_mask", "source")},
            ))
            matching_count += count
            if matching_count < growth.min_samples:
                continue
            match_output, match_batch = _matching_window(matching)
            matching, matching_count = [], 0
            proposal = growth.observe(model, match_output, match_batch, criterion)
            stats["last_match"] = proposal
            stats["ready"] += int(proposal.get("ready", False))
            stats["triggered"] += int(proposal.get("triggered", False))
            if not (proposal.get("triggered", False) and candidate_training):
                if proposal.get("triggered", False):
                    growth.streak = 0
                continue
            new_old = run_epoch(model, encoder, criterion, validation, device, amp_dtype, require_samples=False)
            replay_old = (run_epoch(model, encoder, criterion, replay_validation, device, amp_dtype, require_samples=False)
                          if replay_validation is not None else {"samples": 0, "data_loss": None})
            enough = (new_old["samples"] >= growth.min_validation_samples
                      and replay_old["samples"] >= growth.min_replay_samples)
            candidate = None
            new_candidate = replay_candidate = {"data_loss": None}
            validation_birth = None
            if enough:
                candidate = CandidateTrainer(model, proposal["parent"], expert_id=proposal["candidate_id"])
                candidate_opts = {**options, "learning_rate": options.get("candidate_learning_rate", 1e-3),
                                  "replay_weight": options.get("replay_weight", 1.0),
                                  "distillation_weight": options.get("distillation_weight", 1.0)}
                candidate_optimizer = _optimizer(candidate.model, criterion, encoder, candidate_opts)
                candidate_scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
                for candidate_epoch in range(candidate_epochs):
                    candidate_seed = seed + epoch + batch_index + candidate_epoch
                    seed_everything(candidate_seed)
                    # 候选使用独立 DataLoader 随机流，外层增长扫描保持原顺序。
                    _loader_seed(candidate_loader, candidate_seed)
                    _loader_seed(replay_train, candidate_seed)
                    run_epoch(candidate.model, encoder, criterion, candidate_loader, device, amp_dtype,
                              optimizer=candidate_optimizer, scaler=candidate_scaler,
                              options=candidate_opts, replay_loader=replay_train, teacher=(model, encoder),
                              candidate=candidate, max_steps=candidate_steps)
                new_candidate = run_epoch(candidate.model, encoder, criterion, validation, device, amp_dtype)
                if replay_validation is not None:
                    replay_candidate = run_epoch(candidate.model, encoder, criterion, replay_validation, device, amp_dtype)
                validation_birth = output / f"birth-{proposal['candidate_id']}.pt"
            updated, event = growth.decide(
                model, candidate, new_old_loss=new_old["data_loss"],
                new_candidate_loss=new_candidate["data_loss"], replay_old_loss=replay_old["data_loss"],
                replay_candidate_loss=replay_candidate["data_loss"], new_count=new_old["samples"],
                replay_count=replay_old["samples"], birth_checkpoint=str(validation_birth) if validation_birth else None,
                proposal=proposal,
            )
            if updated is not model:
                old_parameters = dict(model.named_parameters())
                next_optimizer = _optimizer(updated, criterion, encoder, options)
                for name, parameter in updated.named_parameters():
                    if parameter in candidate_optimizer.state:
                        next_optimizer.state[parameter] = candidate_optimizer.state[parameter]
                    elif name in old_parameters and old_parameters[name] in growth_optimizer.state:
                        next_optimizer.state[parameter] = growth_optimizer.state[old_parameters[name]]
                model, growth_optimizer = updated, next_optimizer
            record = {"event": "growth", "stage": "continual", "cohort": cohort, "epoch": epoch + 1,
                      "batch": batch_index, "proposal": proposal, "growth": event,
                      "validation": {"old": new_old, "candidate": new_candidate},
                      "replay_validation": {"old": replay_old, "candidate": replay_candidate},
                      "expert_count": model.num_experts, "signatures": _diagnostics(model)}
            history.append(record)
            if event["status"] == "accepted":
                persist(validation_birth)
            persist(output / "last.pt")
            _export_metrics(output, history)
            print(json.dumps({"stage": "continual", "epoch": epoch + 1, "batch": batch_index,
                              "status": event["status"], "experts": model.num_experts}), flush=True)

        if matching:
            stats["incomplete_match"] = growth.observe(model, *_matching_window(matching), criterion)
        if not stats["valid_samples"]:
            stats["status"] = "pending_insufficient_data"
        validation_metrics = run_epoch(model, encoder, criterion, validation, device, amp_dtype, require_samples=False)
        progress.update(epoch=epoch + 1, batch=0, monitor={})
        history.append({"event": "epoch", "stage": "continual", "epoch": epoch + 1,
                        "cohort": cohort, "monitor": stats, "validation": validation_metrics,
                        "expert_count": model.num_experts,
                        "signatures": _diagnostics(model)})
        persist(output / "last.pt")
        _export_metrics(output, history)
    return {"checkpoint": str(output / "last.pt"), "experts": model.num_experts, "history": history}


def _stage_loaders(data, partitions, options, encoder, gatherer):
    return build_loaders(data, partitions, batch_size=options.get("batch_size", 16),
                         num_workers=options.get("num_workers", 0), training_seed=options["training_seed"],
                         gatherer=gatherer, encoder_metadata=encoder.provenance)


def _phase_options(stage, options, encoder_options):
    if stage == "pretrain":
        patterns = () if encoder_options.get("freeze", True) else encoder_options.get("unfreeze_patterns")
        if patterns is None or (not patterns and not encoder_options.get("freeze", True)):
            raise ValueError("An unfrozen encoder requires explicit unfreeze_patterns")
        return [(None, {**options, "unfreeze_patterns": patterns})]
    entries = options.get("phases", [{"name": "A"}])
    if isinstance(entries, dict):
        entries = [{"name": name, **phase} for name, phase in entries.items()]
    enabled = [entry for entry in entries if entry.get("enabled", True) and entry.get("epochs", 1)]
    names = [entry["name"].upper() for entry in enabled]
    if names not in (["A"], ["A", "B"]):
        raise ValueError("Fine-tuning requires phase A and may then enable phase B")
    phases = []
    for name, entry in zip(names, enabled):
        phase = {**options, **entry}
        if name == "A":
            phase.update(unfreeze_patterns=(), replay_weight=0.0, distillation_weight=0.0)
        else:
            if not phase.get("unfreeze_patterns") or "encoder_lr" not in phase:
                raise ValueError("Phase B requires explicit unfreeze_patterns and encoder_lr")
            if options.get("use_simulation_replay", True):
                phase.setdefault("replay_weight", 1.0)
            else:
                phase["replay_weight"] = 0.0
            phase.setdefault("distillation_weight", 1.0)
        phases.append((name, phase))
    return phases


def _load_best(path, source_checkpoint, provenance):
    if not path.exists() and source_checkpoint:
        path = Path(source_checkpoint).parent / path.name
    if not path.exists():
        raise FileNotFoundError(f"Missing best validation checkpoint for phase transition or resume: {path}")
    payload = load_checkpoint(path)
    if payload["progress"].get("data_provenance") != provenance:
        raise ValueError(f"Best validation checkpoint uses different split provenance: {path}")
    return payload


def _data_options(base, overrides):
    adapters, overrides_by_adapter = base.get("adapters", {}), overrides.get("adapters", {})
    return {**base, **overrides, "adapters": {
        key: {**adapters.get(key, {}), **overrides_by_adapter.get(key, {})}
        for key in dict.fromkeys([*adapters, *overrides_by_adapter])
    }}


def run_training(config, stage, checkpoint=None, resume=None, output_dir=None, *, prepare_data=False):
    """Run an explicitly requested stage on configured real data."""
    if stage not in ("pretrain", "finetune", "continual"):
        raise ValueError(f"Unknown MoE training stage: {stage}")
    options = {**config.get("training", {}).get("common", {}), **config["training"][stage]}
    options.setdefault("training_seed", options.get("seed", 42))
    source_checkpoint = resume or checkpoint
    payload = load_checkpoint(source_checkpoint) if source_checkpoint else None
    prior_stages = {"pretrain": (), "finetune": ("pretrain",), "continual": ("finetune", "continual")}[stage]
    if resume:
        if payload["stage"] != stage:
            raise ValueError(f"Resume checkpoint must be from the {stage} stage")
        if any(payload["config"].get(key) != config.get(key)
               for key in ("encoder", "data", "moe", "training")):
            raise ValueError("Resume requires the same encoder, data, model, and training configuration")
    elif ((payload is None and prior_stages and not (prepare_data and stage == "finetune"))
          or (payload is not None and payload["stage"] not in prior_stages)):
        raise ValueError(f"{stage} expects prior checkpoint stages {prior_stages}")
    encoder_options = config.get("encoder", {})
    data = _data_options(config["data"], options.get("data", {}))
    past_data = payload["progress"].get("data_provenance", {}) if payload else None
    previous_fluxnet = (past_data or {}).get("train", {}).get("products", {}).get("fluxnet", {}).get("target_type")
    data["fluxnet_target_type"] = options.get(
        "fluxnet_target_type", data.get("fluxnet_target_type", previous_fluxnet or "emission")
    )
    phases = _phase_options(stage, options, encoder_options) if stage != "continual" else []
    needs_replay = stage == "continual" or (
        stage == "finetune" and options.get("use_simulation_replay", True)
        and any(phase.get("replay_weight") or phase.get("distillation_weight") for name, phase in phases if name == "B")
    )
    replay_sources = options.get("replay_sources", ["TEM", "MeMo"]) if needs_replay and payload else []
    replay_overrides = options.get("replay_data", {})
    if replay_sources and replay_overrides.keys() - {"embedding_mode", "cache_path", "gather_mode"}:
        raise ValueError("replay_data can only change the embedding mode, cache path, or gather mode.")
    active, split_history, replay_names, intersections = stage_manifests(
        data, options["sources"], stage, past_data, resume=bool(resume), replay_sources=replay_sources,
    )
    if prepare_data:
        return audit_summary(active, intersections, int(options.get("batch_size", 16)))

    device, amp_dtype = _device_and_amp(options)
    if "cpu_threads" in options:
        torch.set_num_threads(options["cpu_threads"])
    seed_everything(options["training_seed"])
    expected_provenance = payload["encoder_provenance"] if payload else None
    encoder = load_environmental_encoder(encoder_options, expected_provenance=expected_provenance).to(device)
    if payload:
        restore_encoder(encoder, payload)
        model = restore_model(payload).to(device)
    else:
        model_options = dict(config.get("moe", {}))
        model_options.setdefault("latent_dim", encoder_options.get("latent_dim", 128))
        model = ContinualMethaneMoE(**model_options).to(device)
    if model.latent_dim != encoder_options.get("latent_dim", 128):
        raise ValueError("MoE and encoder latent dimensions differ")
    encoder.set_trainable(())
    cached = data.get("embedding_mode", "online") == "cached"
    if cached and encoder.adapted:
        raise ValueError("Cached latents require the original, untuned JEPA encoder")

    loaders, replay = (), None
    if any(phase.get("unfreeze_patterns") and cached for _, phase in phases):
        raise ValueError("Cached latents cannot be used in an encoder fine-tuning phase")
    replay_data = {**config["data"], **replay_overrides}
    replay_cached = replay_data.get("embedding_mode", data.get("embedding_mode", "online")) == "cached"
    if replay_sources and replay_cached and (
        encoder.adapted or any(phase.get("unfreeze_patterns") for _, phase in phases)
    ):
        raise ValueError("Replay for an adapted or trainable encoder requires online environmental inputs")

    gatherer = encoder.make_gatherer() if not cached or (replay_sources and not replay_cached) else None
    output = Path(output_dir or config["training"][stage].get(
        "output_dir", Path(options.get("output_dir", "moe/runs")) / stage
    ))
    output.mkdir(parents=True, exist_ok=True)
    previous_continual = payload if payload and stage == payload["stage"] == "continual" and not resume else None
    history = list(payload["progress"].get("history", [])) if resume or previous_continual else []
    inherited_criterion = payload.get("criterion") if payload else None
    calibration = {key.split(".", 1)[1] for key in (inherited_criterion or {}) if key.startswith("calibration.")}
    calibration.update(options.get("calibration_sources", ()))
    options["calibration_sources"] = sorted(calibration)
    try:
        loaders = _stage_loaders(data, loader_partitions([active]), options, encoder, None if cached else gatherer)
        if replay_names:
            replay = _stage_loaders(replay_data, loader_partitions(split_history, replay_names), options,
                                    encoder, None if replay_cached else gatherer)
        provenance = {"active_manifest": active[0], "historical_manifests": [ref for ref, _ in split_history],
                      "intersections": intersections, "train": loaders[0].dataset.provenance,
                      "validation": loaders[1].dataset.provenance}
        if replay:
            provenance.update(replay_train=replay[0].dataset.provenance, replay_validation=replay[1].dataset.provenance)
        if resume and provenance != past_data:
            raise ValueError("Training data provenance changed since the checkpoint was saved")
        if stage == "continual":
            criterion = _make_criterion(options, device, inherited_criterion)
            cohort = sum(manifest["settings"]["kind"] != "simulation" for _, manifest in split_history)
            return _continual(model, encoder, criterion, loaders, replay, config, options,
                              payload if resume else None, previous_continual, device, amp_dtype,
                              output, history, provenance, cohort)
        for loader in loaders:
            if not all(product["valid_labels"] for product in loader.dataset.provenance["products"].values()):
                split = loader.dataset.provenance["split"]
                raise ValueError(f"Every active product requires valid labels in the {split} partition.")
        resume_phase = payload["phase"] if resume else phases[0][0]
        available_phases = [phase for phase, _ in phases]
        if resume_phase not in available_phases:
            raise ValueError(f"Checkpoint phase {resume_phase} is disabled or absent")
        start = available_phases.index(resume_phase)
        for position, (phase, phase_options) in enumerate(phases[start:], start=start):
            current_resume = payload if resume and position == start else None
            phase_options["calibration_sources"] = options["calibration_sources"]
            criterion = _make_criterion(phase_options, device, inherited_criterion)
            encoder.set_trainable(phase_options.get("unfreeze_patterns", ()))
            if current_resume and phase_options.get("distillation_weight", 0) and current_resume["teacher"] is None:
                raise ValueError("The Phase B resume checkpoint is missing its frozen distillation teacher")
            teacher = (
                _teacher_snapshot(model, encoder, current_resume["teacher"] if current_resume else None)
                if phase_options.get("distillation_weight", 0) else None
            )
            if phase == "B" and current_resume is None:
                # 在新的 JEPA 表示坐标中重估环境统计，保留历史响应锚点。
                model.signatures.reset_statistics()
            best, _ = _fit(model, encoder, criterion, loaders, replay, config, phase_options, stage,
                           phase, current_resume, teacher, device, amp_dtype, output, history, provenance)
            if position < len(phases) - 1:
                previous_best = _load_best(best, source_checkpoint, provenance)
                model = restore_model(previous_best).to(device)
                restore_encoder(encoder, previous_best)
                inherited_criterion = previous_best["criterion"]
        return {"checkpoint": str(output / "last.pt"), "best_checkpoint": str(best), "history": history}
    finally:
        for loader in (*loaders, *(replay or ())):
            loader.dataset.close()
        if gatherer is not None:
            for source in gatherer.sources.values():
                source.close()
