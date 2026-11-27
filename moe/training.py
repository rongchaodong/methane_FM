"""Staged optimization, evaluation, and continual growth for methane MoE."""

import csv
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import nullcontext
from copy import deepcopy
from inspect import signature
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from encoding.training import seed_everything, to_device

from .cache import STORAGE_DTYPES, build_embedding_cache
from .checkpoint import load_checkpoint, restore_encoder, restore_model, restore_rng, save_checkpoint
from .data import ADAPTERS, FLUX_UNIT, GRID_PRODUCTS, _site_key, build_loaders, product_names
from .encoder import ENCODER_OPTIONS, load_environmental_encoder
from .growth import CandidateTrainer, GrowthManager
from .model import (
    TARGET_TYPES,
    ContinualMethaneMoE,
    MoELoss,
    physical_response_loss,
    supervision,
    task_mask,
)
from .splits import audit_summary, loader_partitions, stage_manifests, training_label_statistics


def _device_and_amp(options):
    requested = options["device"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError(f"Requested training device is unavailable: {device}")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    amp = options["mixed_precision"]
    if amp not in (False, True, "fp32", "fp16", "bf16"):
        raise ValueError("mixed_precision must be false, true, fp32, fp16, or bf16")
    amp_dtype = {True: torch.float16, "fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    return device, amp_dtype


def _optimizer(model, criterion, options):
    optimizer = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}[
        options.get("optimizer", "adamw").lower()
    ]
    learning_rate = options["learning_rate"]
    trainable = [parameter for module in (model, criterion) for parameter in module.parameters()
                 if parameter.requires_grad]
    return optimizer(trainable, lr=learning_rate, weight_decay=options.get("weight_decay", 1e-4))


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


def _make_criterion(options, model, device, initial=None):
    criterion = MoELoss(
        kind=options.get("loss", "mse"),
        huber_delta=options.get("huber_delta", 1.0),
        source_weights=options.get("source_weights", {}),
        task_weights=options.get("task_weights"),
        calibration_sources=options.get("calibration_sources", ()),
        balance_weight=options.get("balance_weight", 0.0),
        sign_weight=options.get("sign_weight", 0.0),
        expert_weight=options.get("expert_weight", 0.0),
        min_utilization=options.get("min_utilization", 0.05),
        label_normalization=model.label_normalization,
    ).to(device)
    if initial is not None:
        # A new source starts with identity calibration; inherited parameters must be loaded by name.
        current = criterion.state_dict()
        if initial.keys() - current.keys():
            raise ValueError("Include the inherited calibration_sources to retain all checkpoint parameters")
        criterion.load_state_dict(current | initial, strict=True)
    return criterion


class EpochMetrics:
    """Streaming metrics in physical and normalized units, with optional site pooling."""

    def __init__(self, criterion, *, sites=False):
        self.criterion = criterion
        self.site_metrics = sites
        self.groups = defaultdict(lambda: torch.zeros(8, dtype=torch.float64))
        self.sites = defaultdict(lambda: torch.zeros(6, dtype=torch.float64))
        self.routing = None
        self.batches = 0
        self.optimizer_steps = 0
        self.skipped_nonfinite_gradients = 0
        self.objective_steps = 0
        self.objectives = Counter()
        self.batch_tasks = Counter()
        self.optimizer_batch_tasks = Counter()
        self.replay_samples = Counter()
        self.gradients = Counter()
        self.gradient_counts = Counter()
        self.gradient_checks = 0

    def state_dict(self):
        return {name: dict(value) if isinstance(value, dict) else value
                for name, value in vars(self).items() if name not in {"criterion", "site_metrics"}}

    def load_state_dict(self, state):
        for name, value in state.items():
            current = getattr(self, name)
            if isinstance(current, dict):
                current.update(value)
            else:
                setattr(self, name, value)

    @torch.no_grad()
    def update(self, output, batch, objective, auxiliary, updated):
        criterion = self.criterion
        labels = supervision(batch, output["latent"].device)
        target, types, weights, valid = labels
        error = criterion.normalized_error(output, batch, labels)
        indices = types.clamp(0, len(TARGET_TYPES) - 1)
        normalized_squared = error.double().square()
        difference = error.double() * criterion.label_normalization.std[indices].double()
        target = target.double().masked_fill(~valid, 0)
        terms = torch.stack((valid, target, target.square(), difference.square(), difference.abs(),
                             normalized_squared, weights, weights * criterion.error_loss(error).double()), dim=-1).cpu()
        types = types.cpu().tolist()
        sources = criterion.sources(batch, len(terms))
        products = batch.get("product", sources)
        sites = batch.get("site_id", [None] * len(terms))
        present = set()
        for row, task, source, product, site in zip(terms, types, sources, products, sites):
            if not row[0]:
                continue
            name = TARGET_TYPES[task]
            self.groups["task", name] += row
            self.groups["source", source, name] += row
            self.groups["product", product] += row
            present.add(name)
            if self.site_metrics and site is not None and product not in GRID_PRODUCTS:
                self.sites[source, name, product, _site_key(site)] += row[:6]
        if len(present) > 1:
            present.add("mixed")
        self.batch_tasks.update(present)
        if updated:
            self.optimizer_batch_tasks.update(present)

        routing = output["routing"].detach().double()
        mask = task_mask(batch, routing.device).ne(0)
        entropy = -(routing * routing.clamp_min(1e-15).log()).sum(dim=-1)
        winners = F.one_hot(routing.argmax(dim=-1), num_classes=routing.shape[-1])
        route_totals = torch.cat((routing, winners, entropy[..., None], torch.ones_like(mask)[..., None]), dim=-1)
        summary = route_totals.masked_fill(~mask[..., None], 0).sum(dim=0).cpu()
        self.routing = summary if self.routing is None else self.routing + summary

        names, values = zip(*{"objective": objective, **auxiliary}.items())
        self.objectives.update(dict(zip(names, torch.stack(values).detach().cpu().tolist())))
        self.objective_steps += 1

    def count_replay(self, pool, batch):
        _, types, _, valid = supervision(batch)
        for index, source in enumerate(self.criterion.sources(batch, len(types))):
            if valid[index]:
                self.replay_samples[pool, source, TARGET_TYPES[int(types[index])]] += 1

    @staticmethod
    def _scores(values, samples):
        count, sum_y, sum_y2, squared, absolute, norm_squared = values.tolist()
        variation = max(0.0, sum_y2 - sum_y**2 / count)
        return {"samples": samples, "rmse": math.sqrt(squared / count), "mae": absolute / count,
                "r2": 1.0 - squared / variation if variation else None, "normalized_mse": norm_squared / count}

    def result(self, num_experts):
        result = {"samples": 0, "flux_unit": FLUX_UNIT, "by_task": {}, "by_source": {}, "by_product": {},
                  "expert_count": num_experts, "expert_weights": {}, "expert_utilization": {},
                  "router_entropy": {}, "effective_experts": {}}
        selection, data_loss = 0.0, 0.0
        for (kind, name, *task), terms in self.groups.items():
            samples = int(terms[0])
            values = self._scores(terms[:6], samples)
            loss = float(terms[7] / terms[6])
            values["normalized_loss"] = loss
            if kind == "source":
                result["by_source"].setdefault(name, {})[task[0]] = values
                weight = self.criterion.source_weights.get(name, 1.0) * self.criterion.task_weights[task[0]]
                selection += weight * values["normalized_mse"]
                data_loss += weight * loss
            else:
                result[f"by_{kind}"][name] = values
                if kind == "task":
                    result["samples"] += samples
        result["data_loss"] = data_loss if result["samples"] else None
        result["weighted_normalized_mse"] = selection if result["samples"] else None

        if self.sites:
            pooled = defaultdict(lambda: torch.zeros(6, dtype=torch.float64))
            counts = Counter()
            for (source, task, product, site), terms in self.sites.items():
                for key in (("task", task), ("source", source, task), ("product", product)):
                    pooled[key] += terms / terms[0]
                    counts[key] += int(terms[0])
            grouped = {"by_task": {}, "by_source": {}, "by_product": {}}
            score = 0.0
            for key, values in pooled.items():
                kind, name, *task = key
                metrics = self._scores(values, counts[key])
                metrics["sites"] = int(values[0])
                if kind == "source":
                    grouped["by_source"].setdefault(name, {})[task[0]] = metrics
                    score += (self.criterion.source_weights.get(name, 1.0) * self.criterion.task_weights[task[0]]
                              * metrics["normalized_mse"])
                else:
                    grouped[f"by_{kind}"][name] = metrics
            result["site_equal"] = {**grouped, "weighted_normalized_mse": score}

        if self.routing is not None:
            for task, row in zip(TARGET_TYPES, self.routing):
                mass = float(row[-1])
                if mass:
                    entropy = float(row[-2]) / mass
                    result["expert_weights"][task] = (row[:num_experts] / mass).tolist()
                    result["expert_utilization"][task] = (row[num_experts:2 * num_experts] / mass).tolist()
                    result["router_entropy"][task] = entropy
                    result["effective_experts"][task] = math.exp(entropy)
        result.update(batches=self.batches, batches_by_task=dict(self.batch_tasks), optimizer_steps=self.optimizer_steps,
                      optimizer_steps_by_task=dict(self.optimizer_batch_tasks),
                      skipped_nonfinite_gradients=self.skipped_nonfinite_gradients)
        if self.objective_steps:
            result.update({name: value / self.objective_steps for name, value in self.objectives.items()})
        if self.replay_samples:
            result["replay_samples"] = {}
            for (pool, source, task), count in self.replay_samples.items():
                result["replay_samples"].setdefault(pool, {}).setdefault(source, {})[task] = count
        if self.gradient_checks:
            result["gradient_diagnostics"] = {"steps": self.gradient_checks,
                                              **{key: value / self.gradient_counts[key]
                                                 for key, value in self.gradients.items()}}
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
            raise ValueError("Physical response supervision requires online raw drivers")
        latent = encoder.accept_cached(batch["latent"])
    else:
        environment = batch["environment"]
        if response_supervision:
            environment = {**environment, "values": environment["values"].detach().requires_grad_(True)}
            batch = {**batch, "environment": environment}
        attention = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                     if response_supervision else nullcontext())
        # Match the FP32 representation used by the offline encoder.
        with attention, torch.autocast(device.type, enabled=False):
            latent = encoder(environment)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None and not response_supervision):
        output = model(latent)
    return {key: value.float() for key, value in output.items()}, batch


def _physical_loss(model, output, batch):
    references = batch.get("physical_response")
    if not references or references.keys() - set(TARGET_TYPES):
        raise ValueError("Physical response supervision requires trusted references keyed by emission or uptake")
    values = batch["environment"]["values"]
    mask = batch["environment"]["variable_mask"]
    losses = [
        physical_response_loss(output[task] * model.label_normalization.std[TARGET_TYPES.index(task)],
                               values, reference["reference"],
                               mask if "weights" not in reference else reference["weights"] * mask)
        for task, reference in references.items()
    ]
    return torch.stack(losses).sum()


def _distill(student, teacher, batch, criterion):
    _, types, weights, valid = supervision(batch, student["latent"].device)
    predicted = criterion.predicted_target(student, batch, apply_calibration=False, types=types)
    reference = criterion.predicted_target(teacher, batch, apply_calibration=False, types=types)
    losses = (predicted - reference).masked_fill(~valid, 0).square()
    objectives, _ = criterion.grouped_risk(losses, weights, types, batch)
    return sum(objectives)


def _usable(batch):
    return batch is not None and bool(supervision(batch)[-1].any())


def _loader_seed(loader, seed):
    if loader.generator is not None:
        loader.generator.manual_seed(seed)
    if hasattr(loader.batch_sampler, "set_progress"):
        loader.batch_sampler.set_progress(seed)


def _pass_batches(loader, seed, offset=0):
    _loader_seed(loader, seed)
    if hasattr(loader.batch_sampler, "set_progress"):
        loader.batch_sampler.set_progress(seed, offset)
        return iter(loader)
    return islice(loader, offset, None) if offset else iter(loader)


def _replay_batches(loader, seed, cursor):
    total = getattr(loader.batch_sampler, "total_batches", len(loader))
    if total == 0:
        raise ValueError("Historical Training replay has no batches")
    skipped = 0
    while True:
        for batch in _pass_batches(loader, seed + cursor["pass"], cursor["batch"]):
            cursor["batch"] += 1
            if _usable(batch):
                skipped = 0
                yield batch
            else:
                skipped += 1
                if skipped >= total:
                    raise RuntimeError("Historical Training replay has no batches with valid supervision")
        cursor.update({"pass": cursor["pass"] + 1, "batch": 0})


def _set_training_temperature(model, options, steps):
    schedule = options.get("temperature_schedule", {"kind": "fixed"})
    start = model.config["routing_temperature"]
    temperature = start
    if schedule["kind"] == "linear":
        fraction = min(steps / schedule["steps"], 1.0)
        temperature = start + fraction * (schedule["end"] - start)
    model.router.set_temperature(temperature)


def _shared_task_diagnostics(model, objectives, scaler, cosine):
    shared = [parameter for mlp in (model.shared, *model.experts) for parameter in mlp[:-1].parameters()
              if parameter.requires_grad]
    if not shared:
        return {}
    vectors = [
        torch.cat([gradient.detach().reshape(-1).float() for gradient in
                   torch.autograd.grad(scaler.scale(loss), shared, retain_graph=True)])
        for loss in objectives
    ]
    diagnostics = {f"shared_supervised_grad_norm_{task}": vector.norm() / scaler.get_scale()
                   for task, vector in zip(TARGET_TYPES, vectors)}
    if cosine and all(bool(torch.count_nonzero(vector)) for vector in vectors):
        diagnostics["task_gradient_cosine"] = F.cosine_similarity(*vectors, dim=0)
    return diagnostics


def _gradient_norm(parameters):
    norms = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    return torch.stack(norms).norm() if norms else None


def run_epoch(
    model, encoder, criterion, loader, device, amp_dtype, *,
    optimizer=None, scaler=None, options=None, replay_loaders=None, replay_state=None,
    replay_seed=0, step_offset=0, teacher=None, anchors=None, candidate=None, max_steps=None,
    require_samples=True, stats=None, on_step=None, strict_batches=False,
):
    options = options or {}
    training = optimizer is not None
    encoder.eval()
    if candidate:
        candidate.enable_training_mode()
    else:
        model.train(training)
    criterion.train(training and candidate is None)
    if stats is None:
        stats = EpochMetrics(criterion, sites=not training)
    distillation_weight = options.get("distillation_weight", 0.0) if training else 0.0
    physical_weight = options.get("physical_response_weight", 0.0) if training else 0.0
    interval = options.get("gradient_diagnostic_interval_steps", 0)
    if distillation_weight and teacher is None:
        raise ValueError("Distillation requires the frozen snapshot taken at stage entry")
    streams = {}
    if training:
        if replay_state is None:
            replay_state = {}
        streams = {
            name: _replay_batches(replay, replay_seed, replay_state.setdefault(name, {"pass": 0, "batch": 0}))
            for name, replay in (replay_loaders or {}).items()
        }
        scaler = scaler if scaler is not None else torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    for raw_batch in loader:
        stats.batches += 1
        if not _usable(raw_batch):
            if strict_batches:
                raise RuntimeError("A full simulation batch lost valid indexed supervision")
            if on_step is not None:
                on_step(stats, False)
            continue
        batch = to_device(raw_batch, device)
        if training:
            _set_training_temperature(model, options, step_offset + stats.optimizer_steps)
            optimizer.zero_grad(set_to_none=True)
        updated = False
        with torch.set_grad_enabled(training):
            output, batch = _forward(model, encoder, batch, device, amp_dtype,
                                     response_supervision=bool(physical_weight))
            terms = criterion(output, batch)
            objective = terms["loss"]
            task_objectives = terms["task_objectives"]
            auxiliary = {name: terms[name] for name in ("balance_loss", "sign_loss", "expert_loss")}
            replay_outputs = []
            for name, stream in streams.items():
                raw_replay = next(stream)
                stats.count_replay(name, raw_replay)
                replay_batch = to_device(raw_replay, device)
                replay_output, _ = _forward(model, encoder, replay_batch, device, amp_dtype)
                replay_terms = criterion(replay_output, replay_batch)
                replay_weight = options["replay"][name]["weight"]
                auxiliary[f"replay_{name}_loss"] = replay_terms["data_loss"]
                objective = objective + replay_weight * replay_terms["data_loss"]
                task_objectives = [active + replay_weight * historical
                                   for active, historical in zip(task_objectives, replay_terms["task_objectives"])]
                replay_outputs.append((replay_output, replay_batch))
            if distillation_weight:
                with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    old_output = teacher(output["latent"].detach())
                auxiliary["distillation_loss"] = _distill(output, old_output, batch, criterion)
                objective = objective + distillation_weight * auxiliary["distillation_loss"]
            if physical_weight:
                auxiliary["physical_response_loss"] = _physical_loss(model, output, batch)
                objective = objective + physical_weight * auxiliary["physical_response_loss"]
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError(f"MoE loss is not finite at batch {stats.batches}")
            if training:
                measure = interval and (step_offset + stats.optimizer_steps + 1) % interval == 0
                diagnostics = (_shared_task_diagnostics(model, task_objectives, scaler,
                                                        options.get("task_gradient_cosine", False))
                               if measure else {})
                previous_scale = scaler.get_scale()
                scaler.scale(objective).backward()
                scaler.unscale_(optimizer)
                if measure:
                    for task_index, task in enumerate(TARGET_TYPES):
                        norm = _gradient_norm(parameter for branch in model.router.branches
                                              for parameter in branch[task_index].parameters())
                        if norm is not None:
                            diagnostics[f"router_grad_norm_{task}"] = norm
                    for index, expert in enumerate(model.experts):
                        norm = _gradient_norm(expert.parameters())
                        if norm is not None:
                            diagnostics[f"expert_grad_norm_{index}"] = norm
                try:
                    torch.nn.utils.clip_grad_norm_(parameters, options.get("gradient_clip") or math.inf,
                                                   error_if_nonfinite=True)
                except RuntimeError as error:
                    if "non-finite" not in str(error):
                        raise
                    stats.skipped_nonfinite_gradients += 1
                else:
                    scaler.step(optimizer)
                    updated = True
                scaler.update()
                if updated and scaler.get_scale() < previous_scale:
                    updated = False
                    stats.skipped_nonfinite_gradients += 1
                if updated:
                    stats.optimizer_steps += 1
                    selected = [candidate.index] if candidate else None
                    for prediction, labels in [(output, batch), *replay_outputs]:
                        model.signatures.update(prediction["latent"], prediction["routing"],
                                                task_mask(labels, device), expert_indices=selected)
                    if anchors is not None:
                        anchors.update(output["latent"], batch)
                    if measure:
                        stats.gradient_checks += 1
                        values = {key: float(value) for key, value in diagnostics.items()}
                        finite = {key: value for key, value in values.items() if math.isfinite(value)}
                        stats.gradients.update(finite)
                        stats.gradient_counts.update(finite.keys())
                else:
                    optimizer.zero_grad(set_to_none=True)
        stats.update(output, batch, objective, auxiliary, updated)
        if on_step is not None:
            on_step(stats, updated)
        if max_steps is not None and stats.optimizer_steps >= max_steps:
            break
    metrics = stats.result(model.num_experts)
    if require_samples and not metrics["samples"]:
        raise RuntimeError("No valid methane supervision in this split")
    return metrics


def _export_metrics(output, history):
    columns = ("stage", "cohort", "epoch", "batch", "optimizer_steps", "event")

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


def _teacher_snapshot(model, state=None):
    snapshot = restore_model(state).to(next(model.parameters()).device) if state else deepcopy(model)
    if snapshot.label_normalization.statistics != model.label_normalization.statistics:
        raise ValueError("The stage teacher and student must use the same label normalization")
    return snapshot.eval().requires_grad_(False)


def _teacher_state(teacher):
    return {"model_config": teacher.metadata(), "model": teacher.state_dict()} if teacher is not None else None


def _diagnostics(model):
    model.refresh_fingerprints()
    return {"expert_ids": [record["expert_id"] for record in model.expert_records],
            **{key: value.detach().cpu().tolist() for key, value in model.signatures.diagnostics().items()}}


def _log_event(output, event):
    line = json.dumps(event, allow_nan=False)
    with (output / "train.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def _validation_probe(loader, batch_size, batches_per_product):
    if not batches_per_product:
        return None
    batches, offset = [], 0
    for segment in loader.dataset.segments:
        lengths = getattr(segment, "shard_lengths", [len(segment)])
        rows = np.cumsum([0, *lengths]).tolist()
        counts = np.cumsum([(length + batch_size - 1) // batch_size for length in lengths]).tolist()
        if counts and counts[-1]:
            for ordinal in np.linspace(0, counts[-1] - 1, min(counts[-1], batches_per_product), dtype=np.int64):
                shard = bisect_right(counts, ordinal)
                local = int(ordinal) - (counts[shard - 1] if shard else 0)
                start = rows[shard] + local * batch_size
                stop = min(start + batch_size, rows[shard + 1])
                batches.append(list(range(start + offset, stop + offset)))
        offset += len(segment)
    return DataLoader(loader.dataset, batch_sampler=batches, collate_fn=loader.collate_fn,
                      num_workers=loader.num_workers, generator=torch.Generator())


def _validation_score(metrics, options):
    selected = metrics.get("site_equal", {}) if options.get("validation_weighting") == "site_equal" else metrics
    return selected.get("weighted_normalized_mse")


def _fit(
    model, encoder, criterion, loaders, replay, config, options, stage, resume, teacher,
    device, amp_dtype, output, history, provenance,
):
    epochs = options["epochs"]
    optimizer = _optimizer(model, criterion, options)
    scheduler = _scheduler(optimizer, options, epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    progress = {"epoch": 0, "optimizer_steps": 0, "best_score": None, "best_epoch_score": None, "stale_epochs": 0}
    if resume:
        progress.update(resume["progress"])
        optimizer.load_state_dict(resume["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        restore_rng(resume["rng"], device)
    progress.update(history=history, data_provenance=provenance)

    train, validation = loaders
    probe = _validation_probe(validation, options["batch_size"], options.get("validation_probe_batches_per_product", 0))
    seed = options["training_seed"]
    patience = options.get("patience", 0)
    checkpoint_interval = options.get("checkpoint_interval_steps", 10000)
    validation_interval = options.get("validation_interval_steps", 0)
    log_interval = options.get("log_interval_steps", 0)

    def persist(path):
        save_checkpoint(path, model, encoder, config, stage=stage, criterion=criterion, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, progress=_checkpoint_progress(progress),
                        teacher=_teacher_state(teacher))

    def consider(metrics):
        score = _validation_score(metrics, options)
        if score is None or not math.isfinite(score):
            raise FloatingPointError("Validation model-selection score is not finite")
        if progress["best_score"] is None or score < progress["best_score"]:
            progress.update(best_score=score, best_checkpoint=str((output / "best.pt").resolve()))
            persist(output / "best.pt")
        return score

    def evaluate_selection(epoch):
        batches = _pass_batches(probe if probe is not None else validation, seed + epoch)
        return run_epoch(model, encoder, criterion, batches, device, amp_dtype)

    def scan(loader, split, epoch, *, anchors=None):
        training = split == "train"
        saved = progress.get("data_scan", {})
        if saved.get("split") != split:
            saved = {}
        offset = saved.get("batches", 0)
        metrics = EpochMetrics(criterion, sites=not training)
        replay_state = saved.get("replay_state", {})
        if saved:
            metrics.load_state_dict(saved["metrics"])
            if anchors is not None:
                anchors.selected = saved["anchor_samples"]
                anchors.generator.set_state(saved["anchor_rng"])

        def on_step(stats, updated):
            if updated:
                progress["optimizer_steps"] += 1
            steps = progress["optimizer_steps"]
            validate = updated and validation_interval and steps % validation_interval == 0
            interval = checkpoint_interval and (
                (updated and steps % checkpoint_interval == 0) or (not updated and stats.batches % checkpoint_interval == 0)
            )
            if interval or validate:
                progress["data_scan"] = {"split": split, "batches": stats.batches, "metrics": stats.state_dict(),
                                         "replay_state": replay_state}
                if anchors is not None:
                    progress["data_scan"].update(anchor_samples=anchors.selected, anchor_rng=anchors.generator.get_state())
            if updated and log_interval and steps % log_interval == 0:
                _log_event(output, {"stage": stage, "event": "optimizer_progress", "epoch": epoch + 1,
                                    "batch": stats.batches, "optimizer_steps": steps,
                                    "temperature": float(model.router.temperature),
                                    "train": stats.result(model.num_experts)})
            if validate:
                selection_metrics = evaluate_selection(epoch)
                score = consider(selection_metrics)
                _log_event(output, {"stage": stage, "event": "validation_steps", "epoch": epoch + 1,
                                    "batch": stats.batches, "optimizer_steps": steps, "score": score,
                                    "validation": selection_metrics, "temperature": float(model.router.temperature)})
                model.train()
                criterion.train()
            if interval or validate:
                persist(output / "last.pt")

        batches = _pass_batches(loader, seed + epoch, offset)
        return run_epoch(model, encoder, criterion, batches, device, amp_dtype, stats=metrics,
                         optimizer=optimizer if training else None, scaler=scaler, options=options,
                         replay_loaders=replay if training else None, replay_state=replay_state, replay_seed=seed + epoch,
                         step_offset=progress["optimizer_steps"] - metrics.optimizer_steps,
                         teacher=teacher, anchors=anchors, on_step=on_step, strict_batches=loader.dataset.streaming)

    for epoch in range(progress["epoch"], epochs):
        if patience and progress["stale_epochs"] >= patience:
            break
        if "data_scan" not in progress and "completed_train_pass" not in progress:
            seed_everything(seed + epoch)
        if "completed_train_pass" not in progress:
            count = min(options.get("anchor_count", 0), model.signatures.anchors.shape[1])
            anchors = (AnchorReservoir(count, seed, train.dataset.provenance, encoder.provenance)
                       if count > 0 and epoch == 0 and stage == "pretrain" else None)
            progress["completed_train_pass"] = scan(train, "train", epoch, anchors=anchors)
            if anchors is not None:
                anchors.apply(model)
            progress.pop("data_scan", None)
            persist(output / "last.pt")
        train_metrics = progress["completed_train_pass"]
        validation_metrics = scan(validation, "validation", epoch)
        selection = evaluate_selection(epoch) if probe is not None else validation_metrics
        score = _validation_score(selection, options)
        if score is None or not math.isfinite(score):
            raise FloatingPointError("Validation model-selection score is not finite")
        progress["epoch"] = epoch + 1
        progress.pop("completed_train_pass")
        progress.pop("data_scan", None)
        previous = progress["best_epoch_score"]
        if previous is None or score < previous - options.get("min_delta", 0.0):
            progress.update(best_epoch_score=score, stale_epochs=0)
        else:
            progress["stale_epochs"] += 1
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(score)
        elif scheduler is not None:
            scheduler.step()
        record = {"event": "epoch", "stage": stage, "epoch": epoch + 1, "optimizer_steps": progress["optimizer_steps"],
                  "temperature": float(model.router.temperature), "score": score,
                  "learning_rates": [group["lr"] for group in optimizer.param_groups],
                  "train": train_metrics, "validation": validation_metrics, "signatures": _diagnostics(model)}
        if probe is not None:
            record["validation_probe"] = selection
        history.append(record)
        consider(selection)
        persist(output / "last.pt")
        _export_metrics(output, history)
        _log_event(output, {"event": "epoch", "stage": stage, "epoch": epoch + 1, "score": score,
                            "optimizer_steps": progress["optimizer_steps"], "experts": model.num_experts,
                            "checkpoint": str(output / "last.pt")})
    return progress.get("best_checkpoint"), progress


def _matching_window(pairs):
    predictions, batches = zip(*pairs)
    output = {key: torch.cat([prediction[key] for prediction in predictions]) for key in predictions[0]}
    batch = {key: torch.cat([part[key] for part in batches])
             for key in ("target", "target_type", "sample_weight", "quality_mask")}
    batch["source"] = [source for part in batches for source in part["source"]]
    return output, batch


def _continual(
    model, encoder, criterion, loaders, replay, held_out, config, options,
    resume, previous, teacher, device, amp_dtype, output, history, provenance, cohort,
):
    growth = GrowthManager(max_experts=options["max_experts"], **options["growth"])
    candidate_training = options["candidate_training"]
    criterion.requires_grad_(False).eval()
    train, validation = loaders
    growth_optimizer = _optimizer(model, criterion, options)
    progress = {"epoch": 0, "batch": 0, "monitor": {}, "optimizer_steps": 0, "best_score": None}
    inherited = resume or previous
    if inherited:
        growth.load_state_dict(inherited["growth"])
        if inherited["effective_config"]["training"]["optimizer"] != options["optimizer"]:
            raise ValueError("Continual initialization requires the inherited optimizer kind")
        group_options = [{key: value for key, value in group.items() if key != "params"}
                         for group in growth_optimizer.param_groups]
        growth_optimizer.load_state_dict(inherited["optimizer"])
        if previous:
            for group, requested in zip(growth_optimizer.param_groups, group_options):
                group.update(requested)
    if resume:
        progress.update(resume["progress"])
        restore_rng(resume["rng"], device)
    else:
        growth.streak = 0
    progress.update(history=history, data_provenance=provenance)

    def persist(path):
        save_checkpoint(path, model, encoder, config, stage="continual", criterion=criterion,
                        optimizer=growth_optimizer, progress=_checkpoint_progress(progress), growth=growth,
                        teacher=_teacher_state(teacher))

    epochs = options["epochs"]
    candidate_epochs = options["candidate_epochs"]
    candidate_steps = options["candidate_steps"]
    seed = options["training_seed"]
    probe = _validation_probe(validation, options["batch_size"], options["validation_probe_batches_per_product"])

    def save_best(metrics):
        if probe is not None:
            metrics = run_epoch(model, encoder, criterion, probe, device, amp_dtype, require_samples=False)
        score = _validation_score(metrics, options)
        if score is not None and not math.isfinite(score):
            raise FloatingPointError("Validation model-selection score is not finite")
        if score is not None and (progress["best_score"] is None or score < progress["best_score"]):
            progress.update(best_score=score, best_checkpoint=str((output / "best.pt").resolve()))
            persist(output / "best.pt")

    # Candidate shuffling is independent of the live model's matching scan.
    candidate_loader = DataLoader(
        train.dataset, batch_size=train.batch_size, num_workers=train.num_workers,
        collate_fn=train.collate_fn, shuffle=bool(len(train.dataset)), generator=torch.Generator(),
    )
    for epoch in range(progress["epoch"], epochs):
        if not progress["batch"]:
            seed_everything(seed + epoch)
        skip = progress["batch"]
        stats = progress["monitor"] or {"batches": 0, "valid_samples": 0, "ready": 0, "triggered": 0}
        progress["monitor"] = stats
        matching, matching_count = [], 0
        for batch_index, raw_batch in enumerate(_pass_batches(train, seed + epoch, skip), start=skip + 1):
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
            # Keep only the outputs and labels required for expert matching.
            matching.append((
                {key: prediction[key] for key in ("base", "expert_outputs", "latent", *TARGET_TYPES)},
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
            replay_old = (run_epoch(model, encoder, criterion, held_out, device, amp_dtype, require_samples=False)
                          if held_out is not None else {"samples": 0, "data_loss": None})
            enough = (new_old["samples"] >= growth.min_validation_samples
                      and replay_old["samples"] >= growth.min_replay_samples)
            candidate = None
            new_candidate = replay_candidate = {"data_loss": None}
            validation_birth = None
            candidate_training_metrics = []
            if enough:
                candidate = CandidateTrainer(model, proposal["parent"], expert_id=proposal["candidate_id"])
                candidate_opts = {**options, "learning_rate": options["candidate_learning_rate"]}
                candidate_optimizer = _optimizer(candidate.model, criterion, candidate_opts)
                candidate_scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
                candidate_updates = 0
                for candidate_epoch in range(candidate_epochs):
                    candidate_seed = seed + epoch + batch_index + candidate_epoch
                    seed_everything(candidate_seed)
                    batches = _pass_batches(candidate_loader, candidate_seed)

                    def log_step(metrics, updated):
                        step = progress["optimizer_steps"] + candidate_updates + metrics.optimizer_steps
                        interval = options["log_interval_steps"]
                        if updated and interval and step % interval == 0:
                            _log_event(output, {"stage": "continual", "event": "candidate_optimizer_progress",
                                                "epoch": epoch + 1, "batch": batch_index, "optimizer_steps": step,
                                                "candidate": proposal["candidate_id"],
                                                "temperature": float(candidate.model.router.temperature),
                                                "train": metrics.result(candidate.model.num_experts)})

                    metrics = run_epoch(candidate.model, encoder, criterion, batches, device, amp_dtype,
                                        optimizer=candidate_optimizer, scaler=candidate_scaler, options=candidate_opts,
                                        replay_loaders=replay, replay_seed=candidate_seed, step_offset=candidate_updates,
                                        teacher=teacher, candidate=candidate, max_steps=candidate_steps, on_step=log_step)
                    candidate_training_metrics.append(metrics)
                    candidate_updates += metrics["optimizer_steps"]
                progress["optimizer_steps"] += candidate_updates
                if candidate_updates:
                    # Evaluate both models at the live router temperature after any adaptation schedule.
                    candidate.model.router.set_temperature(float(model.router.temperature))
                    new_candidate = run_epoch(candidate.model, encoder, criterion, validation, device, amp_dtype)
                    replay_candidate = run_epoch(candidate.model, encoder, criterion, held_out, device, amp_dtype)
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
                next_optimizer = _optimizer(updated, criterion, options)
                for name, parameter in updated.named_parameters():
                    if parameter in candidate_optimizer.state:
                        next_optimizer.state[parameter] = candidate_optimizer.state[parameter]
                    elif name in old_parameters and old_parameters[name] in growth_optimizer.state:
                        next_optimizer.state[parameter] = growth_optimizer.state[old_parameters[name]]
                model, growth_optimizer = updated, next_optimizer
            record = {"event": "growth", "stage": "continual", "cohort": cohort, "epoch": epoch + 1,
                      "batch": batch_index, "proposal": proposal, "growth": event,
                      "optimizer_steps": progress["optimizer_steps"], "candidate_train": candidate_training_metrics,
                      "validation": {"old": new_old, "candidate": new_candidate},
                      "replay_validation": {"old": replay_old, "candidate": replay_candidate},
                      "expert_count": model.num_experts, "signatures": _diagnostics(model)}
            history.append(record)
            if event["status"] == "accepted":
                persist(validation_birth)
                save_best(new_candidate)
            persist(output / "last.pt")
            _export_metrics(output, history)
            _log_event(output, {"stage": "continual", "event": "growth", "epoch": epoch + 1, "batch": batch_index,
                                "status": event["status"], "experts": model.num_experts})

        if matching:
            stats["incomplete_match"] = growth.observe(model, *_matching_window(matching), criterion)
        if not stats["valid_samples"]:
            stats["status"] = "pending_insufficient_data"
        validation_metrics = run_epoch(model, encoder, criterion, validation, device, amp_dtype, require_samples=False)
        progress.update(epoch=epoch + 1, batch=0, monitor={})
        history.append({"event": "epoch", "stage": "continual", "epoch": epoch + 1,
                        "cohort": cohort, "monitor": stats, "validation": validation_metrics,
                        "optimizer_steps": progress["optimizer_steps"], "temperature": float(model.router.temperature),
                        "expert_count": model.num_experts, "signatures": _diagnostics(model)})
        save_best(validation_metrics)
        persist(output / "last.pt")
        _export_metrics(output, history)
    return {"checkpoint": str(output / "last.pt"), "best_checkpoint": progress.get("best_checkpoint"),
            "experts": model.num_experts, "history": history}


def _stage_loaders(data, partitions, options, encoder, gatherer):
    return build_loaders(data, partitions, batch_size=options.get("batch_size", 16),
                         num_workers=options.get("num_workers", 0), training_seed=options["training_seed"],
                         gatherer=gatherer, encoder_metadata=encoder.provenance)


STAGE_SOURCES = {"pretrain": ["TEM", "MeMo"], "finetune": ["FLUXNET", "Chamber"],
                 "continual": ["FLUXNET", "Chamber"]}
TRAINING_DEFAULTS = {
    "device": "auto", "cpu_threads": None, "training_seed": 42, "batch_size": 16, "num_workers": 0, "epochs": 1,
    "learning_rate": 0.001, "weight_decay": 0.0001, "optimizer": "adamw", "scheduler": "none",
    "scheduler_factor": 0.5, "scheduler_patience": 2, "mixed_precision": "fp32", "gradient_clip": 1.0,
    "patience": 0, "min_delta": 0.0, "loss": "mse", "huber_delta": 1.0, "task_weights": {},
    "source_weights": {}, "balance_weight": 0.0, "min_utilization": 0.05, "sign_weight": 0.0,
    "expert_weight": 0.0, "physical_response_weight": 0.0, "calibration_sources": [], "anchor_count": 32,
    "output_dir": "moe/runs", "temperature_schedule": {"kind": "fixed"}, "validation_weighting": "sample_equal",
    "validation_interval_steps": 0, "validation_probe_batches_per_product": 0, "checkpoint_interval_steps": 10000,
    "log_interval_steps": 1000, "gradient_diagnostic_interval_steps": 1000, "task_gradient_cosine": False,
    "replay": {}, "distillation_weight": 0.0,
}
CONTINUAL_DEFAULTS = {
    "candidate_training": True, "max_experts": 32, "candidate_epochs": 2, "candidate_steps": 100,
    "candidate_learning_rate": 0.0001, "growth_validation_sources": ["FLUXNET", "Chamber"], "growth_validation_data": {},
    "growth": {},
}
DATA_DEFAULTS = {
    "simulation_dir": "data/methane_data/simulations", "observations_dir": "data/methane_data/observations",
    "embedding_mode": "online", "gather_mode": "history", "cache_path": None, "cache_dtype": "float32",
    "cache_chunk_rows": 4096, "split_seed": 42, "split_manifest_dir": "moe/runs/splits-v2",
    "simulation_mode": "full", "simulation_split_mode": "temporal", "simulation_spatial_block_degrees": 5,
    "simulation_date_range": ["1979-01-01", "9999-01-01"], "simulation_tile_mib": 16,
    "simulation_daily_start": "1979-01-01", "samples_per_simulation_source": {"train": 12000, "validation": 3000},
    "observation_split_mode": "temporal", "observation_short_sites": "train_only",
    "observation_date_range": ["1980-01-01", "2026-01-01"], "observation_site_ids": None,
    "observation_cohort": None, "adapters": {},
}
CACHE_OPTIONS = {"embedding_mode", "cache_path", "cache_dtype"}


def _check_keys(mapping, allowed, section):
    if unknown := mapping.keys() - allowed:
        raise ValueError(f"Unsupported {section} options: {sorted(unknown)}")


def _data_options(base, overrides):
    _check_keys(overrides, DATA_DEFAULTS.keys(), "data")
    adapters, updates = base["adapters"], overrides.get("adapters", {})
    _check_keys(updates, ADAPTERS.keys(), "data.adapters")
    return {**base, **overrides, "adapters": {
        key: {**adapters.get(key, {}), **updates.get(key, {})} for key in adapters.keys() | updates.keys()
    }}


def _check_cache_options(data):
    if data["embedding_mode"] not in {"online", "cached"} or data["gather_mode"] not in {"history", "query_window"}:
        raise ValueError("Invalid embedding_mode or gather_mode")
    if data["cache_dtype"] not in STORAGE_DTYPES or data["cache_chunk_rows"] < 1:
        raise ValueError("Expected cache_dtype float32/float16 and positive cache_chunk_rows")
    if data["embedding_mode"] == "cached" and not data["cache_path"]:
        raise ValueError("Cached mode requires data.cache_path")


def _settings(config, stage):
    _check_keys(config, {"encoder", "data", "moe", "training"}, "MoE")
    encoder = config.get("encoder", {})
    _check_keys(encoder, ENCODER_OPTIONS, "encoder")
    model = config.get("moe", {})
    _check_keys(model, signature(ContinualMethaneMoE).parameters.keys() - {"label_normalization"}, "moe")
    training = config.get("training", {})
    _check_keys(training, STAGE_SOURCES.keys() | {"common"}, "training")
    common = training.get("common", {})
    _check_keys(common, TRAINING_DEFAULTS.keys(), "training.common")
    for name in STAGE_SOURCES:
        extra = CONTINUAL_DEFAULTS.keys() if name == "continual" else set()
        _check_keys(training.get(name, {}), TRAINING_DEFAULTS.keys() | {"data", "sources"} | extra, f"training.{name}")

    overrides = training.get(stage, {})
    data = _data_options(_data_options(DATA_DEFAULTS, config.get("data", {})), overrides.get("data", {}))
    _check_cache_options(data)
    options = {**TRAINING_DEFAULTS, "sources": STAGE_SOURCES[stage], **common, **overrides}
    options.pop("data", None)
    for key in ("source_weights", "task_weights"):
        defaults = ({adapter["source"]: 1.0 for adapter in ADAPTERS.values()} if key == "source_weights"
                    else dict.fromkeys(TARGET_TYPES, 1.0))
        requested = common.get(key, {}) | overrides.get(key, {})
        _check_keys(requested, defaults.keys(), key)
        options[key] = defaults | requested
    options["calibration_sources"] = sorted(set(options["calibration_sources"]))
    if set(options["calibration_sources"]) - options["source_weights"].keys():
        raise ValueError("calibration_sources must name supported data sources")
    options["optimizer"] = options["optimizer"].lower()
    options["scheduler"] = options["scheduler"].lower()
    if options["optimizer"] not in {"adam", "adamw", "sgd"} or options["scheduler"] not in {"none", "plateau", "cosine"}:
        raise ValueError("Invalid optimizer or scheduler")
    for key in ("epochs", "batch_size", "checkpoint_interval_steps"):
        if not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("validation_interval_steps", "validation_probe_batches_per_product", "log_interval_steps",
                "gradient_diagnostic_interval_steps", "num_workers", "training_seed", "anchor_count", "patience"):
        if not isinstance(options[key], int) or options[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("distillation_weight", "physical_response_weight", "balance_weight", "sign_weight", "expert_weight",
                "gradient_clip", "min_delta"):
        if not math.isfinite(options[key]) or options[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if options["physical_response_weight"] and data["embedding_mode"] == "cached":
        raise ValueError("Physical response supervision requires data.embedding_mode: online")
    if options["validation_weighting"] not in {"sample_equal", "site_equal"} or (
        stage == "pretrain" and options["validation_weighting"] == "site_equal"
    ):
        raise ValueError("Validation weighting must be sample_equal, or site_equal for observation stages")
    schedule = options["temperature_schedule"]
    if schedule.get("kind") == "fixed":
        _check_keys(schedule, {"kind"}, "fixed temperature schedule")
    elif schedule.get("kind") == "linear":
        _check_keys(schedule, {"kind", "end", "steps"}, "linear temperature schedule")
        if not math.isfinite(schedule.get("end", 0.0)) or schedule.get("end", 0.0) <= 0 or not (
            isinstance(schedule.get("steps"), int) and schedule["steps"] > 0
        ):
            raise ValueError("Linear temperature requires a positive end temperature and optimizer step count")
    else:
        raise ValueError("temperature_schedule.kind must be fixed or linear")
    options["replay"] = {}
    for section in (common, overrides):
        _check_keys(section.get("replay", {}), {"simulation", "observation"}, "replay")
    for group, sources in (("simulation", STAGE_SOURCES["pretrain"]), ("observation", STAGE_SOURCES["finetune"])):
        pool = {"weight": 0.0, "sources": sources, "data": {}}
        for section in (common, overrides):
            entry = section.get("replay", {}).get(group, {})
            _check_keys(entry, pool.keys(), f"replay.{group}")
            _check_keys(entry.get("data", {}), CACHE_OPTIONS, f"replay.{group}.data")
            pool = {**pool, **entry, "data": {**pool["data"], **entry.get("data", {})}}
        if not math.isfinite(pool["weight"]) or pool["weight"] < 0:
            raise ValueError(f"Replay {group} weight must be finite and nonnegative")
        product_names(pool["sources"], grid=group == "simulation")
        options["replay"][group] = pool
    if stage == "continual":
        options = {**CONTINUAL_DEFAULTS, **options}
        defaults = {key: parameter.default for key, parameter in signature(GrowthManager).parameters.items()
                    if key != "max_experts"}
        _check_keys(options["growth"], defaults.keys(), "growth")
        options["growth"] = defaults | options["growth"]
        _check_keys(options["growth_validation_data"], CACHE_OPTIONS, "growth_validation_data")
        if options["candidate_epochs"] < 1 or (options["candidate_steps"] is not None and options["candidate_steps"] < 1):
            raise ValueError("Candidate epochs and steps must be positive")
        if options["scheduler"] != "none" or options["validation_interval_steps"]:
            raise ValueError("Continual evaluation runs at growth decisions and epoch end; use scheduler none")
        if not options["candidate_training"] and (
            options["distillation_weight"] or any(pool["weight"] for pool in options["replay"].values())
        ):
            raise ValueError("Continual replay and distillation require candidate_training")
    return encoder, model, data, options


def _serialize_config(config):
    return json.loads(json.dumps(config, default=str, allow_nan=False))


def run_training(config, stage, checkpoint=None, resume=None, output_dir=None, *, prepare_data=False, build_cache=False):
    """Fit one stage, inspect its frozen partitions, or resume an offline embedding build."""
    if stage not in STAGE_SOURCES:
        raise ValueError(f"Unknown MoE training stage: {stage}")
    encoder_options, model_options, data, options = _settings(config, stage)
    source_checkpoint = resume or checkpoint
    payload = load_checkpoint(source_checkpoint) if source_checkpoint else None
    prior_stages = {"pretrain": (), "finetune": ("pretrain", "finetune", "continual"),
                    "continual": ("finetune", "continual")}[stage]
    if resume:
        if payload["stage"] != stage:
            raise ValueError(f"Resume checkpoint must be from the {stage} stage")
    elif (payload and payload["stage"] not in prior_stages) or (stage == "continual" and not payload):
        raise ValueError(f"{stage} expects prior checkpoint stages {prior_stages}")
    if (stage == "finetune" and checkpoint and payload["stage"] != "pretrain" and not data["observation_cohort"]):
        raise ValueError("Fine-tuning from an observation checkpoint requires a new data.observation_cohort")
    past_settings = payload["effective_config"] if payload else None
    if payload and any(data[key] != past_settings["data"][key] for key in ("gather_mode", "simulation_daily_start")):
        raise ValueError("MoE checkpoint lineage must retain gather_mode and simulation_daily_start")
    if options["distillation_weight"] and payload is None:
        raise ValueError("Distillation requires the model checkpoint from before this stage")
    replay_pools = {name: pool for name, pool in options["replay"].items() if pool["weight"]}
    historical_validation = (options["growth_validation_sources"]
                             if stage == "continual" and options["candidate_training"] else [])
    replay_sources = sorted(set(historical_validation) | {source for pool in replay_pools.values() for source in pool["sources"]})
    past_data = payload["progress"].get("data_provenance", {}) if payload else None
    data_stage = "continual" if stage == "finetune" and data["observation_cohort"] else stage
    active, split_history, _, intersections = stage_manifests(
        data, options["sources"], data_stage, past_data, resume=bool(resume), replay_sources=replay_sources,
    )
    if prepare_data:
        return audit_summary(active, intersections, options["batch_size"])

    device, amp_dtype = _device_and_amp(options)
    if options["cpu_threads"] is not None:
        torch.set_num_threads(options["cpu_threads"])
    seed_everything(options["training_seed"])
    expected_provenance = payload["encoder_provenance"] if payload else None
    encoder = load_environmental_encoder(encoder_options, expected_provenance=expected_provenance).to(device)
    if payload:
        restore_encoder(encoder, payload)
    if build_cache:
        if not data["cache_path"]:
            raise ValueError("Building embeddings requires data.cache_path")
        return build_embedding_cache(data, loader_partitions([active]), encoder, device, options["batch_size"])

    if payload:
        model = restore_model(payload).to(device)
        for key, value in model_options.items():
            if key != "routing_temperature" and _serialize_config(value) != model.config[key]:
                raise ValueError(f"MoE checkpoint topology or behavior conflicts with moe.{key}")
        normalization_source = past_settings["label_normalization"]["training_manifest"]
    else:
        model_options = {"latent_dim": encoder.latent_dim, **model_options}
        model = ContinualMethaneMoE(**model_options, label_normalization=training_label_statistics(active)).to(device)
        normalization_source = active[0]
    if model.latent_dim != encoder.latent_dim:
        raise ValueError("MoE and encoder latent dimensions differ")
    if resume and bool(payload["teacher"]) != bool(options["distillation_weight"]):
        raise ValueError("Resume must preserve the saved stage teacher and distillation configuration")
    teacher = _teacher_snapshot(model, payload["teacher"] if resume else None) if options["distillation_weight"] else None
    if not resume:
        configured_temperature = model_options.get("routing_temperature", float(model.router.temperature))
        model.config["routing_temperature"] = configured_temperature
        model.router.set_temperature(configured_temperature)
    elif "routing_temperature" in model_options and model_options["routing_temperature"] != model.config["routing_temperature"]:
        raise ValueError("Resume requires the saved moe.routing_temperature")

    data_by_pool = {name: {**data, **pool["data"]} for name, pool in replay_pools.items()}
    if historical_validation:
        data_by_pool["growth_validation"] = {**data, **options["growth_validation_data"]}
    for settings in data_by_pool.values():
        _check_cache_options(settings)
    all_data = [data, *data_by_pool.values()]
    gatherer = encoder.make_gatherer() if any(item["embedding_mode"] == "online" for item in all_data) else None
    output = Path(output_dir or (Path(resume).parent if resume else
                                config.get("training", {}).get(stage, {}).get("output_dir",
                                    Path(options["output_dir"]) / stage)))
    output.mkdir(parents=True, exist_ok=True)
    replay_config = {name: pool | {"data": {key: data_by_pool[name][key] for key in sorted(CACHE_OPTIONS)}}
                     if name in data_by_pool else pool for name, pool in options["replay"].items()}
    effective = _serialize_config({
        "stage": stage,
        "source_checkpoint": past_settings["source_checkpoint"] if resume else str(Path(checkpoint).resolve()) if checkpoint else None,
        "encoder": {**encoder.provenance, "frozen": True},
        "moe": {**model.config, "routing_type": "dense"},
        "data": data,
        "training": {**options, "device": str(device), "output_dir": str(output.resolve()), "replay": replay_config,
                     "mixed_precision": str(amp_dtype).removeprefix("torch.") if amp_dtype is not None else "fp32",
                     "enabled_constraints": {key: bool(options[f"{key}_weight"])
                                             for key in ("balance", "sign", "expert", "physical_response")},
                     "calibration_trainable": bool(options["calibration_sources"]) and stage != "continual"},
        "label_normalization": {"tasks": model.label_normalization.statistics, "training_manifest": normalization_source},
        "best_validation": {"metric": "weighted_normalized_mse", "weighting": options["validation_weighting"],
                            "batches_per_product": options["validation_probe_batches_per_product"]},
    })
    if resume and (effective | {"training": effective["training"] | {"output_dir": past_settings["training"]["output_dir"]}}
                   != past_settings):
        changed = [key for key in effective if key != "training" and effective[key] != past_settings.get(key)]
        changed.extend(f"training.{key}" for key, value in effective["training"].items()
                       if key != "output_dir" and value != past_settings["training"].get(key))
        raise ValueError(f"Resume requires the saved effective configuration; changed: {', '.join(changed)}")
    previous = payload if payload and stage == payload["stage"] == "continual" and not resume else None
    history = list(payload["progress"].get("history", [])) if resume or previous else []
    criterion = _make_criterion(options, model, device, payload["criterion"] if payload else None)
    opened = []

    def make_loaders(settings, partitions):
        pair = _stage_loaders(settings, partitions, options, encoder, gatherer)
        opened.extend(pair)
        return pair

    try:
        loaders = make_loaders(data, loader_partitions([active]))
        replay = {}
        provenance = {"active_manifest": active[0], "historical_manifests": [ref for ref, _ in split_history],
                      "intersections": intersections, "train": loaders[0].dataset.provenance,
                      "validation": loaders[1].dataset.provenance}
        for name, pool in replay_pools.items():
            partitions = loader_partitions(split_history, product_names(pool["sources"]))
            replay[name] = make_loaders(data_by_pool[name], partitions)[0]
            provenance.setdefault("replay_train", {})[name] = replay[name].dataset.provenance
        held_out = None
        if historical_validation:
            partitions = loader_partitions(split_history, product_names(historical_validation))
            held_out = make_loaders(data_by_pool["growth_validation"], partitions)[1]
            provenance["growth_validation"] = held_out.dataset.provenance
        if resume and provenance != past_data:
            raise ValueError("Training data provenance changed since the checkpoint was saved")
        if stage != "continual":
            for loader in loaders:
                if not all(product["valid_labels"] for product in loader.dataset.provenance["products"].values()):
                    split = loader.dataset.provenance["split"]
                    raise ValueError(f"Every active product requires valid labels in the {split} partition")
        temporary = output / ".effective_config.json.tmp"
        temporary.write_text(json.dumps(effective, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "effective_config.json")
        if stage == "continual":
            cohort = sum(manifest["settings"]["kind"] != "simulation" for _, manifest in split_history)
            return _continual(model, encoder, criterion, loaders, replay, held_out, effective, options,
                              payload if resume else None, previous, teacher, device, amp_dtype,
                              output, history, provenance, cohort)
        best, _ = _fit(model, encoder, criterion, loaders, replay, effective, options, stage,
                       payload if resume else None, teacher, device, amp_dtype, output, history, provenance)
        return {"checkpoint": str(output / "last.pt"), "best_checkpoint": best, "history": history}
    finally:
        for loader in opened:
            loader.dataset.close()
            cache = loader.collate_fn.cache
            if cache is not None:
                cache.close()
        if gatherer is not None:
            for source in gatherer.sources.values():
                source.close()
