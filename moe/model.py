from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TARGET_TYPES = ("emission", "uptake", "net")


def _mlp(input_dim, hidden_dims, output_dim, activation, normalization, dropout):
    activation_types = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}
    if activation not in activation_types:
        raise ValueError(f"Unknown activation: {activation}")
    if normalization not in ("none", "layer"):
        raise ValueError(f"Unknown normalization: {normalization}")
    layers = []
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, hidden_dim))
        if normalization == "layer":
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation_types[activation]())
        if dropout:
            layers.append(nn.Dropout(dropout))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class SignatureManager(nn.Module):
    def __init__(self, num_experts, latent_dim, anchor_count, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("weight", torch.zeros(2, num_experts))
        self.register_buffer("mean", torch.zeros(2, num_experts, latent_dim))
        self.register_buffer("m2", torch.zeros_like(self.mean))
        self.register_buffer("anchors", torch.zeros(2, anchor_count, latent_dim))
        self.register_buffer("anchor_mask", torch.zeros(2, anchor_count, dtype=torch.bool))
        self.register_buffer("fingerprint", torch.zeros(2, num_experts, anchor_count))
        self.anchor_metadata = [None, None]

    def get_extra_state(self):
        return self.anchor_metadata

    def set_extra_state(self, state):
        self.anchor_metadata = state

    @property
    def initialized(self):
        return self.weight > 0

    @property
    def variance(self):
        divisor = torch.where(self.initialized, self.weight, 1)
        return (self.m2 / divisor[..., None]).clamp_min(0)

    @torch.no_grad()
    def reset_statistics(self):
        self.weight.zero_()
        self.mean.zero_()
        self.m2.zero_()

    def distances(self, latent):
        # Signature 的统计存储精度与 AMP 的前向精度分开，避免小方差触发溢出。
        dtype = torch.promote_types(latent.dtype, self.mean.dtype)
        deviation = latent.to(dtype)[:, None, None, :] - self.mean
        distance = (deviation.square() / (self.variance + self.epsilon)).mean(dim=-1)
        return distance.masked_fill(~self.initialized, 0)

    @torch.no_grad()
    def update(self, latent, routing, task_mask, expert_indices=None):
        latent = latent.detach().to(self.mean.dtype)
        usable = torch.isfinite(latent).all(dim=-1)
        latent = latent.masked_fill(~usable[:, None], 0)
        weights = routing.detach().to(self.weight.dtype) * task_mask.to(self.weight.dtype)[..., None]
        weights = weights.masked_fill(~usable[:, None, None], 0)
        if expert_indices is not None:
            selected = torch.zeros_like(self.weight[0], dtype=torch.bool)
            selected[list(expert_indices)] = True
            weights = weights.masked_fill(~selected, 0)
        mass = weights.sum(dim=0)
        divisor = torch.where(mass > 0, mass, 1)
        # 逐批合并加权中心矩，避免 E[z²] - E[z]² 的消减误差。
        mean = torch.einsum("bqk,bd->qkd", weights, latent) / divisor[..., None]
        centered = latent[:, None, None, :] - mean
        m2 = (weights[..., None] * centered.square()).sum(dim=0)
        total = self.weight + mass
        ratio = mass / torch.where(total > 0, total, 1)
        delta = mean - self.mean
        self.m2.add_(m2 + delta.square() * (self.weight * ratio)[..., None])
        self.mean.add_(delta * ratio[..., None])
        self.weight.copy_(total)

    @torch.no_grad()
    def split(self, parent):
        # 均分父专家的历史权重，同时保留均值、方差和已初始化状态。
        self.weight[:, parent].mul_(0.5)
        self.m2[:, parent].mul_(0.5)
        for name in ("weight", "mean", "m2", "fingerprint"):
            value = getattr(self, name)
            setattr(self, name, torch.cat((value, value[:, parent : parent + 1]), dim=1))

    @torch.no_grad()
    def set_anchors(self, task, latent, metadata, seed=0):
        task = TARGET_TYPES.index(task) if isinstance(task, str) else task
        if task not in (0, 1) or latent.ndim != 2 or latent.shape[-1] != self.anchors.shape[-1]:
            raise ValueError("Anchors require an emission or uptake task and a [N, latent_dim] tensor")
        latent = latent.detach()
        selected = torch.isfinite(latent).all(dim=-1).nonzero().flatten()
        capacity = self.anchors.shape[1]
        if len(selected) > capacity:
            generator = torch.Generator().manual_seed(seed)
            positions = torch.randperm(len(selected), generator=generator)[:capacity].to(selected.device)
            selected = selected[positions]
        count = len(selected)
        self.anchors[task].zero_()
        self.anchor_mask[task].zero_()
        self.fingerprint[task].zero_()
        self.anchors[task, :count].copy_(latent[selected])
        self.anchor_mask[task, :count] = True
        metadata = deepcopy(metadata)
        if metadata is not None and len(metadata.get("samples", ())) == len(latent):
            metadata["samples"] = [metadata["samples"][index] for index in selected.tolist()]
        self.anchor_metadata[task] = metadata
        return count

    @torch.no_grad()
    def diagnostics(self):
        delta = self.mean[:, :, None, :] - self.mean[:, None, :, :]
        variance = self.variance
        distance = (delta.square() / ((variance[:, :, None, :] + variance[:, None, :, :]) / 2
                                      + self.epsilon)).mean(dim=-1)
        signature_valid = self.initialized[:, :, None] & self.initialized[:, None, :]
        normalized = F.normalize(self.fingerprint, dim=-1, eps=1e-6)
        similarity = normalized @ normalized.transpose(-2, -1)
        norms = self.fingerprint.norm(dim=-1)
        fingerprint_valid = ((norms > 0)[:, :, None] & (norms > 0)[:, None, :]
                             & self.anchor_mask.any(dim=-1)[:, None, None])
        total = self.weight.sum(dim=-1, keepdim=True)
        return {
            "signature_distance": distance.masked_fill(~signature_valid, 0),
            "signature_valid": signature_valid,
            "fingerprint_similarity": similarity.masked_fill(~fingerprint_valid, 0),
            "fingerprint_valid": fingerprint_valid,
            "signature_utilization": self.weight / torch.where(total > 0, total, 1),
        }


class TaskSpecificRouter(nn.Module):
    def __init__(self, latent_dim, num_experts, routing_type, top_k, temperature, **mlp_options):
        super().__init__()
        if routing_type not in ("dense", "top_k", "top-k"):
            raise ValueError(f"Unknown routing type: {routing_type}")
        if temperature <= 0 or top_k <= 0:
            raise ValueError("Router temperature and top_k must be positive")
        self.routing_type = routing_type
        self.top_k = top_k
        self.temperature = temperature
        self.family_count = num_experts
        self.split_parents = []
        self.branches = nn.ModuleList(_mlp(latent_dim, output_dim=2, **mlp_options)
                                      for _ in range(num_experts))
        self.register_buffer("family", torch.arange(num_experts))
        self.register_buffer("log_mass", torch.zeros(num_experts))

    @torch.no_grad()
    def split(self, parent):
        self.branches.append(deepcopy(self.branches[parent]))
        self.split_parents.append(parent)
        # 在温度缩放后的分数上减去 log(2)，均分父分支概率质量。
        self.log_mass[parent] -= math.log(2)
        self.log_mass = torch.cat((self.log_mass, self.log_mass[parent : parent + 1]))
        self.family = torch.cat((self.family, self.family[parent : parent + 1]))

    def forward(self, latent, signature_penalty=None):
        logits = torch.stack([branch(latent) for branch in self.branches], dim=-1)
        logits = logits.to(torch.promote_types(logits.dtype, self.log_mass.dtype))
        if signature_penalty is not None:
            logits = logits - signature_penalty
        scaled = logits / self.temperature
        scores = scaled + self.log_mass
        dense = scores.softmax(dim=-1)
        if self.routing_type == "dense" or self.top_k >= self.family_count:
            return dense, dense
        # 按逆拆分顺序合并家族质量；相等的两个子分数合并后保持原值和 top-k 平局。
        families = list(scaled.unbind(dim=-1))
        for parent in reversed(self.split_parents):
            child_score = families.pop()
            parent_score = families[parent]
            maximum = torch.maximum(parent_score, child_score)
            relative_mass = ((parent_score - maximum).exp() + (child_score - maximum).exp()) * 0.5
            families[parent] = maximum + relative_mass.log()
        family_scores = torch.stack(families, dim=-1)
        chosen = family_scores.topk(self.top_k, dim=-1).indices
        active = chosen[..., None].eq(self.family).any(dim=-2)
        return scores.masked_fill(~active, -torch.inf).softmax(dim=-1), dense


class ContinualMethaneMoE(nn.Module):
    def __init__(
        self, latent_dim=128, initial_num_experts=4, expert_hidden_dims=(128, 64),
        shared_hidden_dims=(128, 64), router_hidden_dims=(), activation="gelu",
        normalization="none", dropout=0.0, routing_temperature=1.0,
        signature_guided_routing=True, signature_weight=0.1, signature_epsilon=0.01,
        signature_anchor_count=64, routing_type="dense", top_k=2,
    ):
        super().__init__()
        if initial_num_experts < 1 or latent_dim < 1 or signature_epsilon <= 0 or signature_anchor_count < 0:
            raise ValueError("Invalid number of experts, latent dimension, or signature configuration")
        self.config = {
            "latent_dim": latent_dim, "initial_num_experts": initial_num_experts,
            "expert_hidden_dims": list(expert_hidden_dims), "shared_hidden_dims": list(shared_hidden_dims),
            "router_hidden_dims": list(router_hidden_dims), "activation": activation,
            "normalization": normalization, "dropout": dropout, "routing_temperature": routing_temperature,
            "signature_guided_routing": signature_guided_routing,
            "signature_weight": list(signature_weight) if isinstance(signature_weight, (tuple, list))
            else signature_weight,
            "signature_epsilon": signature_epsilon, "signature_anchor_count": signature_anchor_count,
            "routing_type": routing_type, "top_k": top_k,
        }
        self.latent_dim = latent_dim
        self.signature_guided_routing = signature_guided_routing
        signature_weight = torch.as_tensor(signature_weight, dtype=torch.float32)
        if signature_weight.numel() not in (1, 2) or (signature_weight < 0).any():
            raise ValueError("Signature weight must contain one or two nonnegative values")
        self.register_buffer("signature_weight", signature_weight.expand(2).reshape(1, 2, 1).clone())
        options = {"activation": activation, "normalization": normalization, "dropout": dropout}
        self.shared = _mlp(latent_dim, shared_hidden_dims, 2, **options)
        self.experts = nn.ModuleList(
            _mlp(latent_dim, expert_hidden_dims, 2, **options) for _ in range(initial_num_experts)
        )
        self.router = TaskSpecificRouter(
            latent_dim, initial_num_experts, routing_type, top_k, routing_temperature,
            hidden_dims=router_hidden_dims, **options,
        )
        self.signatures = SignatureManager(initial_num_experts, latent_dim, signature_anchor_count, signature_epsilon)
        self.expert_records = [
            {"expert_id": f"expert_{index:04d}", "parent_expert_id": None, "creation_stage": "pretrain",
             "status": "active", "validation": None, "birth_checkpoint": None}
            for index in range(initial_num_experts)
        ]

    @property
    def num_experts(self):
        return len(self.experts)

    def forward(self, latent):
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected a [B, {self.latent_dim}] latent tensor, got {tuple(latent.shape)}")
        penalty = None
        if self.signature_guided_routing:
            penalty = self.signatures.distances(latent) * self.signature_weight
        routing, routing_dense = self.router(latent, penalty)
        base = self.shared(latent)
        residuals = torch.stack([expert(latent) for expert in self.experts], dim=-1)
        prediction = base + (routing * residuals).sum(dim=-1)
        return {
            "emission": prediction[:, 0], "uptake": prediction[:, 1], "net": prediction.sum(dim=-1),
            "routing": routing, "latent": latent, "expert_outputs": residuals,
            "base": base, "routing_dense": routing_dense,
        }

    @torch.no_grad()
    def refresh_fingerprints(self):
        for index, expert in enumerate(self.experts):
            training = expert.training
            expert.eval()
            self.signatures.fingerprint[0, index].copy_(expert(self.signatures.anchors[0])[:, 0])
            self.signatures.fingerprint[1, index].copy_(expert(self.signatures.anchors[1])[:, 1])
            expert.train(training)
        self.signatures.fingerprint.masked_fill_(~self.signatures.anchor_mask[:, None, :], 0)

    def split_expert(self, parent, expert_id=None, creation_stage="continual"):
        if not 0 <= parent < self.num_experts:
            raise IndexError(f"Unknown parent expert index: {parent}")
        position = self.num_experts
        expert_id = expert_id or f"expert_{position:04d}"
        if any(record["expert_id"] == expert_id for record in self.expert_records):
            raise ValueError(f"Duplicate expert ID: {expert_id}")
        self.experts.append(deepcopy(self.experts[parent]))
        self.router.split(parent)
        self.signatures.split(parent)
        self.expert_records.append({
            "expert_id": expert_id, "parent_expert_id": self.expert_records[parent]["expert_id"],
            "creation_stage": creation_stage, "status": "pending", "validation": None, "birth_checkpoint": None,
        })
        return position

    def metadata(self):
        return {"config": deepcopy(self.config), "experts": deepcopy(self.expert_records)}

    @classmethod
    def from_metadata(cls, metadata):
        model = cls(**metadata["config"])
        records = metadata["experts"]
        positions = {record["expert_id"]: index for index, record in enumerate(records[:model.num_experts])}
        for record in records[model.num_experts:]:
            index = model.split_expert(
                positions[record["parent_expert_id"]], record["expert_id"], record["creation_stage"]
            )
            positions[record["expert_id"]] = index
        model.expert_records = deepcopy(records)
        return model


def supervision(batch, device=None):
    target = torch.as_tensor(batch["target"], device=device).reshape(-1)
    target_type = batch["target_type"]
    if isinstance(target_type, str):
        target_type = [target_type] * len(target)
    if isinstance(target_type, (tuple, list)) and target_type and isinstance(target_type[0], str):
        target_type = [TARGET_TYPES.index(value) for value in target_type]
    types = torch.as_tensor(target_type, dtype=torch.long, device=target.device).reshape(-1).expand_as(target)
    weights = torch.as_tensor(batch.get("sample_weight", 1.0), dtype=torch.float32, device=target.device).reshape(-1)
    quality = torch.as_tensor(batch.get("quality_mask", True), dtype=torch.bool, device=target.device).reshape(-1)
    valid = (torch.isfinite(target) & torch.isfinite(weights) & (weights > 0) & quality
             & (types >= 0) & (types < len(TARGET_TYPES)))
    return target, types, weights.masked_fill(~valid, 0), valid


def task_mask(batch, device=None):
    _, types, weights, _ = supervision(batch, device=device)
    # net 监督包含两个分量，两个任务共享该样本的 Signature 权重。
    tasks = types[:, None].eq(torch.arange(2, device=types.device)) | types[:, None].eq(2)
    return tasks * weights[:, None]


class MoELoss(nn.Module):
    def __init__(
        self, kind="huber", huber_delta=1.0, source_weights=None, calibration_sources=(),
        balance_weight=0.0, sign_weight=0.0, expert_weight=0.0, min_utilization=0.05,
    ):
        super().__init__()
        if kind not in ("huber", "mse") or huber_delta <= 0:
            raise ValueError("Loss must be huber or mse with positive huber_delta")
        self.kind = kind
        self.huber_delta = huber_delta
        self.source_weights = source_weights or {}
        self.balance_weight = balance_weight
        self.sign_weight = sign_weight
        self.expert_weight = expert_weight
        self.min_utilization = min_utilization
        self.calibration = nn.ParameterDict({source: nn.Parameter(torch.zeros(2)) for source in calibration_sources})

    @staticmethod
    def sources(batch, size):
        sources = batch.get("source", "unknown")
        return [sources] * size if isinstance(sources, str) else list(sources)

    @staticmethod
    def source_mask(sources, source, device):
        return torch.tensor([value == source for value in sources], dtype=torch.bool, device=device)

    def predicted_target(self, output, batch, apply_calibration=True):
        physical = torch.stack([output[name] for name in TARGET_TYPES], dim=1).float()
        _, types, _, _ = supervision(batch, physical.device)
        prediction = physical.gather(1, types.clamp(0, len(TARGET_TYPES) - 1)[:, None])
        prediction = prediction[:, 0]
        if apply_calibration and self.calibration:
            sources = self.sources(batch, len(physical))
            for source, affine in self.calibration.items():
                if source in sources:
                    mask = self.source_mask(sources, source, physical.device)
                    prediction = torch.where(mask, (1 + affine[0]) * prediction + affine[1], prediction)
        return prediction

    metric_predictions = predicted_target

    def sample_losses(self, output, batch):
        prediction = self.predicted_target(output, batch)
        target, _, weights, valid = supervision(batch, prediction.device)
        error = prediction.masked_fill(~valid, 0) - target.masked_fill(~valid, 0).float()
        if self.kind == "mse":
            return error.square(), weights
        absolute = error.abs()
        quadratic = absolute.clamp_max(self.huber_delta)
        return 0.5 * quadratic.square() + self.huber_delta * (absolute - quadratic), weights

    @staticmethod
    def weighted_mean(values, weights):
        mass = weights.sum()
        return (values * weights).sum() / torch.where(mass > 0, mass, 1)

    def forward(self, output, batch):
        losses, weights = self.sample_losses(output, batch)
        _, types, _, _ = supervision(batch, losses.device)
        sources = self.sources(batch, len(losses))
        by_source = {
            source: self.weighted_mean(losses, weights * self.source_mask(sources, source, losses.device))
            for source in sorted(set(sources))
        }
        by_task = {
            name: self.weighted_mean(losses, weights * (types == index))
            for index, name in enumerate(TARGET_TYPES)
        }
        data_loss = sum(self.source_weights.get(source, 1.0) * value for source, value in by_source.items())
        effective = task_mask(batch, losses.device).float()
        mass = effective.sum(dim=0)
        present = mass > 0
        divisor = torch.where(present, mass, 1)
        routing = output["routing"].float()
        dense = output.get("routing_dense", routing).float()
        utilization = (routing * effective[..., None]).sum(dim=0) / divisor[:, None]
        dense_utilization = (dense * effective[..., None]).sum(dim=0) / divisor[:, None]
        entropy = -(routing * routing.clamp_min(1e-12).log()).sum(dim=-1)
        entropy = (entropy * effective).sum(dim=0) / divisor
        floor = min(self.min_utilization, 0.5 / routing.shape[-1])
        balance = ((floor - dense_utilization).relu().square() * present[:, None]).mean()
        sign = torch.stack(((-output["emission"]).relu(), output["uptake"].relu()), dim=-1)
        sign_loss = self.weighted_mean(sign.square(), effective)
        expert_loss = output["expert_outputs"].float().square().mean()
        loss = data_loss + self.balance_weight * balance + self.sign_weight * sign_loss + self.expert_weight * expert_loss
        return {
            "loss": loss, "data_loss": data_loss, "by_source": by_source, "by_task": by_task,
            "utilization": utilization, "entropy": entropy, "balance_loss": balance, "sign_loss": sign_loss,
            "expert_loss": expert_loss,
        }


def physical_response_loss(prediction, drivers, reference, weights=None):
    if not drivers.requires_grad or not prediction.requires_grad:
        raise ValueError("Physical response supervision requires online predictions connected to raw drivers")
    if prediction.shape[0] != drivers.shape[0] or reference.shape != drivers.shape:
        raise ValueError("Each physical sensitivity must match its sample and driver shape")
    gradient, = torch.autograd.grad(prediction.sum(), drivers, create_graph=True, allow_unused=True)
    if gradient is None:
        raise ValueError("Predictions are disconnected from the raw physical drivers")
    reference = reference.to(gradient)
    valid = torch.isfinite(reference)
    comparison = gradient.masked_fill(~valid, 0) - reference.masked_fill(~valid, 0)
    weights = valid if weights is None else torch.broadcast_to(weights.to(gradient), valid.shape).masked_fill(~valid, 0)
    return MoELoss.weighted_mean(comparison.square(), weights)
