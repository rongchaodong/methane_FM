from __future__ import annotations

from copy import deepcopy
import math

import torch

from .model import TARGET_TYPES, supervision, task_mask


class CandidateTrainer:
    def __init__(self, model, parent, expert_id=None):
        self.original = model
        self.model = deepcopy(model)
        self.index = self.model.split_expert(parent, expert_id=expert_id)
        self.model.requires_grad_(False)
        self.model.experts[self.index].requires_grad_(True)
        self.model.router.branches[self.index].requires_grad_(True)
        self.enable_training_mode()

    def enable_training_mode(self):
        # 冻结参数时也固定已存在分支的 dropout 行为。
        self.model.eval()
        self.model.experts[self.index].train()
        self.model.router.branches[self.index].train()

    def trainable_parameters(self):
        return (parameter for parameter in self.model.parameters() if parameter.requires_grad)

    @property
    def parameter_delta(self):
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def promote(self):
        for name, parameter in self.original.named_parameters():
            self.model.get_parameter(name).requires_grad_(parameter.requires_grad)
        self.model.train(self.original.training)
        return self.model


class GrowthManager:
    def __init__(
        self, max_experts=32, min_samples=32, min_validation_samples=32, min_replay_samples=32,
        persistence=2, latent_weight=1.0, response_weight=1.0, distance_threshold=None,
        loss_threshold=None, gain_threshold=0.0, forgetting_weight=1.0, parameter_weight=0.0,
        max_forgetting=0.0,
    ):
        if min(max_experts, min_samples, min_validation_samples, min_replay_samples, persistence) < 1:
            raise ValueError("Expert cap, growth sample minima and persistence must be positive")
        self.max_experts = max_experts
        self.min_samples = min_samples
        self.min_validation_samples = min_validation_samples
        self.min_replay_samples = min_replay_samples
        self.persistence = persistence
        self.latent_weight = latent_weight
        self.response_weight = response_weight
        self.distance_threshold = distance_threshold
        self.loss_threshold = loss_threshold
        self.gain_threshold = gain_threshold
        self.forgetting_weight = forgetting_weight
        self.parameter_weight = parameter_weight
        self.max_forgetting = max_forgetting
        self.streak = 0
        self.next_proposal = 0
        self.history = []

    def state_dict(self):
        return {"streak": self.streak, "next_proposal": self.next_proposal, "history": deepcopy(self.history)}

    def load_state_dict(self, state):
        self.streak = state["streak"]
        self.next_proposal = state["next_proposal"]
        self.history = deepcopy(state["history"])

    @staticmethod
    def _response_loss(criterion, prediction, batch):
        output = {"emission": prediction[:, 0], "uptake": prediction[:, 1], "net": prediction.sum(dim=-1)}
        losses, weights = criterion.sample_losses(output, batch)
        sources = criterion.sources(batch, len(losses))
        return sum(
            criterion.source_weights.get(source, 1.0)
            * criterion.weighted_mean(losses, weights * criterion.source_mask(sources, source, losses.device))
            for source in sorted(set(sources))
        )

    @torch.no_grad()
    def observe(self, model, output, batch, criterion):
        latent = output["latent"]
        _, _, _, valid = supervision(batch, latent.device)
        count = int(valid.sum())
        event = {
            "triggered": False, "ready": False, "expert_count": model.num_experts, "new_count": count,
            "parent": None, "proposal_id": None, "candidate_id": None,
        }
        if count < self.min_samples:
            self.streak = 0
            event.update(status="pending_insufficient_data", reason="matching_samples", streak=0)
            self.history.append(deepcopy(event))
            return event

        distance = model.signatures.distances(latent)
        weights = task_mask(batch, latent.device)
        averages = ((distance.masked_fill(~valid[:, None, None], 0) * weights[..., None]).sum(dim=0)
                    / torch.where(weights.sum(dim=0) > 0, weights.sum(dim=0), 1)[:, None])
        known = model.signatures.initialized
        supported = known.sum(dim=-1, keepdim=True)
        mean_known = averages.sum(dim=-1, keepdim=True) / supported.clamp_min(1)
        # 缺失的 Signature 使用同一任务已知距离的平均值；全缺失时距离项为零。
        averages = torch.where(known, averages, mean_known)
        latent_scores = (averages * weights.sum(dim=0)[:, None]).sum(dim=0) / weights.sum()
        base = output["base"]
        residuals = output["expert_outputs"]
        response_scores = torch.stack([
            self._response_loss(criterion, base + residual, batch) for residual in residuals.unbind(dim=-1)
        ])
        scores = self.latent_weight * latent_scores + self.response_weight * response_scores
        baseline = torch.stack([output[task] for task in TARGET_TYPES[:2]], dim=1)
        old_loss = float(self._response_loss(criterion, baseline, batch))
        if not torch.isfinite(scores).all() or not math.isfinite(old_loss):
            raise ValueError("Non-finite values encountered during expert matching")

        parent = int(scores.argmin())
        matching_distance = float(scores[parent])
        event.update(
            parent=parent, parent_expert_id=model.expert_records[parent]["expert_id"], old_loss=old_loss,
            match_distance=matching_distance, expert_scores=scores.tolist(),
            latent_distances=latent_scores.tolist(), response_losses=response_scores.tolist(),
        )
        if model.num_experts >= self.max_experts:
            self.streak = 0
            event.update(status="not_triggered", reason="max_experts")
        elif self.distance_threshold is None or self.loss_threshold is None:
            self.streak = 0
            event.update(status="not_triggered", reason="thresholds_unconfigured")
        elif matching_distance > self.distance_threshold and old_loss > self.loss_threshold:
            self.streak += 1
            if self.streak >= self.persistence:
                proposal_id = self.next_proposal
                self.next_proposal += 1
                event.update(triggered=True, ready=True, status="proposed", proposal_id=proposal_id,
                             candidate_id=f"candidate_{proposal_id:04d}")
                self.history.append({**deepcopy(event), "streak": self.streak})
            else:
                event.update(status="not_triggered", reason="persistence")
        else:
            self.streak = 0
            event.update(status="not_triggered", reason="thresholds")
        event["streak"] = self.streak
        return event

    def decide(
        self, model, candidate, *, new_old_loss=None, new_candidate_loss=None, replay_old_loss=None,
        replay_candidate_loss=None, new_count=0, replay_count=0, birth_checkpoint=None, proposal=None,
    ):
        event = {
            "proposal_id": proposal.get("proposal_id") if proposal else None,
            "candidate_id": proposal.get("candidate_id") if proposal else None,
            "parent_expert_id": proposal.get("parent_expert_id") if proposal else None,
            "new_count": new_count, "replay_count": replay_count,
        }
        if candidate is not None:
            if candidate.original is not model:
                raise ValueError("Candidate was created from a different live model")
            record = candidate.model.expert_records[candidate.index]
            event.update(candidate_id=record["expert_id"], parent_expert_id=record["parent_expert_id"])
        selected = model
        values = (new_old_loss, new_candidate_loss, replay_old_loss, replay_candidate_loss)
        if new_count < self.min_validation_samples or replay_count < self.min_replay_samples:
            event.update(status="pending_insufficient_data", reason="held_out_samples")
        elif candidate is None or any(value is None or not math.isfinite(value) for value in values):
            event.update(status="rejected", reason="invalid_validation_metrics")
        else:
            new_old_loss, new_candidate_loss, replay_old_loss, replay_candidate_loss = map(float, values)
            improvement = new_old_loss - new_candidate_loss
            forgetting = replay_candidate_loss - replay_old_loss
            delta_parameters = candidate.parameter_delta
            gain = improvement - self.forgetting_weight * max(forgetting, 0) - self.parameter_weight * delta_parameters
            event.update(new_old_loss=new_old_loss, new_candidate_loss=new_candidate_loss,
                         replay_old_loss=replay_old_loss, replay_candidate_loss=replay_candidate_loss,
                         improvement=improvement, forgetting=forgetting,
                         parameter_delta=delta_parameters, gain=gain)
            accepted = (
                model.num_experts < self.max_experts and gain > self.gain_threshold and forgetting <= self.max_forgetting
            )
            event["status"] = "accepted" if accepted else "rejected"
            if accepted:
                selected = candidate.promote()
                selected.refresh_fingerprints()
        self.streak = 0
        event["expert_count"] = selected.num_experts
        if candidate is not None:
            record.update(status=event["status"], validation=deepcopy(event))
            if event["status"] == "accepted" and birth_checkpoint is not None:
                record["birth_checkpoint"] = str(birth_checkpoint)
        self.history.append(deepcopy(event))
        return selected, event
