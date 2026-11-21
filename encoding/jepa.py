"""Predict future environmental representations with an EMA target encoder."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .model import QueryEncoder, save_encoder


class LatentPredictor(nn.Module):
    """Condition a past representation on future coordinates and support."""

    def __init__(self, hidden_dim=128, latent_dim=128):
        super().__init__()
        self.query_encoder = QueryEncoder(hidden_dim=hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, context_embedding, future_query):
        query_embedding = self.query_encoder(future_query)
        return self.projection(torch.cat((context_embedding, query_embedding), dim=-1))


class EnvironmentalJEPA(nn.Module):
    """Online encoder, frozen EMA target, and a future latent predictor."""

    def __init__(self, encoder, ema_momentum=0.996, variance_weight=0.0,
                 variance_target=1.0):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = deepcopy(encoder).requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = LatentPredictor(encoder.hidden_dim, encoder.latent_dim)
        self.ema_momentum = ema_momentum
        self.variance_weight = variance_weight
        self.variance_target = variance_target

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self):
        """Update after a successful optimizer step."""
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.lerp_(online, 1 - self.ema_momentum)
        for online, target in zip(self.online_encoder.buffers(), self.target_encoder.buffers()):
            target.copy_(online)

    def forward(self, batch):
        context_embedding = self.online_encoder(batch["context"])
        predictions, targets = [], []
        for future in batch["targets"]:
            predictions.append(self.predictor(context_embedding, future["query"]))
            with torch.no_grad():
                targets.append(self.target_encoder(future))
        predictions = torch.stack(predictions, dim=1)
        targets = torch.stack(targets, dim=1)
        jepa_loss = F.mse_loss(predictions.float(), targets.float())
        variance_loss = jepa_loss.new_zeros(())
        if self.variance_weight:
            # This batch variance penalty discourages, but does not prevent, collapse.
            standard_deviation = (context_embedding.float().var(dim=0, unbiased=False) + 1e-4).sqrt()
            variance_loss = F.relu(self.variance_target - standard_deviation).mean()
        return {
            "loss": jepa_loss + self.variance_weight * variance_loss,
            "jepa_loss": jepa_loss,
            "variance_loss": variance_loss,
            "predictions": predictions,
            "targets": targets,
            "context_embedding": context_embedding,
        }

    def save_encoder(self, path):
        save_encoder(path, self.online_encoder)
