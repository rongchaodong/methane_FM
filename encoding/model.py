"""Coordinate queries attend to environmental tokens on their native grids."""

from pathlib import Path

import torch
from torch import nn

from .specs import DriverRegistry


def coordinate_features(metadata, static_mask=None):
    """Encode spherical position, calendar time, and spatial/temporal support."""
    lat, lon = torch.deg2rad(metadata[..., 0]), torch.deg2rad(metadata[..., 1])
    position = torch.stack((lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()), dim=-1)
    phase = 2 * torch.pi * metadata[..., 7] / metadata[..., 8].clamp_min(1)
    duration = (metadata[..., 5] - metadata[..., 4]).clamp_min(0)
    time = torch.stack((phase.sin(), phase.cos(), (metadata[..., 6] - 2000) / 100,
                        torch.log1p(duration) / 6), dim=-1)
    if static_mask is not None:
        time = time.masked_fill(static_mask.unsqueeze(-1), 0)
    support = torch.log1p(metadata[..., 2:4])
    return torch.cat((position, time, support), dim=-1)


class QueryEncoder(nn.Module):
    """Project query position, time interval, and spatial support into Q."""

    def __init__(self, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(9, hidden_dim), nn.LayerNorm(hidden_dim),
                                         nn.Dropout(dropout))

    def forward(self, query):
        query = query.to(dtype=self.projection[0].weight.dtype)
        return self.projection(coordinate_features(query))


class EnvironmentalTokenEncoder(nn.Module):
    """Aggregate observed drivers within each native space-time token into K/V."""

    def __init__(self, registry, hidden_dim=128, dropout=0.0):
        super().__init__()
        drivers = list(registry.drivers.values())
        self.variable_embeddings = nn.Embedding(len(drivers), hidden_dim)
        self.value_directions = nn.Embedding(len(drivers), hidden_dim)
        self.source_embeddings = nn.Embedding(len(registry.sources), hidden_dim)
        self.category_embeddings = nn.ModuleDict({
            str(index): nn.Embedding(driver.categories, hidden_dim)
            for index, driver in enumerate(drivers) if driver.categories is not None
        })
        self.register_buffer("means", torch.tensor([driver.mean for driver in drivers], dtype=torch.float32))
        self.register_buffer("stds", torch.tensor([driver.std for driver in drivers], dtype=torch.float32))
        # Native grid spacing is already in the token's spatial support.
        self.projection = nn.Sequential(nn.Linear(2 * hidden_dim + 11, hidden_dim),
                                         nn.LayerNorm(hidden_dim), nn.Dropout(dropout))

    def forward(self, batch):
        values = batch["values"].to(dtype=self.variable_embeddings.weight.dtype)
        valid = batch["variable_mask"] & torch.isfinite(values)
        values = values.masked_fill(~valid, 0)
        aggregated = values.new_zeros(*values.shape[:2], self.variable_embeddings.embedding_dim)
        for index in range(values.shape[-1]):
            key = str(index)
            if key in self.category_embeddings:
                embedding = self.category_embeddings[key]
                categories = values[..., index]
                valid[..., index] &= ((categories >= 0) & (categories < embedding.num_embeddings)
                                      & (categories == categories.floor()))
                features = embedding(categories.clamp(0, embedding.num_embeddings - 1).long())
            else:
                normalized = (values[..., index] - self.means[index]) / self.stds[index]
                # A driver-specific slope preserves which variable supplied each value.
                features = normalized.unsqueeze(-1) * self.value_directions.weight[index]
            features = features + self.variable_embeddings.weight[index]
            aggregated = aggregated + features * valid[..., index:index + 1].to(features.dtype)
        aggregated = aggregated / valid.sum(dim=-1, keepdim=True).clamp_min(1)

        metadata = batch["metadata"].to(dtype=values.dtype)
        query = batch["query"].to(dtype=values.dtype)
        static = batch["static_mask"]
        token_midpoint = (metadata[..., 4] + metadata[..., 5]) / 2
        query_midpoint = ((query[:, 4] + query[:, 5]) / 2).unsqueeze(-1)
        relative_time = ((token_midpoint - query_midpoint) / 365.25).masked_fill(static, 0)
        source_ids = batch["source_ids"]
        features = torch.cat((aggregated, self.source_embeddings(source_ids),
                              coordinate_features(metadata, static),
                              relative_time.unsqueeze(-1), static.unsqueeze(-1).to(values.dtype)), dim=-1)
        token_mask = batch["token_mask"] & valid.any(dim=-1)
        return self.projection(features), token_mask


class EnvironmentalEncoder(nn.Module):
    """Use one coordinate-conditioned attention path for point and area queries."""

    def __init__(self, registry, hidden_dim=128, latent_dim=128, attention_heads=4,
                 cross_attention_layers=1, dropout=0.0):
        super().__init__()
        if cross_attention_layers < 1:
            raise ValueError("cross_attention_layers must be at least 1")
        self.registry = DriverRegistry.from_dict(registry.to_dict())
        self.hidden_dim, self.latent_dim = hidden_dim, latent_dim
        self._config = dict(hidden_dim=hidden_dim, latent_dim=latent_dim, attention_heads=attention_heads,
                            cross_attention_layers=cross_attention_layers, dropout=dropout)
        self.query_encoder = QueryEncoder(hidden_dim, dropout)
        self.token_encoder = EnvironmentalTokenEncoder(registry, hidden_dim, dropout)
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
            for _ in range(cross_attention_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(cross_attention_layers)])
        self.final_projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, batch):
        tokens, mask = self.token_encoder(batch)
        if not mask.any(dim=-1).all():
            raise ValueError("Each query must have at least one valid environmental token")
        query = self.query_encoder(batch["query"]).unsqueeze(1)
        for attention, norm in zip(self.attention, self.norms):
            attended, _ = attention(query, tokens, tokens, key_padding_mask=~mask, need_weights=False)
            query = norm(query + attended)
        return self.final_projection(query.squeeze(1))

    def to_config(self):
        return dict(self._config)


def save_encoder(path, encoder):
    """Export the online encoder with its driver schema and model configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"registry": encoder.registry.to_dict(), "model_config": encoder.to_config(),
                "state_dict": encoder.state_dict()}, path)


def load_encoder(path, map_location="cpu"):
    """Load an exported encoder and disable dropout for inference."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    registry = DriverRegistry.from_dict(checkpoint["registry"])
    encoder = EnvironmentalEncoder(registry, **checkpoint["model_config"])
    encoder.load_state_dict(checkpoint["state_dict"])
    return encoder.to(map_location).eval()
