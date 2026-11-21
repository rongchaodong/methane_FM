"""Build historical contexts and future targets at configurable resolutions."""

from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import Dataset

from .specs import QuerySpec, as_datetime


def build_queries(config):
    """Build anchors from explicit queries or from locations and a date range."""
    if isinstance(config, list):
        return [QuerySpec(**item) for item in config]
    start, stop = as_datetime(config["start_time"]), as_datetime(config["end_time"])
    stride = timedelta(days=config.get("stride_days", 1))
    duration = timedelta(days=config.get("query_days", 1))
    if min(stride, duration) <= timedelta(0):
        raise ValueError("Query stride and duration must be positive.")
    queries = []
    while start + duration <= stop:
        queries.extend(QuerySpec(**location, start_time=start, end_time=start + duration)
                       for location in config["locations"])
        start += stride
    return queries


def _next_month(start):
    return start.replace(year=start.year + (start.month == 12), month=start.month % 12 + 1)


def future_query(anchor, target):
    """Use the requested lead time, then align monthly targets to calendar months."""
    horizon = target.get("horizon_days", 1)
    if horizon < 0:
        raise ValueError("Target horizon_days must be nonnegative.")
    start = anchor.start_time + timedelta(days=horizon)
    midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)
    period = target.get("period", "day")
    if period == "month":
        month_start = midnight.replace(day=1)
        start = month_start if start == month_start else _next_month(month_start)
        end = _next_month(start)
    elif period == "day":
        start = midnight + timedelta(days=int(start != midnight))
        end = start + timedelta(days=target.get("duration_days", 1))
    else:
        raise ValueError(f"Unknown target period: {period!r}. Use 'day' or 'month'.")
    delta_lat, delta_lon = target.get("spatial_support", (anchor.delta_lat, anchor.delta_lon))
    return QuerySpec(anchor.lat, anchor.lon, start, end, delta_lat, delta_lon)


class EnvironmentalJEPADataset(Dataset):
    def __init__(self, gatherer, queries, future_targets):
        if not future_targets:
            raise ValueError("At least one future target is required.")
        self.gatherer = gatherer
        self.queries = queries
        self.future_targets = future_targets

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        context = self.gatherer.gather(query)
        if not any(
            self.gatherer.registry.sources[block.source].frequency == "daily"
            for block in context.blocks
        ):
            return None
        targets = []
        for spec in self.future_targets:
            future = future_query(query, spec)
            target = self.gatherer.gather(future, start_time=future.start_time,
                                          end_time=future.end_time, include_static=False)
            if not any(
                self.gatherer.registry.sources[block.source].frequency == "daily"
                for block in target.blocks
            ):
                return None
            targets.append(target)
        return {"context": context, "targets": targets}


class EnvironmentalCollator:
    """Pad variable-length context and target batches for attention."""

    def __init__(self, registry):
        self.driver_ids = {name: index for index, name in enumerate(registry.drivers)}
        self.source_ids = {name: index for index, name in enumerate(registry.sources)}

    def collate_contexts(self, contexts):
        batch_size = len(contexts)
        token_count = max(sum(map(len, item.blocks)) for item in contexts)
        if any(not item.blocks for item in contexts):
            raise ValueError("A query returned no valid environmental observations.")
        values = torch.zeros(batch_size, token_count, len(self.driver_ids))
        valid = torch.zeros_like(values, dtype=torch.bool)
        metadata = torch.zeros(batch_size, token_count, 9)
        static = torch.zeros(batch_size, token_count, dtype=torch.bool)
        sources = torch.zeros(batch_size, token_count, dtype=torch.long)
        for row, context in enumerate(contexts):
            offset = 0
            for block in context.blocks:
                span = slice(offset, offset + len(block))
                ids = [self.driver_ids[name] for name in block.driver_names]
                values[row, span, ids] = torch.from_numpy(block.values)
                valid[row, span, ids] = torch.from_numpy(block.variable_mask)
                metadata[row, span] = torch.from_numpy(block.metadata)
                static[row, span] = torch.from_numpy(block.static_mask)
                sources[row, span] = self.source_ids[block.source]
                offset += len(block)
        return {
            "query": torch.from_numpy(np.stack([item.query.metadata() for item in contexts])),
            "values": values,
            "variable_mask": valid,
            "metadata": metadata,
            "static_mask": static,
            "source_ids": sources,
            "token_mask": valid.any(dim=-1),
        }

    def __call__(self, samples):
        samples = [sample for sample in samples if sample is not None]
        if not samples:
            return None
        return {
            "context": self.collate_contexts([sample["context"] for sample in samples]),
            "targets": [self.collate_contexts(group)
                        for group in zip(*(sample["targets"] for sample in samples))],
        }
