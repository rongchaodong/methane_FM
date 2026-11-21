"""Gather native environmental windows without interpolation or spatial subsampling."""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from .specs import EPOCH, QuerySpec, TokenBlock, as_datetime


@dataclass
class GatheredContext:
    query: QuerySpec
    blocks: list[TokenBlock]


class MultiResolutionGatherer:
    def __init__(self, registry, sources, sampling=None):
        self.registry, self.sources = registry, sources
        sampling = sampling or {}
        self.history_days = sampling.get("history_days", 90)
        self.max_tokens = int(sampling.get("max_tokens", 4096))
        self.spatial_selection = sampling.get("spatial_selection", "centers")
        if self.history_days <= 0 or self.max_tokens < 1:
            raise ValueError("history_days and max_tokens must be positive")
        if self.spatial_selection not in {"centers", "intersects"}:
            raise ValueError("spatial_selection must be centers or intersects")
        if set(registry.sources) - set(sources):
            raise ValueError("A data source implementation is required for each registered source")

    def gather(self, query, *, start_time=None, end_time=None, include_static=True):
        """Default to history ending at query.start_time; targets pass an explicit window.

        Global annual tables are slow context covariates. Like static grids, they are
        excluded when include_static=False, so future targets cannot read them.
        """
        start = as_datetime(start_time) if start_time is not None else query.start_time - timedelta(days=self.history_days)
        end = as_datetime(end_time) if end_time is not None else query.start_time
        if end <= start:
            raise ValueError("Gather requires start_time < end_time")
        blocks, count = [], 0
        for name, source in self.sources.items():
            spec = self.registry.sources[name]
            global_annual = spec.frequency == "annual" and not spec.coordinates
            if not include_static and (spec.frequency == "static" or global_annual):
                continue
            block = source.read_window(query, start, end, self.spatial_selection,
                                       max_tokens=self.max_tokens - count)
            if spec.frequency != "static" and len(block):
                start_day, end_day = ((value - EPOCH).total_seconds() / 86400 for value in (start, end))
                outside = ((block.metadata[:, 4] >= end_day) | (block.metadata[:, 5] > end_day)
                           | (block.metadata[:, 5] <= block.metadata[:, 4]))
                if not global_annual:
                    outside |= block.metadata[:, 4] < start_day
                if outside.any():
                    raise ValueError(f"{name}: dynamic token support lies outside the requested causal window")
            valid = block.variable_mask & np.isfinite(block.values)
            block.variable_mask = valid
            block = block.select(np.flatnonzero(valid.any(axis=1)))
            count += len(block)
            if count > self.max_tokens:
                raise ValueError("Native query exceeds max_tokens; increase the budget or reduce the query window")
            if len(block):
                blocks.append(block)
        return GatheredContext(query, blocks)

    __call__ = gather
