"""Versioned, label-independent embeddings in bounded, disk-backed shards."""

from collections import OrderedDict, defaultdict
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from encoding.specs import EPOCH, as_datetime
from encoding.training import to_device

from .data import DATA_QUERY_VERSION, GRID_PRODUCTS, MoECollator, MoEDatasetAdapter, ShardBatchSampler
from .encoder import EMBEDDING_VERSION_KEYS, driver_inventory


CACHE_FORMAT = 2
STORAGE_DTYPES = {"float32": "<f4", "float16": "<f2"}


def _write_json(path, value, *, create=False):
    with NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
            if create:
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass
            else:
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _key(query, simulation):
    def micros(time):
        elapsed = time - EPOCH
        return (elapsed.days * 86400 + elapsed.seconds) * 1000000 + elapsed.microseconds

    return query.start_time.year, struct.pack(
        "<ddddqq?", query.lat, query.lon, query.delta_lat, query.delta_lon,
        micros(query.start_time), micros(query.end_time), simulation,
    )


class EmbeddingCache:
    def __init__(self, config, encoder_metadata, *, create=False):
        self.path = Path(config["cache_path"])
        metadata_path = self.path / "metadata.json"
        if self.path.is_file() or (not metadata_path.is_file() and not create):
            raise ValueError(f"Missing or incompatible embedding cache at {self.path}. Build a versioned cache directory.")
        self.latent_dim = encoder_metadata["model_config"]["latent_dim"]
        dtype = config.get("cache_dtype", "float32")
        self.dtype = np.dtype(STORAGE_DTYPES[dtype])
        gather_mode = config.get("gather_mode", "history")
        if gather_mode not in {"history", "query_window"}:
            raise ValueError("Embedding cache gather_mode must be history or query_window.")
        metadata = {key: encoder_metadata[key] for key in EMBEDDING_VERSION_KEYS}
        metadata.update(format_version=CACHE_FORMAT, data_query_version=DATA_QUERY_VERSION,
                        gather_mode=gather_mode, simulation_daily_start=as_datetime(
                            config.get("simulation_daily_start", "1979-01-01")).isoformat(),
                        latent_dim=self.latent_dim, encoding_dtype="float32", storage_dtype=dtype)
        if create and not metadata_path.is_file():
            self.path.mkdir(parents=True, exist_ok=True)
            if any(self.path.glob("*.sqlite3")):
                raise ValueError(f"Unversioned embedding shards in {self.path}. Build in a new directory.")
            _write_json(metadata_path, metadata, create=True)
        stored = json.loads(metadata_path.read_text())
        for key, value in metadata.items():
            if stored.get(key) != value:
                raise ValueError(f"Embedding cache version differs in {key} at {self.path}. Build in a new directory.")
        self.writable = create
        self.pid = None
        self.connections = OrderedDict()

    @staticmethod
    def keys(samples):
        return [_key(sample["query"], sample["product"] in GRID_PRODUCTS) for sample in samples]

    def _connection(self, year):
        pid = os.getpid()
        if self.pid != pid:
            self.close()
            self.pid = pid
        if year not in self.connections:
            path = (self.path / f"{year}.sqlite3").resolve()
            if not self.writable and not path.exists():
                return None
            uri = f"{path.as_uri()}?mode={'rwc' if self.writable else 'ro'}"
            connection = sqlite3.connect(uri, uri=True, timeout=60)
            connection.execute("PRAGMA cache_size = -4096")
            if self.writable:
                connection.execute("CREATE TABLE IF NOT EXISTS embeddings (query BLOB PRIMARY KEY, latent BLOB) WITHOUT ROWID")
            self.connections[year] = connection
            if len(self.connections) > 4:
                self.connections.popitem(last=False)[1].close()
        self.connections.move_to_end(year)
        return self.connections[year]

    def lookup(self, keys):
        groups, found = defaultdict(set), {}
        for year, key in keys:
            groups[year].add(key)
        for year, queries in groups.items():
            connection = self._connection(year)
            if connection is None:
                continue
            queries = list(queries)
            for start in range(0, len(queries), 500):
                chunk = queries[start:start + 500]
                placeholders = ",".join("?" for _ in chunk)
                found.update(((year, key), value) for key, value in connection.execute(
                    f"SELECT query, latent FROM embeddings WHERE query IN ({placeholders})", chunk))
        return found

    def read(self, samples):
        keys = self.keys(samples)
        found = self.lookup(keys)
        result = []
        for sample, key in zip(samples, keys):
            if key not in found:
                raise KeyError(f"Missing cached embedding for {sample['query']} in {self.path}. Resume the cache build.")
            raw = found[key]
            if raw is None:
                result.append(None)
                continue
            value = np.frombuffer(raw, dtype=self.dtype)
            if value.size != self.latent_dim or not np.isfinite(value).all():
                raise ValueError(f"Invalid cached embedding for {sample['query']} in {self.path}")
            result.append(torch.from_numpy(value.astype(np.float32)))
        return result

    def write(self, rows):
        groups = defaultdict(list)
        for (year, key), latent in rows.items():
            value = None
            if latent is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    array = latent.detach().float().cpu().numpy().astype(self.dtype, copy=False)
                if array.shape != (self.latent_dim,) or not np.isfinite(array).all():
                    raise ValueError("Cannot store an embedding with the requested shape or precision")
                value = array.tobytes()
            groups[year].append((key, value))
        for year, items in groups.items():
            connection = self._connection(year)
            with connection:
                connection.executemany("INSERT OR IGNORE INTO embeddings VALUES (?, ?)", items)

    def close(self):
        for connection in self.connections.values():
            connection.close()
        self.connections.clear()


def save_embeddings(path, entries, encoder_metadata, *, simulation=False, dtype="float32", gather_mode="history",
                    simulation_daily_start="1979-01-01"):
    config = {"cache_path": path, "cache_dtype": dtype, "gather_mode": gather_mode,
              "simulation_daily_start": simulation_daily_start}
    with closing(EmbeddingCache(config, encoder_metadata, create=True)) as cache:
        rows = {}
        for query, latent in entries:
            rows[_key(query, simulation)] = latent
            if len(rows) >= 4096:
                cache.write(rows)
                rows.clear()
        cache.write(rows)
    return Path(path)


@torch.no_grad()
def build_embedding_cache(config, partitions, encoder, device, batch_size):
    expected = {name: [{key: value for key, value in file.items() if key != "sha256"} for file in files]
                for name, files in encoder.provenance["drivers"].items()}
    chunk_rows = config.get("cache_chunk_rows", 4096)
    gatherer = encoder.make_gatherer()
    collator = MoECollator(gatherer=gatherer, gather_mode=config.get("gather_mode", "history"),
                           simulation_daily_start=config.get("simulation_daily_start", "1979-01-01"))
    totals = {}
    try:
        with closing(EmbeddingCache(config, encoder.provenance, create=True)) as cache:
            for segments, provenance in partitions:
                split = provenance["split"]
                with closing(MoEDatasetAdapter(segments, provenance)) as dataset:
                    sampler = ShardBatchSampler(dataset.shard_lengths, batch_size, shuffle=False, seed=0)
                    traversal = json.dumps(["native_shards_v1", provenance, batch_size], sort_keys=True).encode()
                    cursor = cache.path / f"build-{hashlib.sha256(traversal).hexdigest()[:24]}.json"
                    state = json.loads(cursor.read_text()) if cursor.is_file() else {"batches": 0, "encoded": 0}
                    sampler.set_progress(0, state["batches"])
                    rows, processed = {}, 0

                    def commit():
                        if driver_inventory(encoder.registry, hash_small=False) != expected:
                            raise ValueError("Environmental driver files changed during the cache build")
                        cache.write(rows)
                        _write_json(cursor, state)
                        print(json.dumps({"event": "cache_progress", "split": split, **state,
                                          "total_batches": sampler.total_batches}), flush=True)
                        rows.clear()

                    for indices in sampler:
                        samples = [dataset[index] for index in indices]
                        keys = cache.keys(samples)
                        found = cache.lookup(keys)
                        missing = {key: sample for key, sample in zip(keys, samples)
                                   if key not in found and key not in rows}
                        contexts = {}
                        for key, sample in missing.items():
                            context = collator.gather_context(sample)
                            if context is not None:
                                contexts[key] = context
                            else:
                                rows[key] = None
                        if contexts:
                            inputs = collator.environmental_collator.collate_contexts(list(contexts.values()))
                            latents = encoder(to_device(inputs, device)).float().cpu()
                            rows.update(zip(contexts, latents))
                            state["encoded"] += len(contexts)
                        state["batches"] += 1
                        processed += len(samples)
                        if processed >= chunk_rows:
                            commit()
                            processed = 0
                    if processed:
                        commit()
                    totals[split] = state
    finally:
        for source in gatherer.sources.values():
            source.close()
    return {"cache_path": str(config["cache_path"]), "splits": totals}
