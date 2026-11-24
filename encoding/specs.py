"""Queries, native-grid tokens, and the ordered driver/source registry."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np

EPOCH = datetime(1970, 1, 1)
MAX_QUERY_WIDTH_DEGREES = 2.0
DEFAULT_MAX_TOKENS = 5120


def as_datetime(value):
    """Interpret naive timestamps as UTC and return a naive UTC datetime."""
    if isinstance(value, np.datetime64):
        value = np.datetime_as_string(value, unit="us")
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise TypeError("Time must be an ISO string, datetime, or numpy.datetime64")
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def normalize_longitude(value):
    value = np.asarray(value)
    return np.where((value >= -180) & (value < 180), value, (value + 180.0) % 360.0 - 180.0)


def time_metadata(starts, ends):
    """Return start/end epoch days, year, zero-based day of year, and year length."""
    rows = []
    for start, end in zip(np.atleast_1d(starts), np.atleast_1d(ends)):
        start, end = as_datetime(start), as_datetime(end)
        year_start, year_end = datetime(start.year, 1, 1), datetime(start.year + 1, 1, 1)
        rows.append([(start - EPOCH).total_seconds() / 86400,
                     (end - EPOCH).total_seconds() / 86400, start.year,
                     (start - year_start).total_seconds() / 86400, (year_end - year_start).days])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


@dataclass(frozen=True)
class QuerySpec:
    """Full rectangular spatial widths and a left-closed, right-open UTC interval."""
    lat: float
    lon: float
    start_time: object
    end_time: object
    delta_lat: float = 0.0
    delta_lon: float = 0.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.lat, self.lon, self.delta_lat, self.delta_lon)):
            raise ValueError("Query coordinates and spatial widths must be finite")
        if not -90 <= self.lat <= 90:
            raise ValueError("Query latitude must be in [-90, 90] degrees")
        if not all(0 <= width <= MAX_QUERY_WIDTH_DEGREES for width in (self.delta_lat, self.delta_lon)):
            raise ValueError(f"Query full width must be between 0 and {MAX_QUERY_WIDTH_DEGREES:g} degrees on each axis")
        start, end = as_datetime(self.start_time), as_datetime(self.end_time)
        if end <= start:
            raise ValueError("Query requires start_time < end_time")
        object.__setattr__(self, "lon", float(normalize_longitude(self.lon)))
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)

    @property
    def is_point(self):
        return self.delta_lat == 0 and self.delta_lon == 0

    @property
    def latitude_bounds(self):
        return max(-90, self.lat - self.delta_lat / 2), min(90, self.lat + self.delta_lat / 2)

    def metadata(self):
        return np.concatenate((np.asarray([self.lat, self.lon, self.delta_lat, self.delta_lon],
                                          dtype=np.float32),
                               time_metadata([self.start_time], [self.end_time])[0]))


@dataclass(frozen=True)
class DriverSpec:
    name: str
    source: str
    path: str = ""
    variable: str | None = None
    mean: float = 0.0
    std: float = 1.0
    categories: int | None = None
    units: str = ""

    def __post_init__(self):
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std <= 0:
            raise ValueError(f"{self.name}: normalization requires finite mean and std > 0")
        if self.categories is not None and (not isinstance(self.categories, int) or self.categories < 1):
            raise ValueError(f"{self.name}: categories must be a positive class count")


@dataclass(frozen=True)
class DataSourceSpec:
    name: str
    resolution: tuple
    frequency: str
    coordinates: tuple = ()
    format: str = "netcdf"

    def __post_init__(self):
        resolution, coordinates = tuple(self.resolution), tuple(self.coordinates)
        if len(resolution) != 2 or any(not np.isfinite(v) or v <= 0 for v in resolution):
            raise ValueError(f"{self.name}: resolution must contain two positive spacings")
        if self.frequency not in {"daily", "static", "annual"}:
            raise ValueError(f"{self.name}: frequency must be daily, static, or annual")
        if self.format not in {"netcdf", "geotiff"}:
            raise ValueError(f"{self.name}: format must be netcdf or geotiff")
        if self.format == "geotiff" and (self.frequency != "static" or coordinates):
            raise ValueError(f"{self.name}: GeoTIFF sources require frequency=static and no coordinate axes")
        allowed = (0,) if self.format == "geotiff" else {"daily": (3,), "static": (2,), "annual": (0, 2, 3)}[self.frequency]
        if len(coordinates) not in allowed:
            raise ValueError(f"{self.name}: coordinates must list latitude, longitude, and any time axis")
        if self.format == "netcdf" and not coordinates and resolution != (180, 360):
            raise ValueError(f"{self.name}: global annual tables require resolution [180, 360]")
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "coordinates", coordinates)


class DriverRegistry:
    """Insertion order defines embedding IDs and is preserved in checkpoints."""
    def __init__(self, drivers=(), sources=()):
        self.sources, self.drivers = {}, {}
        for source in sources:
            self.register_source(source)
        for driver in drivers:
            self.register_driver(driver)

    def register_source(self, spec):
        if spec.name in self.sources:
            raise ValueError(f"Duplicate source: {spec.name}")
        self.sources[spec.name] = spec

    def register_driver(self, spec):
        if spec.name in self.drivers or spec.source not in self.sources:
            raise ValueError(f"Duplicate driver or unknown source: {spec.name}/{spec.source}")
        self.drivers[spec.name] = spec

    def driver_id(self, name):
        return list(self.drivers).index(name)

    def source_id(self, name):
        return list(self.sources).index(name)

    def drivers_for_source(self, name):
        return [driver for driver in self.drivers.values() if driver.source == name]

    def to_dict(self):
        sources = []
        for source in self.sources.values():
            drivers = {}
            for driver in self.drivers_for_source(source.name):
                drivers[driver.name] = {"path": driver.path, "variable": driver.variable or driver.name,
                                        "normalization": [driver.mean, driver.std],
                                        "categories": driver.categories, "units": driver.units}
            sources.append({"name": source.name, "resolution": list(source.resolution),
                            "frequency": source.frequency, "coordinates": list(source.coordinates),
                            "format": source.format, "drivers": drivers})
        return {"sources": sources, "driver_order": list(self.drivers)}

    @classmethod
    def from_dict(cls, config):
        registry = cls()
        for item in config["sources"]:
            registry.register_source(DataSourceSpec(**{key: value for key, value in item.items() if key != "drivers"}))
            for name, options in item["drivers"].items():
                options = dict(options)
                mean, std = options.pop("normalization", [0.0, 1.0])
                registry.register_driver(DriverSpec(name, item["name"], mean=mean, std=std, **options))
        if "driver_order" in config:
            registry.drivers = {name: registry.drivers[name] for name in config["driver_order"]}
        return registry

    @classmethod
    def from_file(cls, path):
        with Path(path).open(encoding="utf-8") as stream:
            if Path(path).suffix == ".json":
                config = json.load(stream)
            else:
                import yaml
                config = yaml.safe_load(stream)
        return cls.from_dict(config.get("registry", config))


@dataclass
class TokenBlock:
    source: str
    driver_names: tuple
    values: np.ndarray
    variable_mask: np.ndarray
    metadata: np.ndarray
    static_mask: np.ndarray

    def __len__(self):
        return len(self.values)

    def select(self, indices):
        return TokenBlock(self.source, self.driver_names, self.values[indices],
                          self.variable_mask[indices], self.metadata[indices], self.static_mask[indices])
