"""Read local native-grid windows directly from NetCDF and annual global tables."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

from .specs import DEFAULT_MAX_TOKENS, TokenBlock, normalize_longitude, time_metadata


def _empty(source, drivers):
    return TokenBlock(source, tuple(driver.name for driver in drivers),
                      np.empty((0, len(drivers)), dtype=np.float32),
                      np.empty((0, len(drivers)), dtype=bool),
                      np.empty((0, 9), dtype=np.float32), np.empty(0, dtype=bool))


def _axis_bounds(dataset, name, centers, spacing, longitude):
    coordinate = dataset[name]
    if hasattr(coordinate, "bounds"):
        var = dataset[coordinate.bounds]
        edges = np.moveaxis(np.asarray(var[:], dtype=float), var.dimensions.index(coordinate.dimensions[0]), 0)
        if edges.shape != (len(centers), 2) or not np.isfinite(edges).all():
            raise ValueError(f"{name}: expected two finite cell bounds per coordinate")
        if longitude:
            edges += 360 * np.rint((centers[:, None] - edges) / 360)
        bounds = np.sort(edges, axis=1)
    else:
        # Use coordinate midpoints. At an open edge of a regional grid the
        # registered spacing supplies the missing neighbour.
        bounds = centers[:, None] + np.array([-0.5, 0.5]) * spacing
        order = np.argsort(centers)
        lower, upper = order[:-1], order[1:]
        gaps = centers[upper] - centers[lower]
        adjacent = np.isclose(gaps, spacing, rtol=0, atol=1e-6)
        mid = centers[lower[adjacent]] + gaps[adjacent] / 2
        bounds[lower[adjacent], 1] = bounds[upper[adjacent], 0] = mid
        if longitude and len(order) > 1:
            first, last = order[0], order[-1]
            gap = (centers[first] + 360) - centers[last]
            if np.isclose(gap, spacing, rtol=0, atol=1e-6):
                bounds[last, 1] = centers[last] + gap / 2
                bounds[first, 0] = bounds[last, 1] - 360
    return bounds if longitude else np.clip(bounds, -90, 90)


def _axis_indices(coordinates, bounds, center, width, spacing, selection, longitude=False):
    if longitude:
        shift = 360 * np.rint((center - coordinates) / 360)
        coordinates = coordinates + shift
        bounds = bounds + shift[:, None]
    lower, upper = center - width / 2, center + width / 2
    if selection == "centers":  # Historical checkpoint sampling.
        if width <= spacing:
            offsets = coordinates - center
            index = np.lexsort((-offsets, np.abs(offsets)))[0]
            return np.asarray([index]) if abs(offsets[index]) <= spacing / 2 + 1e-8 else np.empty(0, int)
        if not longitude:
            upper = min(90, upper)
        return np.flatnonzero((coordinates >= lower) & (coordinates < upper))
    west, east = bounds.T
    if width == 0:
        # The higher-coordinate cell owns an internal edge. Close the outermost edge.
        inside = np.flatnonzero((west <= center) & (center < east))
        return inside[:1] if len(inside) else np.flatnonzero(east == center)[:1]
    return np.flatnonzero((east > lower) & (west < upper))


def _periods(start, end, frequency):
    if frequency == "static":
        return [None], [None]
    if frequency == "daily":
        first = start.replace(hour=0, minute=0, second=0, microsecond=0)
        first += timedelta(days=int(first < start))
        count = max(0, (end - first).days)
        starts = [first + timedelta(days=index) for index in range(count)]
        return starts, [value + timedelta(days=1) for value in starts]
    first_year = start.year + int(start > datetime(start.year, 1, 1))
    starts = [datetime(year, 1, 1) for year in range(first_year, end.year)]
    return starts, [datetime(value.year + 1, 1, 1) for value in starts]


class BaseEnvironmentalDataSource(ABC):
    def __init__(self, spec, registry):
        self.spec = spec
        self.drivers = registry.drivers_for_source(spec.name)
        if not self.drivers:
            raise ValueError(f"{spec.name}: no registered drivers")

    @abstractmethod
    def read_window(self, query, start_time, end_time, spatial_selection="intersects", max_tokens=None):
        """Return native values without spatial interpolation or subsampling."""

    def close(self):
        pass


class NetCDFDataSource(BaseEnvironmentalDataSource):
    """Cache a few file handles; only selected grid cells are read into memory."""

    def __init__(self, spec, registry):
        super().__init__(spec, registry)
        self._files = OrderedDict()
        self._grid = None

    def _open(self, path):
        if path in self._files:
            self._files.move_to_end(path)
            return self._files[path]
        if not Path(path).is_file():
            raise FileNotFoundError(f"{self.spec.name}: required data file does not exist: {path}")
        dataset = netCDF4.Dataset(path)
        try:
            lat_name, lon_name = self.spec.coordinates[:2]
            lat = np.asarray(dataset[lat_name][:], dtype=float)
            lon = normalize_longitude(np.asarray(dataset[lon_name][:], dtype=float))
            if (lat.ndim != 1 or lon.ndim != 1 or not np.isfinite(lat).all()
                    or not np.isfinite(lon).all() or np.any(np.abs(lat) > 90)):
                raise ValueError(f"{path}: finite one-dimensional latitude/longitude coordinates required")
            for index, (axis, spacing) in enumerate(zip((lat, lon), self.spec.resolution)):
                differences = np.diff(np.sort(axis))
                if index == 1 and len(axis) > 1:
                    differences = np.diff(np.r_[np.sort(axis), np.min(axis) + 360])
                    irregular = ~np.isclose(differences, spacing, rtol=0, atol=1e-6)
                    regular = np.count_nonzero(irregular) <= 1 and np.all(differences >= spacing - 1e-6)
                else:
                    regular = np.allclose(differences, spacing, rtol=0, atol=1e-6)
                if not regular:
                    raise ValueError(f"{path}: coordinate spacing does not match configured resolution")
            lat_bounds = _axis_bounds(dataset, lat_name, lat, self.spec.resolution[0], False)
            lon_bounds = _axis_bounds(dataset, lon_name, lon, self.spec.resolution[1], True)
            grid = lat, lon, lat_bounds, lon_bounds
            if self._grid is None:
                self._grid = grid
            elif not all(np.array_equal(old, new) for old, new in zip(self._grid, grid)):
                raise ValueError(f"{path}: drivers in one source must share coordinate arrays; separate shifted grids")
            entry = {"dataset": dataset, "lat": lat, "lon": lon, "lat_bounds": lat_bounds, "lon_bounds": lon_bounds}
            if len(self.spec.coordinates) == 3:
                time = dataset[self.spec.coordinates[2]]
                if self.spec.frequency == "annual" and not hasattr(time, "units"):
                    years = np.asarray(time[:])
                    if np.any(years != np.floor(years)):
                        raise ValueError(f"{path}: annual year coordinates must be integers")
                    dates = [datetime(int(year), 1, 1) for year in years]
                else:
                    calendar = getattr(time, "calendar", "standard")
                    if calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
                        raise ValueError(f"{path}: only Gregorian calendars are supported")
                    dates = list(netCDF4.num2date(time[:], time.units, calendar=calendar,
                                                 only_use_cftime_datetimes=False,
                                                 only_use_python_datetimes=True))
                if any(value.time() != datetime.min.time() for value in dates):
                    raise ValueError(f"{path}: daily and annual labels must start at UTC midnight")
                if self.spec.frequency == "annual" and any((value.month, value.day) != (1, 1) for value in dates):
                    raise ValueError(f"{path}: annual labels must start on January 1")
                lookup = {value: index for index, value in enumerate(dates)}
                if len(lookup) != len(dates):
                    raise ValueError(f"{path}: duplicate time coordinates")
                entry["time_lookup"] = lookup
        except Exception:
            dataset.close()
            raise
        self._files[path] = entry
        if len(self._files) > 16:
            self._files.popitem(last=False)[1]["dataset"].close()
        return entry

    def _path_groups(self, driver, starts):
        if not driver.path:
            raise ValueError(f"{driver.name}: a data path is required")
        if "{year" not in driver.path:
            if self.spec.frequency == "annual" and len(self.spec.coordinates) == 2:
                raise ValueError(f"{driver.name}: annual files without a time axis require a {{year}} path")
            return [(driver.path, np.arange(len(starts)))]
        if self.spec.frequency == "static":
            raise ValueError(f"{driver.name}: static paths cannot contain {{year}}")
        return [(driver.path.format(year=year), np.asarray([index for index, date in enumerate(starts)
                                                          if date.year == year]))
                for year in sorted({date.year for date in starts})]

    def _read(self, entry, driver, positions, starts, lat_indices, lon_indices):
        dataset = entry["dataset"]
        lat_name, lon_name = self.spec.coordinates[:2]
        lat_dim, lon_dim = dataset[lat_name].dimensions[0], dataset[lon_name].dimensions[0]
        selectors = {lat_dim: lat_indices, lon_dim: lon_indices}
        dimensions = [lat_dim, lon_dim]
        if len(self.spec.coordinates) == 3:
            requested = [starts[index] for index in positions]
            missing = [date.isoformat() for date in requested if date not in entry["time_lookup"]]
            if missing:
                raise ValueError(f"{driver.name}: missing complete {self.spec.frequency} records: {missing[:3]}")
            time_dim = dataset[self.spec.coordinates[2]].dimensions[0]
            selectors[time_dim] = np.asarray([entry["time_lookup"][date] for date in requested])
            dimensions.insert(0, time_dim)
        variable = dataset[driver.variable or driver.name]
        if set(variable.dimensions) != set(dimensions):
            raise ValueError(f"{driver.name}: expected native dimensions {dimensions}, found {variable.dimensions}")
        # netCDF4 uses orthogonal indexing for one-dimensional index arrays.
        values = np.ma.asarray(variable[tuple(selectors[dim] for dim in variable.dimensions)],
                               dtype=np.float32).filled(np.nan)
        values = values.transpose([variable.dimensions.index(dim) for dim in dimensions])
        return values if len(dimensions) == 3 else values[None, ...]

    def read_window(self, query, start_time, end_time, spatial_selection="intersects", max_tokens=None):
        starts, ends = _periods(start_time, end_time, self.spec.frequency)
        if not starts:
            return _empty(self.spec.name, self.drivers)
        first_path = self._path_groups(self.drivers[0], starts)[0][0]
        first = self._open(first_path)
        lat_indices = _axis_indices(first["lat"], first["lat_bounds"], query.lat, query.delta_lat,
                                    self.spec.resolution[0], spatial_selection)
        lon_indices = _axis_indices(first["lon"], first["lon_bounds"], query.lon, query.delta_lon,
                                    self.spec.resolution[1], spatial_selection, longitude=True)
        count = len(starts) * len(lat_indices) * len(lon_indices)
        if not count:
            return _empty(self.spec.name, self.drivers)
        if max_tokens is not None and count > max_tokens:
            raise ValueError(f"{self.spec.name}: native window needs {count} tokens, exceeding remaining max_tokens={max_tokens}; "
                             "increase the budget or reduce the query/history window")
        columns = []
        for driver in self.drivers:
            values = np.empty((len(starts), len(lat_indices), len(lon_indices)), dtype=np.float32)
            for path, positions in self._path_groups(driver, starts):
                values[positions] = self._read(self._open(path), driver, positions, starts, lat_indices, lon_indices)
            columns.append(values.ravel())
        values = np.column_stack(columns)
        valid = np.isfinite(values)
        for index, driver in enumerate(self.drivers):
            if driver.categories is not None:
                valid[:, index] &= ((values[:, index] >= 0) & (values[:, index] < driver.categories)
                                    & (values[:, index] == np.floor(values[:, index])))
        ti, yi, xi = np.meshgrid(np.arange(len(starts)), lat_indices, lon_indices, indexing="ij")
        static = self.spec.frequency == "static"
        temporal = np.asarray([[0, 0, 0, 0, 365]], dtype=np.float32) if static else time_metadata(starts, ends)
        lat, lon = first["lat"][yi.ravel()], first["lon"][xi.ravel()]
        dy, dx = (np.full(count, width) for width in self.spec.resolution)
        if spatial_selection == "intersects":
            south, north = first["lat_bounds"][yi.ravel()].T
            west, east = first["lon_bounds"][xi.ravel()].T
            lat, lon = (south + north) / 2, normalize_longitude((west + east) / 2)
            dy, dx = north - south, east - west
        metadata = np.column_stack((lat, lon, dy, dx, temporal[ti.ravel()])).astype(np.float32)
        return TokenBlock(self.spec.name, tuple(driver.name for driver in self.drivers), values, valid,
                          metadata, np.full(count, static, dtype=bool))

    def close(self):
        for entry in self._files.values():
            entry["dataset"].close()
        self._files.clear()


class AnnualGlobalDataSource(BaseEnvironmentalDataSource):
    """Use the latest completed calendar year as a causal global context covariate."""
    def __init__(self, spec, registry):
        super().__init__(spec, registry)
        self.tables = []
        for driver in self.drivers:
            data = np.loadtxt(driver.path, ndmin=2)
            if data.shape[1] != 2 or not np.isfinite(data[:, 0]).all() or np.any(data[:, 0] != np.floor(data[:, 0])):
                raise ValueError(f"{driver.path}: expected two columns containing integer year and value")
            table = {int(year): float(value) for year, value in data}
            if len(table) != len(data):
                raise ValueError(f"{driver.path}: duplicate annual years")
            self.tables.append(table)

    def read_window(self, query, start_time, end_time, spatial_selection="intersects", max_tokens=None):
        common_years = set.intersection(*(set(table) for table in self.tables))
        available = [year for year in common_years if datetime(year + 1, 1, 1) <= end_time]
        if not available:
            return _empty(self.spec.name, self.drivers)
        if max_tokens is not None and max_tokens < 1:
            raise ValueError(f"{self.spec.name}: max_tokens has no room for the annual global token")
        year = max(available)
        values = np.asarray([[table[year] for table in self.tables]], dtype=np.float32)
        metadata = np.concatenate(([0, 0, 180, 360], time_metadata([datetime(year, 1, 1)],
                                                                 [datetime(year + 1, 1, 1)])[0]))[None].astype(np.float32)
        return TokenBlock(self.spec.name, tuple(driver.name for driver in self.drivers), values,
                          np.isfinite(values), metadata, np.zeros(1, dtype=bool))


class GeoTIFFDataSource(BaseEnvironmentalDataSource):
    """Read native projected pixels and describe their local WGS84 envelopes."""

    def __init__(self, spec, registry):
        super().__init__(spec, registry)
        import rasterio
        from rasterio.warp import transform
        from rasterio.windows import Window
        self.rasterio, self.transform = rasterio, transform
        self.Window = Window
        self._files = {}

    def _open(self, path):
        if path not in self._files:
            dataset = self.rasterio.open(path)
            try:
                affine = dataset.transform
                if (dataset.count != 1 or dataset.crs is None or not dataset.crs.is_projected
                        or dataset.crs.linear_units_factor[1] != 1
                        or affine.b or affine.d or affine.a <= 0 or affine.e >= 0):
                    raise ValueError(f"{path}: a single-band north-up raster with a projected metre CRS is required")
                if not np.allclose((dataset.res[1], dataset.res[0]), self.spec.resolution):
                    raise ValueError(f"{path}: native pixel size does not match source resolution in metres")
                if self._files:
                    reference = next(iter(self._files.values()))
                    if (dataset.crs != reference.crs or dataset.transform != reference.transform
                            or dataset.shape != reference.shape):
                        raise ValueError(f"{path}: drivers in one GeoTIFF source must share CRS, affine, and shape")
            except Exception:
                dataset.close()
                raise
            self._files[path] = dataset
        return self._files[path]

    def _inverse(self, dataset, x, y):
        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        try:
            lon, lat = self.transform(dataset.crs, "EPSG:4326", x.tolist(), y.tolist())
        except Exception as error:
            if "projection domain" not in str(error).lower() and "err = 2050" not in str(error):
                raise
            # GDAL rejects the entire batch when just one pixel lies between lobes.
            if len(x) == 1:
                return np.full(1, np.nan), np.full(1, np.nan)
            middle = len(x) // 2
            first = self._inverse(dataset, x[:middle], y[:middle])
            second = self._inverse(dataset, x[middle:], y[middle:])
            return tuple(np.concatenate((left, right)) for left, right in zip(first, second))
        lon, lat = np.asarray(lon), np.asarray(lat)
        finite = np.isfinite(lon) & np.isfinite(lat)
        canonical = np.zeros_like(finite)
        if finite.any():
            lon[finite] = normalize_longitude(lon[finite])
            fx, fy = self.transform("EPSG:4326", dataset.crs, lon[finite], lat[finite])
            canonical[finite] = np.hypot(fx - x[finite], fy - y[finite]) < 0.01
        lon[~canonical], lat[~canonical] = np.nan, np.nan
        return lon, lat

    def _windows(self, dataset, query, selection, limit):
        from .projected import covered_windows, pixel_xy, snap_pixel_edges

        if query.is_point or selection == "centers":
            x, y = self.transform("EPSG:4326", dataset.crs, [query.lon], [query.lat])
            if not np.isfinite([x[0], y[0]]).all():
                raise ValueError("Query center cannot be projected to a finite native pixel")
            a = dataset.transform
            col, row = np.floor(snap_pixel_edges(np.array([(x[0] - a.c) / a.a, (y[0] - a.f) / a.e]))).astype(int)
            inside = 0 <= row < dataset.height and 0 <= col < dataset.width
            point = query.is_point
            if inside and not point:
                # Reproduce the small-window convention of the original training run.
                px, py = pixel_xy(a, np.array([col, col + 1, col + 1, col]),
                                  np.array([row, row, row + 1, row + 1]))
                lon, lat = self._inverse(dataset, px, py)
                point = (0 < query.delta_lat <= np.ptp(lat)
                         and 0 < query.delta_lon <= np.ptp(normalize_longitude(lon - query.lon)))
            if point:
                if inside:
                    yield self.Window(col, row, 1, 1), np.ones((1, 1), dtype=bool), False
                return
        if not query.delta_lat or not query.delta_lon:
            raise ValueError("GeoTIFF area queries require positive widths on both axes")
        for window, mask in covered_windows(dataset, query, limit):
            yield window, mask, selection == "centers"

    def _spatial_metadata(self, dataset, rows, cols):
        from .projected import igh_edge_support, pixel_xy

        x, y = pixel_xy(dataset.transform, cols + 0.5, rows + 0.5)
        lon, lat = self._inverse(dataset, x, y)
        offsets = np.asarray([(0, 0), (0, .5), (0, 1), (.5, 1), (1, 1), (1, .5), (1, 0), (.5, 0)])
        x, y = pixel_xy(dataset.transform, cols[:, None] + offsets[:, 0], rows[:, None] + offsets[:, 1])
        corner_lon, corner_lat = self._inverse(dataset, x, y)
        dy = 2 * np.max(np.abs(corner_lat.reshape(-1, 8) - lat[:, None]), axis=1)
        dx = 2 * np.max(np.abs(normalize_longitude(corner_lon.reshape(-1, 8) - lon[:, None])), axis=1)
        spatial = np.column_stack((lat, lon, dy, dx))
        boundary = ~np.isfinite(spatial).all(axis=1) | (dx > 180)
        if boundary.any() and dataset.crs.to_dict().get("proj") == "igh":
            for i in np.flatnonzero(boundary):
                spatial[i] = igh_edge_support(dataset, rows[i], cols[i], lon[i], lat[i])
        return spatial

    def read_window(self, query, start_time, end_time, spatial_selection="intersects", max_tokens=None):
        from .projected import pixel_xy

        if spatial_selection not in {"intersects", "centers"}:
            raise ValueError("spatial_selection must be centers or intersects")
        first = self._open(self.drivers[0].path)
        limit = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        pieces, seen = [], set()
        for window, mask, filter_centers in self._windows(first, query, spatial_selection, limit):
            iy, ix = np.nonzero(mask)
            rows, cols = iy + int(window.row_off), ix + int(window.col_off)
            ids = rows * first.width + cols
            keep = np.asarray([index not in seen for index in ids], dtype=bool)
            if filter_centers and keep.any():
                x, y = pixel_xy(first.transform, cols + .5, rows + .5)
                lon, lat = self._inverse(first, x, y)
                south, north = query.latitude_bounds
                offset = normalize_longitude(lon - query.lon)
                keep &= ((lat >= south) & (lat < north)
                         & (offset >= -query.delta_lon / 2) & (offset < query.delta_lon / 2))
            if not keep.any():
                continue
            seen.update(ids[keep])
            if len(seen) > limit:
                raise ValueError(f"{self.spec.name}: native selection exceeds max_tokens={limit}; no pixels were subsampled")
            iy, ix = iy[keep], ix[keep]
            values = np.column_stack([
                self._open(driver.path).read(1, window=window, masked=True)[iy, ix].astype(np.float32).filled(np.nan)
                for driver in self.drivers
            ])
            valid = np.isfinite(values)
            for index, driver in enumerate(self.drivers):
                if driver.categories is not None:
                    valid[:, index] &= ((values[:, index] >= 0) & (values[:, index] < driver.categories)
                                        & (values[:, index] == np.floor(values[:, index])))
            spatial = self._spatial_metadata(first, rows[keep], cols[keep])
            geographic = np.isfinite(spatial).all(1) & (spatial[:, 3] <= 180)
            if np.any(~geographic & valid.any(1)):
                raise ValueError("An observed pixel has no representable geographic footprint")
            spatial = spatial[geographic]
            metadata = np.column_stack((spatial, np.tile([0, 0, 0, 0, 365], (len(spatial), 1)))).astype(np.float32)
            pieces.append(TokenBlock(self.spec.name, tuple(driver.name for driver in self.drivers),
                                     values[geographic], valid[geographic], metadata, np.ones(len(spatial), dtype=bool)))
        if not pieces:
            return _empty(self.spec.name, self.drivers)
        return TokenBlock(self.spec.name, pieces[0].driver_names,
                          *(np.concatenate([getattr(piece, field) for piece in pieces])
                            for field in ("values", "variable_mask", "metadata", "static_mask")))

    def close(self):
        for dataset in self._files.values():
            dataset.close()
        self._files.clear()


def build_data_sources(registry):
    return {name: (GeoTIFFDataSource if spec.format == "geotiff" else
                   AnnualGlobalDataSource if not spec.coordinates else NetCDFDataSource)(spec, registry)
            for name, spec in registry.sources.items()}
