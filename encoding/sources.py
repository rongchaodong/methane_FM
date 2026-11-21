"""Read local native-grid windows directly from NetCDF and annual global tables."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

from .specs import TokenBlock, normalize_longitude, time_metadata


def _empty(source, drivers):
    return TokenBlock(source, tuple(driver.name for driver in drivers),
                      np.empty((0, len(drivers)), dtype=np.float32),
                      np.empty((0, len(drivers)), dtype=bool),
                      np.empty((0, 9), dtype=np.float32), np.empty(0, dtype=bool))


def _axis_indices(coordinates, center, width, spacing, selection, longitude=False):
    offsets = ((coordinates - center + 180) % 360 - 180) if longitude else coordinates - center
    # Each narrow axis independently contributes one nearest native cell.
    if width <= spacing:
        if np.min(np.abs(offsets)) > spacing / 2 + 1e-8:
            return np.empty(0, dtype=int)
        index = np.lexsort((-offsets, np.abs(offsets)))[0]
        return np.asarray([index])
    if longitude:
        if width == 360:
            return np.arange(len(coordinates))
        selected = ((offsets >= -width / 2) & (offsets < width / 2)) if selection == "centers" else (
            np.abs(offsets) < (width + spacing) / 2)
    else:
        lower, upper = max(-90, center - width / 2), min(90, center + width / 2)
        selected = ((coordinates >= lower) & (coordinates < upper)) if selection == "centers" else (
            (np.minimum(90, coordinates + spacing / 2) > lower)
            & (np.maximum(-90, coordinates - spacing / 2) < upper))
    return np.flatnonzero(selected)


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
    def read_window(self, query, start_time, end_time, spatial_selection="centers", max_tokens=None):
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
            if self._grid is None:
                self._grid = lat, lon
            elif not all(np.array_equal(old, new) for old, new in zip(self._grid, (lat, lon))):
                raise ValueError(f"{path}: drivers in one source must share coordinate arrays; separate shifted grids")
            entry = {"dataset": dataset, "lat": lat, "lon": lon}
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

    def read_window(self, query, start_time, end_time, spatial_selection="centers", max_tokens=None):
        starts, ends = _periods(start_time, end_time, self.spec.frequency)
        if not starts:
            return _empty(self.spec.name, self.drivers)
        first_path = self._path_groups(self.drivers[0], starts)[0][0]
        first = self._open(first_path)
        lat_indices = _axis_indices(first["lat"], query.lat, query.delta_lat,
                                    self.spec.resolution[0], spatial_selection)
        lon_indices = _axis_indices(first["lon"], query.lon, query.delta_lon,
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
        metadata = np.column_stack((first["lat"][yi.ravel()], first["lon"][xi.ravel()],
                                    np.full(count, self.spec.resolution[0]), np.full(count, self.spec.resolution[1]),
                                    temporal[ti.ravel()])).astype(np.float32)
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

    def read_window(self, query, start_time, end_time, spatial_selection="centers", max_tokens=None):
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
        from rasterio.warp import transform, transform_bounds
        from rasterio.windows import Window
        self.rasterio, self.transform = rasterio, transform
        self.transform_bounds, self.Window = transform_bounds, Window
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

    def _regions(self, query, crs):
        """Split the date line and IGH lobes before projecting local rectangles."""
        left, right = query.lon - query.delta_lon / 2, query.lon + query.delta_lon / 2
        longitude_ranges = [(left, right)]
        if left < -180:
            longitude_ranges = [(left + 360, 180), (-180, right)]
        elif right > 180:
            longitude_ranges = [(left, 180), (-180, right - 360)]
        south, north = query.latitude_bounds
        latitudes = [south] + ([0.0] if south < 0 < north else []) + [north]
        projection = crs.to_dict()
        for bottom, top in zip(latitudes[:-1], latitudes[1:]):
            cuts = []
            if projection.get("proj") == "igh":
                # These are the land-view lobe boundaries used by PROJ's igh.
                native_cuts = [-40] if bottom >= 0 else [-100, -20, 80]
                cuts = [float(normalize_longitude(cut + projection.get("lon_0", 0))) for cut in native_cuts]
            elif projection.get("proj") == "igh_o":
                raise ValueError("Ocean-view Homolosine area queries require a separate lobe adapter")
            for west, east in longitude_ranges:
                boundaries = [west] + sorted(cut for cut in cuts if west < cut < east) + [east]
                for low, high in zip(boundaries[:-1], boundaries[1:]):
                    # Stay on this side of a cut instead of projecting the adjacent lobe.
                    yield np.nextafter(low, high), bottom, np.nextafter(high, low), top

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
        lon[~np.isfinite(lon)], lat[~np.isfinite(lat)] = np.nan, np.nan
        return normalize_longitude(lon), lat

    def _window(self, dataset, bounds):
        left, bottom, right, top = self.transform_bounds("EPSG:4326", dataset.crs, *bounds, densify_pts=21)
        if not np.isfinite([left, bottom, right, top]).all():
            raise ValueError("Query cannot be projected to a finite native window")
        affine = dataset.transform
        col0 = max(0, int(np.floor((left - affine.c) / affine.a)) - 1)
        col1 = min(dataset.width, int(np.ceil((right - affine.c) / affine.a)) + 1)
        row0 = max(0, int(np.floor((top - affine.f) / affine.e)) - 1)
        row1 = min(dataset.height, int(np.ceil((bottom - affine.f) / affine.e)) + 1)
        return self.Window(col0, row0, max(0, col1 - col0), max(0, row1 - row0))

    def read_window(self, query, start_time, end_time, spatial_selection="centers", max_tokens=None):
        if spatial_selection != "centers":
            raise ValueError("GeoTIFF currently supports spatial_selection=centers only")
        first = self._open(self.drivers[0].path)
        limit = 4096 if max_tokens is None else max_tokens
        x, y = self.transform("EPSG:4326", first.crs, [query.lon], [query.lat])
        if not np.isfinite([x[0], y[0]]).all():
            raise ValueError("Query center cannot be projected to a finite native pixel")
        row, col = first.index(x[0], y[0])
        point_window = self.Window(col, row, 1, 1) if 0 <= row < first.height and 0 <= col < first.width else None
        point_selection = query.is_point
        if not point_selection and point_window is not None:
            px, py = first.transform * (np.asarray([col, col + 1, col + 1, col]),
                                        np.asarray([row, row, row + 1, row + 1]))
            plon, plat = self._inverse(first, px, py)
            pixel_lat = np.ptp(plat)
            pixel_lon = np.ptp(normalize_longitude(plon - query.lon))
            point_selection = (query.delta_lat > 0 and query.delta_lon > 0
                               and query.delta_lat <= pixel_lat and query.delta_lon <= pixel_lon)
        if point_selection:
            windows = [point_window] if point_window is not None else []
        else:
            if query.delta_lat == 0 or query.delta_lon == 0:
                raise ValueError("GeoTIFF area queries require positive widths on both axes")
            windows = [self._window(first, bounds) for bounds in self._regions(query, first.crs)]
        pieces, seen = [], set()
        count = 0
        for window in windows:
            candidates = int(window.width * window.height)
            if not candidates:
                continue
            if candidates > max(1024, limit * 4):
                raise ValueError("Projected candidate window is too large; reduce the query or increase max_tokens")
            rows, cols = np.meshgrid(np.arange(int(window.row_off), int(window.row_off + window.height)),
                                     np.arange(int(window.col_off), int(window.col_off + window.width)), indexing="ij")
            rows, cols = rows.ravel(), cols.ravel()
            x, y = first.transform * (cols + 0.5, rows + 0.5)
            lon, lat = self._inverse(first, x, y)
            selected = np.isfinite(lon) & np.isfinite(lat)
            if not point_selection:
                south, north = query.latitude_bounds
                offset = normalize_longitude(lon - query.lon)
                selected &= ((lat >= south) & (lat < north)
                             & (offset >= -query.delta_lon / 2) & (offset < query.delta_lon / 2))
            ids = rows * first.width + cols
            selected &= np.asarray([int(index) not in seen for index in ids])
            positions = np.flatnonzero(selected)
            if not len(positions):
                continue
            # Exclude noncanonical inverse solutions in interrupted-projection overlaps.
            fx, fy = self.transform("EPSG:4326", first.crs, lon[positions].tolist(), lat[positions].tolist())
            positions = positions[np.hypot(np.asarray(fx) - x[positions], np.asarray(fy) - y[positions]) < 0.01]
            count += len(positions)
            if count > limit:
                raise ValueError(f"{self.spec.name}: native selection exceeds max_tokens={limit}; no pixels were subsampled")
            if not len(positions):
                continue
            seen.update(int(index) for index in ids[positions])
            values = np.column_stack([
                self._open(driver.path).read(1, window=window, masked=True).astype(np.float32).filled(np.nan).ravel()[positions]
                for driver in self.drivers
            ])
            valid = np.isfinite(values)
            for index, driver in enumerate(self.drivers):
                if driver.categories is not None:
                    valid[:, index] &= ((values[:, index] >= 0) & (values[:, index] < driver.categories)
                                        & (values[:, index] == np.floor(values[:, index])))
            rows, cols, lat, lon = rows[positions], cols[positions], lat[positions], lon[positions]
            # The metadata support is a local angular envelope, not a reprojected pixel.
            offsets = np.asarray([(0, 0), (0, .5), (0, 1), (.5, 1), (1, 1), (1, .5), (1, 0), (.5, 0)])
            corner_x, corner_y = first.transform * (cols[:, None] + offsets[:, 0], rows[:, None] + offsets[:, 1])
            corner_lon, corner_lat = self._inverse(first, corner_x, corner_y)
            corner_lon, corner_lat = corner_lon.reshape(-1, 8), corner_lat.reshape(-1, 8)
            geographic = np.isfinite(corner_lon).all(1) & np.isfinite(corner_lat).all(1)
            if np.any(~geographic & valid.any(1)):
                raise ValueError("An observed pixel crosses an invalid projection boundary; split or move the query")
            if not geographic.all():
                values, valid, rows, cols, lat, lon = (array[geographic] for array in (values, valid, rows, cols, lat, lon))
                corner_lon, corner_lat = corner_lon[geographic], corner_lat[geographic]
            delta_lat = 2 * np.max(np.abs(corner_lat - lat[:, None]), axis=1)
            delta_lon = 2 * np.max(np.abs(normalize_longitude(corner_lon - lon[:, None])), axis=1)
            if np.any(delta_lon > 180):
                raise ValueError("Pixel angular support crosses a projection interruption")
            metadata = np.column_stack((lat, lon, delta_lat, delta_lon,
                                        np.tile([0, 0, 0, 0, 365], (len(lat), 1)))).astype(np.float32)
            pieces.append(TokenBlock(self.spec.name, tuple(driver.name for driver in self.drivers), values,
                                     valid, metadata, np.ones(len(lat), dtype=bool)))
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
