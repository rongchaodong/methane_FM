"""Native pixel coverage for small geographic rectangles on north-up rasters."""

import numpy as np
from rasterio.warp import transform
from rasterio.windows import Window

from .specs import normalize_longitude


WGS84 = "EPSG:4326"
# Western edge, central meridian and eastern edge in PROJ's land-view IGH.
_NORTH = ((-180, -100, -40), (-40, 30, 180))
_SOUTH = ((-180, -160, -100), (-100, -60, -20), (-20, 20, 80), (80, 140, 180))


def _igh_parallels(crs, meridian, y):
    """Each IGH parallel is linear in longitude within one lobe."""
    x0 = transform(WGS84, crs, [meridian], [0])[0][0]
    lat = np.asarray(transform(crs, WGS84, np.full(len(y), x0), y)[1])
    x = transform(WGS84, crs, np.full(len(y), meridian + 10), lat)[0]
    return lat, x0, (np.asarray(x) - x0) / 10


def _igh_outline(dataset, meridian, west, south, east, north, max_candidates):
    # Within a hemisphere each meridian is monotone in x. Its extrema in a
    # native row occur at the row edges or the query limits, including at a cut.
    a = dataset.transform
    bottom, top = transform(WGS84, dataset.crs, [meridian, meridian], [south, north])[1]
    bottom, top = max(bottom, dataset.bounds.bottom), min(top, dataset.bounds.top)
    if bottom >= top:
        return np.empty((0, 2))
    first = int(np.floor((top - a.f) / a.e)) + 1
    stop = int(np.ceil((bottom - a.f) / a.e))
    # An over-height window needs only its bounding box for the budget check.
    rows = np.arange(first, stop) if stop - first <= max_candidates + 1 else np.empty(0)
    y = np.r_[top, a.f + a.e * rows, bottom]
    _, x0, scale = _igh_parallels(dataset.crs, meridian, y)
    left, right = x0 + (west - meridian) * scale, x0 + (east - meridian) * scale
    return np.column_stack((np.r_[left, right[::-1]], np.r_[y, y[::-1]]))


def _projected_outlines(dataset, query, max_candidates):
    projection = dataset.crs.to_dict()
    igh = projection.get("proj") == "igh"
    if projection.get("proj") == "igh_o":
        raise ValueError("Ocean-view Homolosine area queries require a separate lobe adapter")
    origin = projection.get("lon_0", 0) if igh else 0
    center = float(normalize_longitude(query.lon - origin))
    south, north = query.latitude_bounds
    for shift in (-360, 0, 360):
        west = max(-180, center + shift - query.delta_lon / 2)
        east = min(180, center + shift + query.delta_lon / 2)
        if west >= east:
            continue
        if igh:
            for low, high, lobes in ((south, min(0, north), _SOUTH), (max(0, south), north, _NORTH)):
                if low >= high:
                    continue
                for left, meridian, right in lobes:
                    w, e = max(west, left), min(east, right)
                    if w < e:
                        yield _igh_outline(dataset, origin + meridian, origin + w, low, origin + e, high, max_candidates)
        else:
            # Geographic parallels/meridians can be curved on other projected grids.
            lat, lon = np.linspace(south, north, 65), np.linspace(west, east, 65)
            x, y = transform(WGS84, dataset.crs,
                             np.r_[lon, np.full_like(lat, east), lon[::-1], np.full_like(lat, west)],
                             np.r_[np.full_like(lon, south), lat, np.full_like(lon, north), lat[::-1]])
            yield np.column_stack((x, y))


def snap_pixel_edges(values):
    # Remove affine/CRS roundoff on an exact grid line, in native pixel units.
    nearest = np.rint(values)
    return np.where(np.abs(values - nearest) <= 1e-10, nearest, values)


def pixel_xy(affine, col, row):
    return affine.c + affine.a * col, affine.f + affine.e * row


def covered_windows(dataset, query, limit):
    """Yield a read window and a mask of pixels with positive intersection area."""
    a = dataset.transform
    max_candidates = max(1024, limit * 4)
    for polygon in _projected_outlines(dataset, query, max_candidates):
        if not len(polygon):
            continue
        if not np.isfinite(polygon).all():
            raise ValueError("Query cannot be projected to a finite native window")
        x, y = snap_pixel_edges((polygon - [a.c, a.f]) / [a.a, a.e]).T
        c0, c1 = max(0, int(np.floor(x.min()))), min(dataset.width, int(np.ceil(x.max())))
        r0, r1 = max(0, int(np.floor(y.min()))), min(dataset.height, int(np.ceil(y.max())))
        if c1 <= c0 or r1 <= r0:
            continue
        if (r1 - r0) * (c1 - c0) > max_candidates:
            raise ValueError("Projected candidate window is too large; reduce the query or increase max_tokens")
        mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        # Between two consecutive vertex/row heights every interior interval is
        # bounded by the same pair of linear edges. Pure edge touches have zero width.
        levels = np.union1d(np.arange(r0, r1 + 1), np.clip(y, r0, r1))
        next_y = np.roll(y, -1)
        dy = next_y - y
        slope = np.divide(np.roll(x, -1) - x, dy, out=np.zeros_like(x), where=dy != 0)
        for lower, upper in zip(levels[:-1], levels[1:]):
            active = (np.minimum(y, next_y) <= lower) & (np.maximum(y, next_y) >= upper)
            x_low = x[active] + slope[active] * (lower - y[active])
            x_high = x[active] + slope[active] * (upper - y[active])
            order = np.argsort(x_low + x_high)
            starts = np.floor(snap_pixel_edges(np.minimum(x_low, x_high)[order[::2]])).astype(int)
            stops = np.ceil(snap_pixel_edges(np.maximum(x_low, x_high)[order[1::2]])).astype(int)
            for start, stop in zip(starts, stops):
                start, stop = max(c0, start), min(c1, stop)
                if start < stop:
                    mask[int(np.floor(lower)) - r0, start - c0:stop - c0] = True
        yield Window(c0, r0, c1 - c0, r1 - r0), mask


def _clip_meridian(polygon, edge, keep_east):
    result = []
    for (lon0, lat0), (lon1, lat1) in zip(np.roll(polygon, 1, axis=0), polygon):
        inside0 = lon0 >= edge if keep_east else lon0 <= edge
        inside1 = lon1 >= edge if keep_east else lon1 <= edge
        if inside0 != inside1:
            result.append((edge, lat0 + (lat1 - lat0) * (edge - lon0) / (lon1 - lon0)))
        if inside1:
            result.append((lon1, lat1))
    return np.asarray(result).reshape(-1, 2)


def igh_edge_support(dataset, row, col, center_lon, center_lat):
    """Describe the full valid part of a cut pixel independently of the query."""
    left, top = pixel_xy(dataset.transform, col, row)
    right, bottom = pixel_xy(dataset.transform, col + 1, row + 1)
    crs = dataset.crs
    origin = crs.to_dict().get("lon_0", 0)
    equator = transform(WGS84, crs, [origin], [0])[1][0]
    pieces = []
    for low, high, lobes in ((bottom, min(top, equator), _SOUTH), (max(bottom, equator), top, _NORTH)):
        if low >= high:
            continue
        for west, meridian, east in lobes:
            lat, x0, scale = _igh_parallels(crs, origin + meridian, np.linspace(low, high, 17))
            w, e = meridian + (left - x0) / scale, meridian + (right - x0) / scale
            if e.max() <= west or w.min() >= east:
                continue
            polygon = np.column_stack((np.r_[w, e[::-1]], np.r_[lat, lat[::-1]]))
            polygon = _clip_meridian(_clip_meridian(polygon, west, True), east, False)
            if len(polygon) < 3:
                continue
            # Mid-latitude of the valid fragment gives a representative point
            # inside the native pixel even when its nominal center is in a gap.
            low_lat, high_lat = polygon[:, 1].min(), polygon[:, 1].max()
            mid_lat = (low_lat + high_lat) / 2
            x10 = transform(WGS84, crs, [origin + meridian + 10], [mid_lat])[0][0]
            k = (x10 - x0) / 10
            lo, hi = max(west, meridian + (left - x0) / k), min(east, meridian + (right - x0) / k)
            if hi > lo and high_lat > low_lat:
                polygon[:, 0] += origin
                pieces.append(((hi - lo) * (high_lat - low_lat), origin + (lo + hi) / 2, mid_lat, polygon))
    if not pieces:
        raise ValueError(f"No geographic footprint for IGH pixel ({row}, {col})")
    if not np.isfinite(center_lon):
        _, center_lon, center_lat, _ = max(pieces, key=lambda piece: piece[0])
    lon, lat = np.concatenate([piece[3] for piece in pieces]).T
    return (center_lat, float(normalize_longitude(center_lon)), 2 * np.max(np.abs(lat - center_lat)),
            2 * np.max(np.abs(normalize_longitude(lon - center_lon))))
