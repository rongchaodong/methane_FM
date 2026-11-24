import netCDF4
import numpy as np
import pytest
import rasterio
from rasterio.warp import transform

from encoding import DataSourceSpec, DriverRegistry, DriverSpec, QuerySpec, build_data_sources


def query(lat, lon, height=0, width=0):
    return QuerySpec(lat, lon, "2020-07-15", "2020-07-16", height, width)


def source(path, resolution, geotiff=False):
    spec = DataSourceSpec("test", resolution, "static", () if geotiff else ("lat", "lon"),
                          "geotiff" if geotiff else "netcdf")
    registry = DriverRegistry([DriverSpec("value", "test", str(path))], [spec])
    return build_data_sources(registry)["test"]


def read(reader, q, mode="intersects", budget=None):
    return reader.read_window(q, q.start_time, q.end_time, mode, budget)


def test_product_limit_is_per_axis_and_inclusive():
    for height, width in ((0, 0), (2, 2), (.01, 2), (2, .01)):
        assert query(90, 180, height, width).lon == -180
    for height, width in ((2.001, .01), (.01, 2.001), (2, 3)):
        with pytest.raises(ValueError, match="2 degrees on each axis"):
            query(0, 0, height, width)


def test_netcdf_intersection_dateline_and_pole(tmp_path):
    path = tmp_path / "grid.nc"
    with netCDF4.Dataset(path, "w") as ds:
        for name, values in [("lat", [90, 89.75, 89.5]), ("lon", [179.5, 179.75, 180, 180.25, 180.5])]:
            ds.createDimension(name, len(values))
            ds.createVariable(name, "f8", (name,))[:] = values
        ds.createVariable("value", "f4", ("lat", "lon"))[:] = np.arange(15).reshape(3, 5)
    reader = source(path, (.25, .25))
    try:
        area = read(reader, query(89.9, 179.95, .5, .65))
        assert area.values[:, 0].tolist() == [1, 2, 3, 6, 7, 8]
        np.testing.assert_array_equal(area.values, read(reader, query(89.9, -180.05, .5, .65)).values)
        np.testing.assert_array_equal(area.metadata[:3, 0], [89.9375] * 3)
        np.testing.assert_array_equal(area.metadata[:3, 2], [.125] * 3)
        assert read(reader, query(90, -180)).values.item() == 2
        assert read(reader, query(89.875, 179.875)).values.item() == 2
        assert read(reader, query(89.875, 179.875, .002, .002)).values[:, 0].tolist() == [1, 2, 6, 7]
    finally:
        reader.close()


def test_declared_netcdf_bounds(tmp_path):
    path = tmp_path / "bounds.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("lat", 1)
        ds.createDimension("lon", 2)
        ds.createDimension("nv", 2)
        ds.createVariable("lat", "f8", ("lat",))[:] = [0]
        lon = ds.createVariable("lon", "f8", ("lon",))
        lon[:] = [179.75, -180]
        lon.bounds = "lon_bounds"
        ds.createVariable(lon.bounds, "f8", ("nv", "lon"))[:] = [[179.6, 179.9], [179.9, -179.85]]
        ds.createVariable("value", "f4", ("lat", "lon"))[:] = [[10, 20]]
    reader = source(path, (.25, .25))
    try:
        assert read(reader, query(0, 179.89)).values.item() == 10
        assert read(reader, query(0, 179.9, .01, .02)).values[:, 0].tolist() == [10, 20]
    finally:
        reader.close()


def indexed_igh_tile(tmp_path, row, col, height, width):
    # Same affine as the 5 km product; values identify global native rows and columns.
    path = tmp_path / "soil.tif"
    affine = rasterio.Affine(5000, 0, -19949750 + 5000 * col, 0, -5000, 8361000 - 5000 * row)
    rows, cols = np.indices((height, width))
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1,
                       crs="ESRI:54052", transform=affine, dtype="int32") as ds:
        ds.write(((rows + row) * 8000 + cols + col).astype("int32"), 1)
    return source(path, (5000, 5000), geotiff=True)


def test_igh_native_cells_and_tiny_rectangles(tmp_path):
    reader = indexed_igh_tile(tmp_path, 755, 1855, 31, 34)
    try:
        q = query(40.5, -93.5, 1, 1)
        area = read(reader, q)
        assert len(area) == len(set(area.values[:, 0])) == 414
        assert 775 * 8000 + 1882 in area.values
        assert len(read(reader, q, "centers")) == 388
        p = (40.52300246663162, -93.53337605720688)
        assert read(reader, query(*p)).values.item() == 770 * 8000 + 1873
        assert set(read(reader, query(*p, .002, .002)).values.ravel()) == {
            r * 8000 + c for r in (769, 770) for c in (1872, 1873)}
        assert len(read(reader, query(*p, .002, .25))) == 12
        with pytest.raises(ValueError, match="max_tokens"):
            read(reader, q, budget=400)
    finally:
        reader.close()


def test_igh_cut_pixel_is_unique_and_query_independent(tmp_path):
    reader = indexed_igh_tile(tmp_path, 256, 830, 15, 5570)
    try:
        area = read(reader, query(65.5, 179.95, .45, .65))
        ids = area.values[:, 0]
        assert len(ids) == len(set(ids)) == 114
        assert np.isfinite(area.metadata).all() and (area.metadata[:, 2:4] > 0).all()
        x, y = transform("EPSG:4326", "ESRI:54052", area.metadata[:, 1], area.metadata[:, 0])
        native_ids = np.floor((8361000 - np.asarray(y)) / 5000) * 8000 + np.floor((np.asarray(x) + 19949750) / 5000)
        np.testing.assert_array_equal(ids, native_ids)
        point = read(reader, query(65.72144255839295, 179.90835374048444))
        assert point.values.item() == 258 * 8000 + 6373
        np.testing.assert_array_equal(point.metadata, area.metadata[ids == point.values.item()])
    finally:
        reader.close()


def test_generic_projected_touch_and_crossing(tmp_path):
    path = tmp_path / "mercator.tif"
    with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=1, crs="EPSG:3857",
                       transform=rasterio.Affine(5000, 0, -10000, 0, -5000, 10000), dtype="int16") as ds:
        ds.write(np.arange(16, dtype="int16").reshape(4, 4), 1)
    reader = source(path, (5000, 5000), geotiff=True)
    try:
        east, north = transform("EPSG:3857", "EPSG:4326", [5000], [5000])
        q = query(north[0] / 2, east[0] / 2, north[0], east[0])
        assert read(reader, q).values.item() == 6
        assert read(reader, query(0, 0, .0001, .0001)).values[:, 0].tolist() == [5, 6, 9, 10]
    finally:
        reader.close()
