from glob import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
from geopandas import GeoSeries
from shapely.geometry import box
from scipy.ndimage import gaussian_filter


def create_s2_mosaic(s2_filepath, bathymetry_filepath, bands, scale=10000):
    s2_file_list = sorted(glob(s2_filepath))
    s2_raster_list = [rasterio.open(file) for file in s2_file_list]
    bathymetry_raster = rasterio.open(bathymetry_filepath)

    intersecting_rasters = [
        s2_raster_list[idx]
        for idx in intersecting_tiles(s2_raster_list, bathymetry_raster)
    ]

    mosaic, transform = merge(
        intersecting_rasters,
        indexes=bands,
        dtype=np.float32
    )

    mosaic, transform = change_crs(
        mosaic,
        s2_raster_list[0].crs,
        transform,
        bathymetry_raster.crs
    )

    mosaic /= scale

    return mosaic, transform


def return_s2_projected_depth(bathymetry_filepath, s2_transform, s2_shape):
    bathymetry_raster = rasterio.open(bathymetry_filepath)

    depth, _ = project_file(
        bathymetry_raster,
        bathymetry_raster.crs,
        s2_transform,
        s2_shape
        )

    return depth


def intersecting_tiles(obj_lst, raster_obj):
    """
    List indices in a list of rasterio objects that intersect
    with a single raster_obj
    """
    ref = box(*raster_obj.bounds)

    bboxes = GeoSeries(
        [box(*obj.bounds) for obj in obj_lst],
        crs=obj_lst[0].crs,
    )

    return [
        idx
        for idx, bbox in enumerate(bboxes.to_crs(raster_obj.crs))
        if ref.intersects(bbox)
    ]


def project_file(raster_file, dst_crs, dst_transform, dst_shape):
    bd = raster_file.read(1)
    bd[bd < -1e6] = 13

    result, tfm = reproject(
        bd,
        np.zeros((1, dst_shape[-2], dst_shape[-1]), dtype=np.float32),
        src_transform=raster_file.transform,
        src_crs=raster_file.crs,
        src_nodata=13,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_resolution=10,
        resampling=Resampling.bilinear,
    )
    return result, tfm


def change_crs(data, src_crs, src_transform, dst_crs):
    data_bounds = array_bounds(data.shape[1], data.shape[2], src_transform)

    new_transform, width, height = calculate_default_transform(
        src_crs,
        dst_crs,
        data.shape[1],
        data.shape[2],
        *data_bounds,
        resolution=10
    )

    result, tfm = reproject(
        data,
        np.zeros((4, width, height), dtype=np.float32),
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=None,
        dst_transform=new_transform,
        dst_crs=dst_crs,
        dst_resolution=10,
        resampling=Resampling.bilinear,
    )

    return result, tfm


def raw_features(data):
    blue = data[0].ravel()
    green = data[1].ravel()
    red = data[2].ravel()
    nir = data[3].ravel()  # Actually nearest red-edge band
    blue2 = gaussian_filter(data[0], 2.).ravel()
    green2 = gaussian_filter(data[1], 2.).ravel()
    red2 = gaussian_filter(data[2], 2.).ravel()
    nir2 = gaussian_filter(data[3], 2.).ravel()
    return np.vstack((blue, green, red, nir, blue2, green2, red2, nir2)).T


def create_training_data(s2_mosaic, depth_map):
    # Find no_data values; mask is true where there is valid data
    mask = depth_map != 13
    # Keep only values with depth data
    X = raw_features(s2_mosaic)[mask.ravel()]
    # Flip the depth to positive values
    y = abs(depth_map)[mask].copy()

    return X, y
