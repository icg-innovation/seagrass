from glob import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
from geopandas import GeoSeries
from shapely.geometry import box
from scipy.ndimage import gaussian_filter


def create_s2_mosaic(
    s2_filepath,
    bathymetry_filepath,
    bands=None,
    scale=10000
):
    """Generates a Sentinel 2 mosaic for the areas intersecting the input
    bathymetry raster.

    Args:
        s2_filepath (str): Directory containing the Sentinel 2 raster files.
        bathymetry_filepath (str): Filepath to the bathymetry raster
            file.
        bands (list): List of integers corresponding to the desired Sentinel 2
            bands.
        scale (int, optional): Scale factor to obtain the true Sentinel 2
            pixel value. Defaults to 10000.

    Returns:
        A tuple containing the output raster mosaic and the
        affine transform matrix.
    """
    s2_file_list = sorted(glob(s2_filepath))
    s2_raster_list = [rasterio.open(file) for file in s2_file_list]
    bathymetry_raster = rasterio.open(bathymetry_filepath)

    intersecting_rasters = intersecting_tiles(
        s2_raster_list,
        bathymetry_raster
    )

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


def return_s2_projected_depth(
    bathymetry_filepath,
    s2_transform,
    s2_shape,
):
    """Returns depth raster projected onto the Sentinel 2 mosaic.

    Args:
        bathymetry_filepath (str): Filepath to the bathymetry raster
            file.
        s2_transform (numpy.ndarray): Transform matrix generated when creating
            the Sentinel 2 mosaic.
        s2_shape (tuple): Dimensions of the Sentinel 2 mosaic.

    Returns:
        numpy.ndarray: Projected depth raster data.
    """
    bathymetry_raster = rasterio.open(bathymetry_filepath)

    bathymetry_data = bathymetry_raster.read(1)
    bathymetry_data[bathymetry_data < -1e6] = 13

    # Resamples to Sentinel 2 image resolution?
    depth, _ = reproject(
        bathymetry_data,
        np.zeros((1, s2_shape[-2], s2_shape[-1]), dtype=np.float32),
        src_transform=bathymetry_raster.transform,
        src_crs=bathymetry_raster.crs,
        src_nodata=13,
        dst_transform=s2_transform,
        dst_crs=bathymetry_raster.crs,
        dst_resolution=10,
        resampling=Resampling.bilinear,
    )

    return depth


def intersecting_tiles(raster_list, reference_raster):
    """Returns list of raster objects that intersect with an input reference
    raster.

    Args:
        raster_list (list): List of input raster objects.
        reference_raster (rasterio.open object): Reference raster objects.

    Returns:
        list: List of intersecting raster objects
    """
    ref = box(*reference_raster.bounds)

    boundary_boxes = GeoSeries(
        [box(*raster.bounds) for raster in raster_list],
        crs=raster_list[0].crs,
    )

    intersecting_tiles = [
        raster_list[idx]
        for idx, bbox in enumerate(boundary_boxes.to_crs(reference_raster.crs))
        if ref.intersects(bbox)
    ]

    return intersecting_tiles


def change_crs(data, src_crs, src_transform, dst_crs):
    """Convert CRS of input to destination.
    THIS FUNCTION MAY BE MADE OBSOLETE IN FUTURE VERSIONS.

    Args:
        data (np.ndarray): Input raster data.
        src_crs (CRS or dict): CRS of input raster.
        src_transform (Affine): Transform matrix of input.
        dst_crs (CRS or dict): Target CRS.

    Returns:
        np.ndarray: Raster reprojected to new coordinate system.
    """
    data_bounds = array_bounds(data.shape[1], data.shape[2], src_transform)

    new_transform, width, height = calculate_default_transform(
        src_crs,
        dst_crs,
        data.shape[1],
        data.shape[2],
        *data_bounds,
        resolution=10
    )

    reprojected, transform = reproject(
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

    return reprojected, transform


def return_features(data, bands):
    """Returns a array of default features for the training data.

    Args:
        data (np.ndarray): Input raster data.
        bands (list): List of indices corresponding to the desired raster
        bands. WARNING: If using the bands attr when creating an Sentinel 2
        mosaic, these indices may differ as order is reset when creating a
        mosaic, e.g. [1,2,4,6] --> [0,1,2,3].

    Returns:
        np.ndarray: Reshaped feature data.
    """
    bands_1D = [data[band].ravel() for band in bands]
    blurred_1D = [gaussian_filter(data[band], 2.).ravel() for band in bands]

    return np.vstack(
        tuple(
            *bands_1D,
            *blurred_1D,
        )
    ).T


def create_training_data(
    s2_data,
    bathymetry_data,
    bands,
):
    """Turns the input s2_data and depth map into training data.

    Args:
        s2_data (np.ndarray): Input s2 raster.
        bathymetry_data (np.ndarray): Input bathymetry raster.
        bands (list): List of indices corresponding to the desired raster
        bands. WARNING: If using the bands attr when creating an Sentinel 2
        mosaic, these indices may differ as order is reset when creating a
        mosaic, e.g. [1,2,4,6] --> [0,1,2,3].

    Returns:
        tuple: Tuple of numpy ndarrays with input features and ground truth
        values.
    """
    # Find no_data values; mask is true where there is valid data
    mask = bathymetry_data != 13
    # Keep only values with depth data
    X = return_features(s2_data, bands)[mask.ravel()]
    # Flip the depth to positive values
    y = abs(bathymetry_data)[mask].copy()

    return X, y
