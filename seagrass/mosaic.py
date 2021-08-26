import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
from geopandas import GeoSeries
from shapely.geometry import box


def create_raster_mosaic(
    s2_file_list,
    ground_truth_filepath,
    bands=None,
    scale=10000
):
    """Generates a Sentinel 2 mosaic for the areas intersecting the input
    ground truth raster.

    Args:
        s2_filepath (str): List of filepaths of Sentinel 2 rasters to be
            merged.
        ground_truth_filepath (str): Filepath to the ground truth raster
            file.
        bands (list): List of integers corresponding to the desired Sentinel 2
            bands. WARNING: There is an issue when specifying more than four
            bands.
        scale (int, optional): Scale factor to obtain the true Sentinel 2
            pixel value. Defaults to 10000.

    Returns:
        A tuple containing the output raster mosaic and the
        affine transform matrix.
    """
    intersecting_rasters = intersecting_tiles(
        s2_file_list,
        ground_truth_filepath
    )

    s2_raster_list = [rasterio.open(file) for file in intersecting_rasters]
    ground_truth_raster = rasterio.open(ground_truth_filepath)

    mosaic, transform = merge(
        s2_raster_list,
        indexes=bands,
        dtype=np.float32
    )

    mosaic, transform = change_crs(
        mosaic,
        s2_raster_list[0].crs,
        transform,
        ground_truth_raster.crs
    )

    mosaic /= scale

    return mosaic, transform


def return_mosaic_projected_ground_truth(
    ground_truth_filepath,
    s2_transform,
    s2_shape,
    no_data_value=None,
    no_data_threshold=None,
):
    """Returns ground truth raster projected onto the Sentinel 2 mosaic.

    Args:
        ground_truth_filepath (str): Filepath to the ground truth raster
            file.
        s2_transform (numpy.ndarray): Transform matrix generated when creating
            the Sentinel 2 mosaic.
        s2_shape (tuple): Dimensions of the Sentinel 2 mosaic.
        no_data_value (int): Integer value representing pixels containing no
            data.
        no_data_threshold (float, optional): Determines threshold where pixels
            with values less than no_data_threshold will be set equal to
            no_data_value instead.

    Returns:
        numpy.ndarray: Projected ground truth raster data.
    """
    ground_truth_data = rasterio.open(ground_truth_filepath)
    ground_truth_raster = ground_truth_data.read(1)

    if no_data_threshold:
        ground_truth_raster[
            ground_truth_raster < no_data_threshold
        ] = no_data_value

    # Resamples to Sentinel 2 image resolution?
    s2_projected_ground_truth, _ = reproject(
        ground_truth_raster,
        np.zeros((1, s2_shape[-2], s2_shape[-1]), dtype=np.float32),
        src_transform=ground_truth_data.transform,
        src_crs=ground_truth_data.crs,
        src_nodata=no_data_value,
        dst_transform=s2_transform,
        dst_crs=ground_truth_data.crs,
        dst_resolution=10,
        resampling=Resampling.nearest,
    )

    return s2_projected_ground_truth


def intersecting_tiles(raster_filepath_list, reference_raster_filepath):
    """Returns list of raster objects that intersect with an input reference
    raster.

    Args:
        raster_list (list): List of input raster objects.
        reference_raster (rasterio.open object): Reference raster objects.

    Returns:
        list: List of intersecting raster objects.
    """
    raster_list = [rasterio.open(file) for file in raster_filepath_list]
    reference_raster = rasterio.open(reference_raster_filepath)

    ref = box(*reference_raster.bounds)

    boundary_boxes = GeoSeries(
        [box(*raster.bounds) for raster in raster_list],
        crs=raster_list[0].crs,
    )

    intersecting_tiles = [
        raster_filepath_list[idx]
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
    no_of_channels = data.shape[0]

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
        np.zeros((no_of_channels, width, height), dtype=np.float32),
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=None,
        dst_transform=new_transform,
        dst_crs=dst_crs,
        dst_resolution=10,
        resampling=Resampling.bilinear,
    )

    return reprojected, transform
