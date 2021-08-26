import json
import rioxarray

from seagrass.mosaic import (
    create_raster_mosaic,
    return_mosaic_projected_ground_truth
)


def open_from_json(json_filepath):
    """Opens Sentinel 2 and ground truth rasters from json file.

    Args:
        json_filepath (str): Filepath to json file containing image filepaths
            and relevant information.

    Returns:
        tuple: Tuple of numpy.ndarrays containing a Sentinel 2 raster
        (or mosaic) and a ground truth raster.
    """
    with open(json_filepath) as f:
        data_dict = json.load(f)

    raster_filepath = data_dict["raster_filepath"]
    ground_truth_filepath = data_dict["ground_truth_filepath"]

    raster_bands = data_dict.get("raster_bands")
    raster_scale = data_dict.get("raster_scale", 1)
    ground_truth_nodata = data_dict.get("ground_truth_nodata")
    ground_truth_nodata_threshold = data_dict.get("ground_truth_nodata_threshold")  # noqa: E501

    if type(raster_filepath) is list and len(raster_filepath) == 1:
        raster_filepath = raster_filepath[0]

    if type(raster_filepath) is list:
        raster_mosaic, ground_truth = open_and_match_rasters_mosaic(
            raster_filepath,
            ground_truth_filepath,
            raster_bands,
            raster_scale,
            ground_truth_nodata,
            ground_truth_nodata_threshold
        )

        return raster_mosaic, ground_truth

    else:
        raster_image, ground_truth = open_and_match_rasters(
            raster_filepath,
            ground_truth_filepath,
            raster_scale
        )

        return raster_image, ground_truth


def open_and_match_rasters_mosaic(
    raster_filepath_list,
    ground_truth_filepath,
    raster_bands=None,
    raster_scale=1,
    ground_truth_nodata=None,
    ground_truth_nodata_threshold=None
):
    """Opens and returns a Sentinel 2 mosaic and a ground truth raster with matched
    projections.

    Args:
        raster_filepath_list (str): List of filepaths pointing towards
            Sentinel 2 raster files to be merged into a single mosaic.
        ground_truth_filepath (str): Filepath to the ground truth raster
            file.
        raster_bands (list, optional): List of integers corresponding to
            the desired Sentinel 2 bands. WARNING: There is an issue when
            specifying more than four bands.
        raster_scale (int, optional): Scale factor to obtain the true
            Sentinel 2 pixel value. Defaults to 10000.
        ground_truth_nodata (int, optional): Integer value representing pixels
            containing no data.
        ground_truth_nodata_threshold (float, optional): Determines threshold
            where pixels with values less than ground_truth_nodata_threshold
            will be set equal to ground_truth_nodata instead.

    Returns:
        tuple: Tuple of numpy.ndarrays containing a Sentinel 2 mosaic raster
        and a ground truth raster.
    """
    raster_mosaic, raster_transform = create_raster_mosaic(
        raster_filepath_list,
        ground_truth_filepath,
        raster_bands,
        raster_scale
    )

    ground_truth = return_mosaic_projected_ground_truth(
        ground_truth_filepath,
        raster_transform,
        raster_mosaic.shape,
        ground_truth_nodata,
        ground_truth_nodata_threshold
    )

    return raster_mosaic, ground_truth


def open_raster_image(raster_filepath, raster_scale=1):
    """Opens and returns the specified Sentinel 2 image, scaled to the true
    pixel value.

    Args:
        raster_filepath (str): Filepath to the Sentinel 2 raster
            file.
        raster_scale (int, optional): Scale factor to obtain the true
            Sentinel 2 pixel value. Defaults to 10000.

    Returns:
        xarray.DataArray: Sentinel 2 raster.
    """
    raster = rioxarray.open_rasterio(raster_filepath)
    raster = raster / raster_scale

    return raster


def open_and_match_rasters(
    raster_filepath,
    ground_truth_filepath,
    raster_scale=1
):
    """Opens and returns Sentinel 2 and ground truth rasters with matched
    projections.

    Args:
        raster_filepath (str): Filepath to the Sentinel 2 raster
            file.
        ground_truth_filepath (str): Filepath to the ground truth raster
            file.

    Returns:
        tuple: Tuple of xarray.DataArrays containing a Sentinel 2 raster
        and a ground truth raster.
    """
    raster = open_raster_image(raster_filepath, raster_scale)

    ground_truth = rioxarray.open_rasterio(ground_truth_filepath)
    ground_truth = ground_truth.rio.reproject_match(raster)

    return raster, ground_truth
