import json
import rioxarray

from seagrass.mosaic import (
    create_s2_mosaic,
    return_s2_mosaic_projected_ground_truth
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

    sentinel2_filepath = data_dict["sentinel2_filepath"]
    ground_truth_filepath = data_dict["ground_truth_filepath"]

    sentinel2_bands = data_dict.get("sentinel2_bands")
    sentinel2_scale = data_dict.get("sentinel2_scale", 10000)
    ground_truth_nodata = data_dict.get("ground_truth_nodata")
    ground_truth_nodata_threshold = data_dict.get("ground_truth_nodata_threshold")  # noqa: E501

    if type(sentinel2_filepath) is list and len(sentinel2_filepath) == 1:
        sentinel2_filepath = sentinel2_filepath[0]

    if type(sentinel2_filepath) is list:
        sentinel2_mosaic, ground_truth = open_and_match_rasters_mosaic(
            sentinel2_filepath,
            ground_truth_filepath,
            sentinel2_bands,
            sentinel2_scale,
            ground_truth_nodata,
            ground_truth_nodata_threshold
        )

        return sentinel2_mosaic, ground_truth

    else:
        sentinel2_image, ground_truth = open_and_match_rasters(
            sentinel2_filepath,
            ground_truth_filepath,
            sentinel2_scale
        )

        return sentinel2_image, ground_truth


def open_and_match_rasters_mosaic(
    sentinel2_filepath_list,
    ground_truth_filepath,
    sentinel2_bands=None,
    sentinel2_scale=10000,
    ground_truth_nodata=None,
    ground_truth_nodata_threshold=None
):
    """Opens and returns a Sentinel 2 mosaic and a ground truth raster with matched
    projections.

    Args:
        sentinel2_filepath_list (str): List of filepaths pointing towards
            Sentinel 2 raster files to be merged into a single mosaic.
        ground_truth_filepath (str): Filepath to the ground truth raster
            file.
        sentinel2_bands (list, optional): List of integers corresponding to
            the desired Sentinel 2 bands. WARNING: There is an issue when
            specifying more than four bands.
        sentinel2_scale (int, optional): Scale factor to obtain the true
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
    sentinel2_mosaic, sentinel2_transform = create_s2_mosaic(
        sentinel2_filepath_list,
        ground_truth_filepath,
        sentinel2_bands,
        sentinel2_scale
    )

    ground_truth = return_s2_mosaic_projected_ground_truth(
        ground_truth_filepath,
        sentinel2_transform,
        sentinel2_mosaic.shape,
        ground_truth_nodata,
        ground_truth_nodata_threshold
    )

    return sentinel2_mosaic, ground_truth


def open_sentinel2_image(sentinel2_filepath, sentinel2_scale=10000):
    """Opens and returns the specified Sentinel 2 image, scaled to the true
    pixel value.

    Args:
        sentinel2_filepath (str): Filepath to the Sentinel 2 raster
            file.
        sentinel2_scale (int, optional): Scale factor to obtain the true
            Sentinel 2 pixel value. Defaults to 10000.

    Returns:
        xarray.DataArray: Sentinel 2 raster.
    """
    sentinel2_raster = rioxarray.open_rasterio(sentinel2_filepath)
    sentinel2_raster /= sentinel2_scale

    return sentinel2_raster


def open_and_match_rasters(
    sentinel2_filepath,
    ground_truth_filepath,
    sentinel2_scale=10000
):
    """Opens and returns Sentinel 2 and ground truth rasters with matched
    projections.

    Args:
        sentinel2_filepath (str): Filepath to the Sentinel 2 raster
            file.
        ground_truth_filepath (str): Filepath to the ground truth raster
            file.

    Returns:
        tuple: Tuple of xarray.DataArrays containing a Sentinel 2 raster
        and a ground truth raster.
    """
    sentinel2 = open_sentinel2_image(sentinel2_filepath, sentinel2_scale)

    ground_truth = rioxarray.open_rasterio(ground_truth_filepath)
    ground_truth = ground_truth.rio.reproject_match(sentinel2)

    return sentinel2, ground_truth
