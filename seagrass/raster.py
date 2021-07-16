import json
import rioxarray

from seagrass.mosaic import create_s2_mosaic, return_s2_mosaic_projected_depth


def open_from_json(json_filepath):
    """Opens Sentinel 2 and bathymetry rasters from json file

    Args:
        json_filepath (str): Filepath to json file containing image filepaths
            and relevant information.

    Returns:
        tuple: Tuple of numpy.ndarrays containing a Sentinel 2 raster
        (or mosaic) and a bathymetry raster.
    """
    with open(json_filepath) as f:
        data_dict = json.load(f)

    sentinel2_filepath = data_dict["sentinel2_filepath"]
    bathymetry_filepath = data_dict["bathymetry_filepath"]

    sentinel2_bands = data_dict.get("sentinel2_bands")
    sentinel2_scale = data_dict.get("sentinel2_scale", 10000)
    bathymetry_nodata = data_dict.get("bathymetry_nodata")
    bathymetry_nodata_threshold = data_dict.get("bathymetry_nodata_threshold")

    if type(sentinel2_filepath) is list and len(sentinel2_filepath) == 1:
        sentinel2_filepath = sentinel2_filepath[0]

    if type(sentinel2_filepath) is list:
        sentinel2_mosaic, bathymetry = return_s2_mosaic_projected_depth(
            sentinel2_filepath,
            bathymetry_filepath,
            sentinel2_bands,
            sentinel2_scale,
            bathymetry_nodata,
            bathymetry_nodata_threshold
        )

        return sentinel2_mosaic, bathymetry

    else:
        sentinel2_image, bathymetry = open_and_match_rasters(
            sentinel2_filepath,
            bathymetry_filepath,
            sentinel2_scale
        )

        return sentinel2_image, bathymetry


def open_sentinel2_mosaic_and_matched_depth(
    sentinel2_filepath,
    bathymetry_filepath,
    sentinel2_bands=None,
    sentinel2_scale=None,
    bathymetry_nodata=None,
    bathymetry_nodata_threshold=None
):
    sentinel2_mosaic, sentinel2_transform = create_s2_mosaic(
        sentinel2_filepath,
        bathymetry_filepath,
        sentinel2_bands,
        sentinel2_scale
    )

    bathymetry, _ = return_s2_mosaic_projected_depth(
        bathymetry_filepath,
        sentinel2_transform,
        sentinel2_mosaic.shape,
        bathymetry_nodata,
        bathymetry_nodata_threshold
    )

    return sentinel2_mosaic, bathymetry


def open_and_match_rasters(
    sentinel2_filepath,
    bathymetry_filepath,
    sentinel2_scale=10000
):
    """Opens and returns Sentinel 2 and bathymetry rasters with matched
    projections, crs, resolution, etc.

    Args:
        sentinel2_filepath (str): Filepath to the Sentinel 2 raster
            file.
        bathymetry_filepath (str): Filepath to the bathymetry raster
            file.

    Returns:
        tuple: Tuple of numpy.ndarrays containing a Sentinel 2 raster
        and a bathymetry raster.
    """
    sentinel2 = rioxarray.open_rasterio(sentinel2_filepath)

    bathymetry = rioxarray.open_rasterio(bathymetry_filepath)
    bathymetry = bathymetry.rio.reproject_match(sentinel2)

    return sentinel2.values/sentinel2_scale, bathymetry.values
