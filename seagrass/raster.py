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

    sentinel_2_filepath = data_dict["s2_filepath"]
    bathymetry_filepath = data_dict["bathymetry_filepath"]

    s2_bands = data_dict.get("s2_bands")
    s2_scale = data_dict.get("s2_scale", 10000)
    bathymetry_nodata = data_dict.get("bathymetry_nodata")
    bathymetry_nodata_threshold = data_dict.get("bathymetry_nodata_threshold")

    if type(sentinel_2_filepath) is list and len(sentinel_2_filepath) == 1:
        sentinel_2_filepath = sentinel_2_filepath[0]

    if type(sentinel_2_filepath) is list:
        s2_mosaic, s2_transform = create_s2_mosaic(
            sentinel_2_filepath,
            bathymetry_filepath,
            s2_bands,
            s2_scale
        )

        bathymetry, _ = return_s2_mosaic_projected_depth(
            bathymetry_filepath,
            s2_transform,
            s2_mosaic.shape,
            bathymetry_nodata,
            bathymetry_nodata_threshold
        )

        return s2_mosaic, bathymetry

    else:
        s2_image, bathymetry = open_and_match_rasters(
            sentinel_2_filepath,
            bathymetry_filepath
        )

        return s2_image, bathymetry


def open_and_match_rasters(sentinel_2_filepath, bathymetry_filepath):
    """Opens and returns Sentinel 2 and bathymetry rasters with matched
    projections, crs, resolution, etc.

    Args:
        sentinel_2_filepath (str): Filepath to the Sentinel 2 raster
            file.
        bathymetry_filepath (str): Filepath to the bathymetry raster
            file.

    Returns:
        tuple: Tuple of numpy.ndarrays containing a Sentinel 2 raster
        and a bathymetry raster.
    """
    sentinel_2 = rioxarray.open_rasterio(sentinel_2_filepath)

    bathymetry = rioxarray.open_rasterio(bathymetry_filepath)
    bathymetry = bathymetry.rio.reproject_match(sentinel_2)

    return sentinel_2.values, bathymetry.values
