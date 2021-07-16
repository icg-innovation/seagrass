import rioxarray


def open_and_match_rasters(sentinel_2_filepath, bathymetry_filepath):
    sentinel_2 = rioxarray.open_rasterio(sentinel_2_filepath)

    bathymetry = rioxarray.open_rasterio(bathymetry_filepath)
    bathymetry = bathymetry.rio.reproject_match(sentinel_2)

    return sentinel_2, bathymetry
