import numpy as np
from scipy.ndimage import gaussian_filter


def return_features(data, bands=None):
    """Returns a array of default features for the training data.

    Args:
        data (np.ndarray): Input raster data.
        bands (list, optional): List of indices corresponding to the desired
        raster bands. WARNING: If using the bands attr when creating a
        Sentinel 2 mosaic, these indices may differ as order is reset when
        creating a mosaic, e.g. [1,2,4,6] --> [0,1,2,3].

    Returns:
        np.ndarray: Reshaped feature data.
    """

    if bands is None:
        bands = list(np.arange(len(data)))

    bands_1D = [data[band].ravel() for band in bands]
    blurred_1D = [gaussian_filter(data[band], 2.).ravel() for band in bands]

    all_bands = bands_1D + blurred_1D

    return np.vstack(all_bands).T


def create_training_data(
    s2_data,
    bathymetry_data,
    no_data_value,
    s2_bands=None,
):
    """Turns the input s2_data and depth map into training data.

    Args:
        s2_data (np.ndarray): Input s2 raster.
        bathymetry_data (np.ndarray): Input bathymetry raster.
        s2_bands (list, optional): List of indices corresponding to the desired
        raster bands. WARNING: If using the bands attr when creating a
        Sentinel 2 mosaic, these indices may differ as order is reset when
        creating a mosaic, e.g. [1,2,4,6] --> [0,1,2,3].

    Returns:
        tuple: Tuple of numpy ndarrays with input features and ground truth
        values.
    """

    if s2_bands is None:
        s2_bands = list(np.arange(len(s2_data)))

    # Find no_data values; mask is true where there is valid data
    mask = bathymetry_data != no_data_value
    # Keep only values with depth data
    X = return_features(s2_data, s2_bands)[mask.ravel()]
    # Flip the depth to positive values
    y = abs(bathymetry_data)[mask].copy()

    return X, y
