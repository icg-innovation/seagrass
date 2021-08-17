import numpy as np
from scipy.ndimage import gaussian_filter


def return_features(raster, bands=None, blurring=False):
    """Returns a feature array using the input raster image.

    Args:
        raster (np.ndarray): Input raster data.
        bands (list, optional): List of indices corresponding to the desired
            raster bands. WARNING: If using the bands attr when creating a
            Sentinel 2 mosaic, these indices may differ as order is reset when
            creating a mosaic, e.g. [1,2,4,6] --> [0,1,2,3].

    Returns:
        np.ndarray: Reshaped feature data.
    """

    if bands is None:
        bands = list(np.arange(len(raster)))

    bands_1D = [raster[band].ravel() for band in bands]

    if blurring:
        blurred_1D = [
            gaussian_filter(raster[band], 2.0).ravel() for band in bands
        ]
        all_bands = bands_1D + blurred_1D

    else:
        all_bands = bands_1D

    return np.vstack(all_bands).T


def create_training_data(
    raster,
    ground_truth_data,
    no_data_value,
    bands=None,
    blurring=False,
):
    """Turns the input raster and ground truth map into training data.

    Args:
        raster (np.ndarray): Input raster image.
        ground_truth_data (np.ndarray): Input ground truth raster.
        no_data_value (int): Integer value representing pixels containing no
            data.
        bands (list, optional): List of indices corresponding to the desired
            raster bands. WARNING: If using the bands attr when creating a
            Sentinel 2 mosaic, these indices may differ as order is reset when
            creating a mosaic, e.g. [1,2,4,6] --> [0,1,2,3].

    Returns:
        tuple: Tuple of numpy ndarrays with input features and ground truth
        values.
    """

    # Find no_data values; mask is true where there is valid data
    mask = ground_truth_data != no_data_value
    # Keep only values with ground truth data
    X = return_features(raster, bands, blurring)[mask.ravel()]
    # Flip the ground truth to positive values - maybe should change this
    y = abs(ground_truth_data)[mask].copy().reshape(-1, 1)

    return X, y
