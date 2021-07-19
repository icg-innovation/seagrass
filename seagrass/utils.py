import numpy as np
import pandas as pd

import json


def save_training_data(filepath, X, y, type=None, **kwargs):
    """Save training data in desired format.

    Args:
        filepath (str): Filepath to save training data.
        X (numpy.ndarray): Training data features.
        y (numpy.ndarray): Machine learning target values.
        type (str, optional): Desired file type. Accepted options are
            currently `npy` and `csv`.
    """

    if type is None:
        type = filepath.split('.')[-1]

    if type == "npy":
        save_training_data_npy(filepath, X, y)

    elif type == "csv":
        save_training_data_csv(filepath, X, y, **kwargs)

    elif type == "modulos":
        save_training_data_modulos(filepath, X, y, **kwargs)
    else:
        raise ValueError("Invalid filetype! Check your filepath.")


def save_training_data_npy(filepath, X, y):
    """Save training data in npy format.

    Args:
        filepath (str): Filepath to save training data.
        X (numpy.ndarray): Training data features.
        y (numpy.ndarray): Machine learning target values.
    """

    filepath_extension = filepath.split(".")[-1]
    if filepath_extension != "npy":
        raise ValueError(
            f"Extension .{filepath_extension} is not valid for "
            "the specified filetype! Check the input filepath."
        )

    np.save(filepath, np.hstack([X, y]))


def save_training_data_csv(filepath, X, y, **kwargs):
    """Save training data in csv format.

    Args:
        filepath (str): Filepath to save training data.
        X (numpy.ndarray): Training data features.
        y (numpy.ndarray): Machine learning target values.
    """

    filepath_extension = filepath.split(".")[-1]
    if filepath_extension != "csv":
        raise ValueError(
            f"Extension .{filepath_extension} is not valid for "
            "the specified filetype! Check the input filepath."
        )

    cols = kwargs.pop("column_labels", None)

    df = pd.DataFrame(np.hstack([X, y]), columns=cols)
    df.to_csv(filepath, **kwargs)


def save_training_data_modulos(filepath, X, y, **kwargs):
    """Save training data in format to be passed to modulos. WARNING: THIS
    METHOD IS CURRENTLY A PLACEHOLDER AND IS YET TO BE WRITTEN.

    Args:
        filepath (str): Filepath to save training data.
        X (numpy.ndarray): Training data features.
        y (numpy.ndarray): Machine learning target values.
    """
    print("This function hasn't been written yet!")
    pass


def extract_training_data(filepath, type=None):
    """Extract training data from file.

    Args:
        filepath (str): Filepath to the training data.
        type (str, optional): Filetype of the training data.

    Returns:
        tuple: Tuple containing training data features and target values for
        machine learning.
    """

    if type is None:
        type = filepath.split('.')[-1]

    if type == "npy":
        X, y = extract_training_data_npy(filepath)
    elif type == "csv":
        X, y = extract_training_data_csv(filepath)
    else:
        raise ValueError("Invalid filetype! Check your filepath.")

    return X, y


def extract_training_data_npy(filepath):
    """Extract training data from npy file.

    Args:
        filepath (str): Filepath to training data in .npy format.

    Returns:
        tuple: Tuple containing training data features and target values for
        machine learning.
    """
    data = np.load(filepath)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)

    return X, y


def extract_training_data_csv(filepath):
    """Extract training data from csv file.

    Args:
        filepath (str): Filepath to training data in .csv format.

    Returns:
        tuple: Tuple containing training data features and target values for
        machine learning.
    """

    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    return X, y


def make_json(
    output_filepath,
    sentinel2_filepath,
    bathymetry_filepath,
    sentinel2_bands=None,
    sentinel2_scale=10000,
    bathymetry_nodata=None,
    bathymetry_nodata_threshold=None,
):
    """Creates import json file and saves to disk.

    Args:
        output_filepath (str): Filepath for output json.
        sentinel2_filepath (str): Filepath to the Sentinel 2 raster
            file.
        bathymetry_filepath (str): Filepath to the bathymetry raster
            file.
        sentinel2_bands (list, optional): List of integers corresponding to
            the desired Sentinel 2 bands. WARNING: There is an issue when
            specifying more than four bands.
        sentinel2_scale (int, optional): Scale factor to obtain the true
            Sentinel 2 pixel value. Defaults to 10000.
        bathymetry_nodata (int, optional): Integer value representing pixels
            containing no data.
        bathymetry_nodata_threshold (float, optional): Determines threshold
            where pixels with values less than bathymetry_nodata_threshold
            will be set equal to bathymetry_nodata instead.
    """
    output_dict = {
        "sentinel2_filepath": sentinel2_filepath,
        "bathymetry_filepath": bathymetry_filepath,
        "sentinel2_bands": sentinel2_bands,
        "sentinel2_scale": sentinel2_scale,
        "bathymetry_nodata": bathymetry_nodata,
        "bathymetry_nodata_threshold": bathymetry_nodata_threshold,
    }

    with open(output_filepath, "w") as f:
        json.dump(output_dict, f)
