import os
import json
import tarfile

import numpy as np
import pandas as pd
import geopandas as gpd

from shutil import rmtree
from geocube.api.core import make_geocube


def save_training_data(filepath, X, y, filetype=None, **kwargs):
    """Save training data in desired format.

    Args:
        filepath (str): Filepath to save training data.
        X (numpy.ndarray): Training data features.
        y (numpy.ndarray): Machine learning target values.
        filetype (str, optional): Desired file type. Accepted options are
            currently `npy`, `csv` and `tar`.
    """

    if filetype is None:
        filetype = filepath.split('.')[-1]

    if filetype == "npy":
        save_training_data_npy(filepath, X, y)

    elif filetype == "csv":
        save_ml_data_csv(filepath, [X, y], 'training', **kwargs)

    elif filetype == "tar":
        save_ml_data_modulos(filepath, [X, y], 'training', **kwargs)

    else:
        raise ValueError("Invalid filetype! Check your filepath.")


def save_prediction_features(filepath, prediction_features, **kwargs):
    """Wrapper function for saving predictions in a Modulos compatible format.
    Future work can be expand the function to other data formats.

    Args:
        filepath (str): Filepath to save prediction data.
        prediction_features (numpy.ndarray): Features to be passed to the
            machine learning model for prediction.
    """

    save_ml_data_modulos(filepath, prediction_features, 'prediction', **kwargs)


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


def save_ml_data_csv(filepath, data, data_purpose, **kwargs):
    """Save training data in csv format.

    Args:
        filepath (str): Filepath to save training data.
        data (numpy.ndarray or tuple or list): Input data for the machine
            learning model. Can be a single numpy array containing features for
            predictions, or a tuple/list containing both features and target
            values for training.
        data_purpose (str): The purpose of the input data. Accepted inputs are
            'training' and 'prediction'.
    """
    if data_purpose != "training" or data_purpose != "prediction":
        raise ValueError(
            "Must specify purpose of input machine learning data! "
            "Accepted values are 'training' and 'prediction'."
        )

    filepath_extension = filepath.split(".")[-1]
    if filepath_extension != "csv":
        raise ValueError(
            f"Extension .{filepath_extension} is not valid for "
            "the specified filetype! Check the input filepath."
        )

    cols = kwargs.pop("column_labels", None)

    if data_purpose == "training":
        data = np.hstack(data)

    df = pd.DataFrame(data, columns=cols)
    df.to_csv(filepath, index=False, **kwargs)


def save_ml_data_modulos(
    tar_filepath,
    data,
    data_purpose,
    delete_tmp=True,
    **kwargs
):
    """Save data in a tar file (containing a csv file and a data structure
    json file) to be passed onto a Modulos machine learning model for either
    training or predictions.

    Args:
        tar_filepath (str): Filepath to save tar file.
        data (numpy.ndarray or tuple or list): Input data for the machine
            learning model. Can be a single numpy array containing features for
            predictions, or a tuple/list containing both features and target
            values for training.
        data_purpose (str): The purpose of the input data. Accepted inputs are
            'training' and 'prediction'.
        delete_tmp (bool): If True, then the 'tmp' directory containing the csv
            and data structure json is deleted after creation of the tar file.
            Otherwise it is not deleted.
    """

    if data_purpose != "training" or data_purpose != "prediction":
        raise ValueError(
            "Must specify purpose of input machine learning data! "
            "Accepted values are 'training' and 'prediction'."
        )

    directory = os.path.dirname(tar_filepath)
    tmp_dir = _make_tmp_dir(directory)

    csv_filename = f"{os.path.basename(tar_filepath).split('.')[0]}.csv"
    csv_filepath = f"{tmp_dir}/{csv_filename}"

    json_filename = "dataset_structure.json"
    json_filepath = f"{tmp_dir}/data_structure.json"

    save_ml_data_csv(csv_filepath, data, data_purpose, **kwargs)
    _make_data_structure_json(csv_filename, json_filepath)

    with tarfile.open(tar_filepath, "w") as tar:
        tar.add(csv_filepath, arcname=csv_filename)
        tar.add(json_filepath, arcname=json_filename)

    if delete_tmp is True:
        rmtree(tmp_dir)


def _make_data_structure_json(csv_filename, json_filepath):
    structure_dict = {
        "type": "table",
        "path": csv_filename,
        "name": csv_filename.split('.')[0],
    }
    version_dict = {"_version": "0.2"}
    data_structure = [structure_dict, version_dict]

    with open(json_filepath, "w") as f:
        json.dump(data_structure, f, indent=4)


def _make_tmp_dir(directory):
    if directory == "":
        directory = "."

    tmp_dir = f"{directory}/tmp"

    if not(os.path.exists(tmp_dir) and os.path.isdir(tmp_dir)):
        os.mkdir(tmp_dir)

    return tmp_dir


def extract_training_data(filepath, filetype=None):
    """Extract training data from file.

    Args:
        filepath (str): Filepath to the training data.
        filetype (str, optional): Filetype of the training data.

    Returns:
        tuple: Tuple containing training data features and target values for
        machine learning.
    """

    if filetype is None:
        filetype = filepath.split('.')[-1]

    if filetype == "npy":
        X, y = extract_training_data_npy(filepath)
    elif filetype == "csv":
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
    ground_truth_filepath,
    sentinel2_bands=None,
    sentinel2_scale=10000,
    ground_truth_nodata=None,
    ground_truth_nodata_threshold=None,
):
    """Creates import json file and saves to disk.

    Args:
        output_filepath (str): Filepath for output json.
        sentinel2_filepath (str): Filepath to the Sentinel 2 raster
            file.
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
            where pixels with values less than ground truth_nodata_threshold
            will be set equal to ground_truth_nodata instead.
    """
    output_dict = {
        "sentinel2_filepath": sentinel2_filepath,
        "ground_truth_filepath": ground_truth_filepath,
        "sentinel2_bands": sentinel2_bands,
        "sentinel2_scale": sentinel2_scale,
        "ground_truth_nodata": ground_truth_nodata,
        "ground_truth_nodata_threshold": ground_truth_nodata_threshold,
    }

    with open(output_filepath, "w") as f:
        json.dump(output_dict, f, indent=4)


def shape_to_binary_raster(shp_filepath, out_dir):
    """Converts vector shapefile into a binary raster file.

    Args:
        shp_filepath (str): Filepath to the shapefile.
        out_dir (str): Output directory to store the binary raster.
    """
    filename = os.path.basename(shp_filepath).split(".")[0]

    geo_df = gpd.read_file(shp_filepath)
    geo_df["data"] = 1   # Data to fill pixel values with.

    out_grid = make_geocube(
        vector_data=geo_df,
        measurements=["data"],
        resolution=(-10, 10),
        fill=0,
    )

    out_grid["data"].rio.to_raster(f"{out_dir}/{filename}.tif")
