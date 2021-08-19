import os
import json
import tarfile

import numpy as np
import pandas as pd
import geopandas as gpd

from shutil import rmtree
from geocube.api.core import make_geocube


def save_training_data(filepath, X, y, **kwargs):
    """Save training data for machine learning in desired format.

    Args:
        filepath (str): Filepath to save training data.
        X (numpy.ndarray): Training data features.
        y (numpy.ndarray): Machine learning target values.
    """

    filepath_extension = os.path.splitext(filepath)[-1]

    if filepath_extension == ".npy":
        save_ml_data_npy(filepath, [X, y], 'training', **kwargs)

    elif filepath_extension == ".csv":
        save_ml_data_csv(filepath, [X, y], 'training', **kwargs)

    elif filepath_extension == ".tar":
        save_ml_data_modulos(filepath, [X, y], 'training', **kwargs)

    else:
        raise ValueError(
            f"Filepath extension '{filepath_extension}' is invalid! Accepted "
            "extensions are '.npy', '.csv' and '.tar'."
        )


def save_prediction_features(filepath, features, **kwargs):
    """Save features for machine learning predictions in desired format.
    Accepted filepath extensions are '.npy', '.csv' and '.tar'.

    Args:
        filepath (str): Filepath to save prediction data.
        prediction_features (numpy.ndarray): Features to be passed to the
            machine learning model for prediction.
    """
    filepath_extension = os.path.splitext(filepath)[-1]

    if filepath_extension == ".npy":
        save_ml_data_npy(filepath, features, 'prediction', **kwargs)

    elif filepath_extension == ".csv":
        save_ml_data_csv(filepath, features, 'prediction', **kwargs)

    elif filepath_extension == ".tar":
        save_ml_data_modulos(filepath, features, 'prediction', **kwargs)

    else:
        raise ValueError(
            f"Filepath extension '{filepath_extension}' is invalid! Accepted "
            "extensions are '.npy', '.csv' and '.tar'."
        )


def save_ml_data_npy(filepath, data, data_purpose, **kwargs):
    """Save machine learning data in npy format.

    Args:
        filepath (str): Filepath to save machine learning data.
        data (numpy.ndarray or tuple or list): Input data for the machine
            learning model. Can be a single numpy array containing features for
            predictions, or a tuple/list containing both features and target
            values for training.
        data_purpose (str): The purpose of the input data. Accepted inputs are
            'training' and 'prediction'.
    """
    if data_purpose != "training" and data_purpose != "prediction":
        raise ValueError(
            "Must specify purpose of input machine learning data! "
            "Accepted values are 'training' and 'prediction'."
        )

    filepath_extension = os.path.splitext(filepath)[-1]
    if filepath_extension != ".npy":
        raise ValueError(
            f"Extension {filepath_extension} is not valid for "
            "the specified filetype! Check the input filepath."
        )
    if data_purpose == "training":
        data = np.hstack(data)

    np.save(filepath, data, **kwargs)


def save_ml_data_csv(filepath, data, data_purpose, **kwargs):
    """Save machine learning data in csv format.

    Args:
        filepath (str): Filepath to save machine learning data.
        data (numpy.ndarray or tuple or list): Input data for the machine
            learning model. Can be a single numpy array containing features for
            predictions, or a tuple/list containing both features and target
            values for training.
        data_purpose (str): The purpose of the input data. Accepted inputs are
            'training' and 'prediction'.
    """
    if data_purpose != "training" and data_purpose != "prediction":
        raise ValueError(
            "Must specify purpose of input machine learning data! "
            "Accepted values are 'training' and 'prediction'."
        )

    filepath_extension = os.path.splitext(filepath)[-1]
    if filepath_extension != ".csv":
        raise ValueError(
            f"Extension {filepath_extension} is not valid for "
            "the specified filetype! Check the input filepath."
        )

    cols = kwargs.pop("column_labels", None)

    if data_purpose == "training":
        data = np.hstack(data)

    df = pd.DataFrame(data, columns=cols)
    df.to_csv(filepath, index=False, **kwargs)


def save_ml_data_modulos(
    filepath,
    data,
    data_purpose,
    delete_tmp=True,
    **kwargs
):
    """Save data in a tar file (containing a csv file and a data structure
    json file) to be passed onto a Modulos machine learning model for either
    training or predictions.

    Args:
        filepath (str): Filepath to save tar file.
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

    if data_purpose != "training" and data_purpose != "prediction":
        raise ValueError(
            "Must specify purpose of input machine learning data! "
            "Accepted values are 'training' and 'prediction'."
        )

    filepath_extension = os.path.splitext(filepath)[-1]
    if filepath_extension != ".tar":
        raise ValueError(
            f"Extension {filepath_extension} is not valid for "
            "the specified filetype! Check the input filepath."
        )

    directory = os.path.dirname(filepath)
    tmp_dir = _make_tmp_dir(directory)

    file_basename = os.path.basename(filepath)
    csv_filename = f"{os.path.splitext(file_basename)[0]}.csv"
    csv_filepath = f"{tmp_dir}/{csv_filename}"

    json_filename = "dataset_structure.json"
    json_filepath = f"{tmp_dir}/{json_filename}"

    save_ml_data_csv(csv_filepath, data, data_purpose, **kwargs)
    _make_data_structure_json(csv_filename, json_filepath)

    with tarfile.open(filepath, "w") as tar:
        tar.add(csv_filepath, arcname=csv_filename)
        tar.add(json_filepath, arcname=json_filename)

    if delete_tmp is True:
        rmtree(tmp_dir)


def _make_data_structure_json(csv_filename, json_filepath):
    """Makes a data_structure.json file to be included in a Modulos compatible
    tar file.

    Args:
        csv_filename (str): Filename of the csv file to be included in the tar
            file.
        json_filepath (str): Filepath of output json file.
    """
    structure_dict = {
        "type": "table",
        "path": csv_filename,
        "name": os.path.splitext(csv_filename)[0],
    }
    version_dict = {"_version": "0.2"}
    data_structure = [structure_dict, version_dict]

    with open(json_filepath, "w") as f:
        json.dump(data_structure, f, indent=4)


def _make_tmp_dir(directory):
    """Makes a temporary directory to store the generated csv and
    data_structure.json files when creating a Modulos compatible tar file.

    Args:
        directory (str): Parent directory.

    Returns:
        str: Filepath of the temporary directory.
    """
    if directory == "":
        directory = "."

    tmp_dir = f"{directory}/tmp"

    if not(os.path.exists(tmp_dir) and os.path.isdir(tmp_dir)):
        os.mkdir(tmp_dir)

    return tmp_dir


def extract_training_data(filepath):
    """Extract training data from file.

    Args:
        filepath (str): Filepath to the training data.

    Returns:
        tuple: Tuple containing training data features and target values for
        machine learning.
    """

    filepath_extension = os.path.splitext(filepath)[-1]

    if filepath_extension == ".npy":
        X, y = extract_training_data_npy(filepath)
    elif filepath_extension == ".csv":
        X, y = extract_training_data_csv(filepath)
    else:
        raise ValueError(
            f"Filepath extension '{filepath_extension}' is invalid! Accepted "
            "extensions are '.npy' and '.csv'"
        )

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
    raster_filepath,
    ground_truth_filepath,
    raster_bands=None,
    raster_scale=1,
    ground_truth_nodata=None,
    ground_truth_nodata_threshold=None,
):
    """Creates import json file and saves to disk.

    Args:
        output_filepath (str): Filepath for output json.
        raster_filepath (str): Filepath to the raster image.
        ground_truth_filepath (str): Filepath to the ground truth raster
            file.
        raster_bands (list, optional): List of integers corresponding to
            the desired raster bands. WARNING: There is an issue when
            specifying more than four bands.
        raster_scale (int, optional): Scale factor to obtain the true
            raster pixel value. Defaults to 1.
        ground_truth_nodata (int, optional): Integer value representing pixels
            containing no data.
        ground_truth_nodata_threshold (float, optional): Determines threshold
            where pixels with values less than ground truth_nodata_threshold
            will be set equal to ground_truth_nodata instead.
    """
    output_dict = {
        "raster_filepath": raster_filepath,
        "ground_truth_filepath": ground_truth_filepath,
        "raster_bands": raster_bands,
        "raster_scale": raster_scale,
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
    shp_basename = os.path.basename(shp_filepath)
    filename = os.path.splitext(shp_basename)[0]

    geo_df = gpd.read_file(shp_filepath)
    geo_df["data"] = 1   # Data to fill pixel values with.

    out_grid = make_geocube(
        vector_data=geo_df,
        measurements=["data"],
        resolution=(-10, 10),
        fill=0,
    )

    out_grid["data"].rio.to_raster(f"{out_dir}/{filename}.tif")
