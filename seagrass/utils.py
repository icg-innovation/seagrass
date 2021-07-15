import numpy as np
import pandas as pd


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
    """Save training data in format to be passed to modulos.

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
