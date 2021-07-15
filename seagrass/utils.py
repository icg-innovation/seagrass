import numpy as np
import pandas as pd


def save_training_data(filepath, X, y, type, **kwargs):
    """Save training data in desired format.

    Args:
        filepath (str): Filepath to save training data.
        X (numpy.ndarray): Training data features.
        y (numpy.ndarray): Machine learning target values.
    """
    if type == 'npy':
        save_training_data_npy(filepath, X, y)

    if type == 'csv':
        save_training_data_csv(filepath, X, y, **kwargs)

    if type == 'modulos':
        save_training_data_modulos(filepath, X, y, **kwargs)


def save_training_data_npy(filepath, X, y):
    """Save training data in npy format.

    Args:
        filepath (str): Filepath to save training data.
        X (numpy.ndarray): Training data features.
        y (numpy.ndarray): Machine learning target values.
    """
    np.save(
        filepath,
        np.hstack([X, y])
    )


def save_training_data_csv(filepath, X, y, **kwargs):
    """Save training data in csv format.

    Args:
        filepath (str): Filepath to save training data.
        X (numpy.ndarray): Training data features.
        y (numpy.ndarray): Machine learning target values.
    """
    cols = kwargs.get('columns')

    df = pd.DataFrame(np.hstack([X, y]), columns=cols)
    df.to_csv(filepath)


def save_training_data_modulos(filepath, X, y, **kwargs):
    """Save training data in format to be passed to modulos.

    Args:
        filepath (str): Filepath to save training data.
        X (numpy.ndarray): Training data features.
        y (numpy.ndarray): Machine learning target values.
    """
    pass


def extract_training_data(filepath):
    """Extract training data from .npy file.

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
