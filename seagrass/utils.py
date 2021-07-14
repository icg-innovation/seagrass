import numpy as np


def save_training_data(filepath, X, y):
    """Save training data in .npy format.

    Args:
        filepath (str): Filepath to save training data. Must end in .npy.
        X (numpy.ndarray): Training data features.
        y (numpy.ndarray): Machine learning target values.
    """
    np.save(
        filepath,
        np.hstack([X, y])
    )


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
