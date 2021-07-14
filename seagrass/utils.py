import numpy as np


def save_training_data(filepath, X, y):
    np.save(
        filepath,
        np.hstack([X, y])
    )


def extract_training_data(filepath):
    data = np.load(filepath)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)

    return X, y
