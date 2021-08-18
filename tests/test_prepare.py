import unittest
from unittest.mock import patch

from seagrass import prepare

import numpy as np
from numpy.testing import assert_array_equal


class TestPrepare(unittest.TestCase):

    def setUp(self):
        self.data = np.array(
            (
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]
                ],
                [
                    [10, 11, 12],
                    [13, 14, 15],
                    [16, 17, 18]
                ]
            )
        )

    def test_return_features(self):
        expected_features = np.array(
            (
                [1, 10],
                [2, 11],
                [3, 12],
                [4, 13],
                [5, 14],
                [6, 15],
                [7, 16],
                [8, 17],
                [9, 18]
            )
        )

        features = prepare.return_features(self.data)

        assert_array_equal(features, expected_features)

    def test_return_features_bands(self):
        expected_features = np.array(
                (
                    [1],
                    [2],
                    [3],
                    [4],
                    [5],
                    [6],
                    [7],
                    [8],
                    [9]
                )
            )

        bands = [0]
        features = prepare.return_features(self.data, bands)

        assert_array_equal(features, expected_features)

    @patch("seagrass.prepare.gaussian_filter")
    def test_return_features_blurring(self, mock_gaussian_filter):
        expected_features = np.array(
                (
                    [1, -1],
                    [2, -2],
                    [3, -3],
                    [4, -4],
                    [5, -5],
                    [6, -6],
                    [7, -7],
                    [8, -8],
                    [9, -9]
                )
            )

        mock_gaussian_filter.return_value.ravel.return_value = [
            -1, -2, -3, -4, -5, -6, -7, -8, -9
        ]

        bands = [0]
        features = prepare.return_features(self.data, bands, blurring=True)

        mock_gaussian_filter.assert_called_once()
        mock_gaussian_filter.return_value.ravel.assert_called_once()
        assert_array_equal(features, expected_features)

    def test_create_training_data(self):
        ground_truth = np.array(
            (
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]

            )
        )

        no_data_value = -9999

        expected_X = np.array(
            (
                [1, 10],
                [2, 11],
                [3, 12],
                [4, 13],
                [5, 14],
                [6, 15],
                [7, 16],
                [8, 17],
                [9, 18]

            )
        )

        expected_y = np.array(
            (
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
                [9]

            )
        )

        X, y = prepare.create_training_data(
            self.data, ground_truth, no_data_value
        )

        assert_array_equal(X, expected_X)
        assert_array_equal(y, expected_y)

    def test_create_training_data_nodata(self):
        ground_truth = np.array(
            (
                [-9999, 2, -9999],
                [4, 5, 6],
                [7, -9999, 9]

            )
        )

        no_data_value = -9999

        expected_X = np.array(
            (
                [2, 11],
                [4, 13],
                [5, 14],
                [6, 15],
                [7, 16],
                [9, 18]

            )
        )

        expected_y = np.array(
            (
                [2],
                [4],
                [5],
                [6],
                [7],
                [9]

            )
        )

        X, y = prepare.create_training_data(
            self.data, ground_truth, no_data_value
        )

        assert_array_equal(X, expected_X)
        assert_array_equal(y, expected_y)


if __name__ == '__main__':
    unittest.main()
