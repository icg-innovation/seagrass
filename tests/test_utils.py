import unittest
from unittest.mock import patch, Mock, call
from numpy.testing import assert_array_equal

from seagrass import utils

import numpy as np
import pandas as pd


class TestUtils(unittest.TestCase):

    @patch("seagrass.utils.save_ml_data_modulos")
    @patch("seagrass.utils.save_ml_data_csv")
    @patch("seagrass.utils.save_ml_data_npy")
    def test_save_training_data(
        self,
        mock_save_ml_data_npy,
        mock_save_ml_data_csv,
        mock_save_ml_data_modulos
    ):
        test_filepath_dict = {
            "test_file.path.npy": mock_save_ml_data_npy,
            "test_file.path.csv": mock_save_ml_data_csv,
            "test_file.path.tar": mock_save_ml_data_modulos
        }

        mock_X = Mock()
        mock_y = Mock()

        for filepath, mock_function in test_filepath_dict.items():
            utils.save_training_data(filepath, mock_X, mock_y)

            with self.subTest(filepath=filepath):
                mock_function.assert_called_once_with(
                    filepath,
                    [mock_X, mock_y],
                    "training"
                )

    @patch("seagrass.utils.save_ml_data_modulos")
    @patch("seagrass.utils.save_ml_data_csv")
    @patch("seagrass.utils.save_ml_data_npy")
    def test_save_training_data_kwargs(
        self,
        mock_save_ml_data_npy,
        mock_save_ml_data_csv,
        mock_save_ml_data_modulos
    ):
        test_filepath_dict = {
            "test_file.path.npy": mock_save_ml_data_npy,
            "test_file.path.csv": mock_save_ml_data_csv,
            "test_file.path.tar": mock_save_ml_data_modulos
        }

        mock_X = Mock()
        mock_y = Mock()

        test_kwargs = {
            "testInt": 1,
            "testBool": True,
            "testStr": "test",
            }

        for filepath, mock_function in test_filepath_dict.items():
            utils.save_training_data(filepath, mock_X, mock_y, **test_kwargs)

            with self.subTest(filepath=filepath):
                mock_function.assert_called_once_with(
                    filepath,
                    [mock_X, mock_y],
                    "training",
                    **test_kwargs
                )

    @patch("seagrass.utils.save_ml_data_modulos")
    @patch("seagrass.utils.save_ml_data_csv")
    @patch("seagrass.utils.save_ml_data_npy")
    def test_save_training_data_invalid_filetypes(
        self,
        mock_save_ml_data_npy,
        mock_save_ml_data_csv,
        mock_save_ml_data_modulos
    ):
        invalid_filepaths = [
            "test_file.path.py",
            "test_file.path.npy.",
            "test_file.path",
            "test_file",
        ]

        mock_X = Mock()
        mock_y = Mock()

        for filepath in invalid_filepaths:
            with self.subTest(invalid_filepath=filepath):
                with self.assertRaises(ValueError):
                    utils.save_training_data(filepath, mock_X, mock_y)

    @patch("seagrass.utils.save_ml_data_modulos")
    @patch("seagrass.utils.save_ml_data_csv")
    @patch("seagrass.utils.save_ml_data_npy")
    def test_save_prediction_features(
        self,
        mock_save_ml_data_npy,
        mock_save_ml_data_csv,
        mock_save_ml_data_modulos
    ):
        test_filepath_dict = {
            "test_file.path.npy": mock_save_ml_data_npy,
            "test_file.path.csv": mock_save_ml_data_csv,
            "test_file.path.tar": mock_save_ml_data_modulos
        }

        mock_X = Mock()

        for filepath, mock_function in test_filepath_dict.items():
            utils.save_prediction_features(filepath, mock_X)

            with self.subTest(filepath=filepath):
                mock_function.assert_called_once_with(
                    filepath,
                    mock_X,
                    "prediction"
                )

    @patch("seagrass.utils.save_ml_data_modulos")
    @patch("seagrass.utils.save_ml_data_csv")
    @patch("seagrass.utils.save_ml_data_npy")
    def test_save_prediction_features_kwargs(
        self,
        mock_save_ml_data_npy,
        mock_save_ml_data_csv,
        mock_save_ml_data_modulos
    ):
        test_filepath_dict = {
            "test_file.path.npy": mock_save_ml_data_npy,
            "test_file.path.csv": mock_save_ml_data_csv,
            "test_file.path.tar": mock_save_ml_data_modulos
        }

        mock_X = Mock()

        test_kwargs = {
            "testInt": 1,
            "testBool": True,
            "testStr": "test",
            }

        for filepath, mock_function in test_filepath_dict.items():
            utils.save_prediction_features(filepath, mock_X, **test_kwargs)

            with self.subTest(filepath=filepath):
                mock_function.assert_called_once_with(
                    filepath,
                    mock_X,
                    "prediction",
                    **test_kwargs
                )

    @patch("seagrass.utils.save_ml_data_modulos")
    @patch("seagrass.utils.save_ml_data_csv")
    @patch("seagrass.utils.save_ml_data_npy")
    def test_save_prediction_features_invalid_filetypes(
        self,
        mock_save_ml_data_npy,
        mock_save_ml_data_csv,
        mock_save_ml_data_modulos
    ):
        invalid_filepaths = [
            "test_file.path.py",
            "test_file.path.npy.",
            "test_file.path",
            "test_file",
        ]

        mock_X = Mock()

        for filepath in invalid_filepaths:
            with self.subTest(invalid_filepath=filepath):
                with self.assertRaises(ValueError):
                    utils.save_prediction_features(filepath, mock_X)

    @patch("seagrass.utils.np.save")
    @patch("seagrass.utils.np.hstack")
    def test_save_ml_data_npy_training(self, mock_hstack, mock_save):
        test_filepath = "test_file.npy"
        mock_data = Mock()
        data_purpose = "training"

        utils.save_ml_data_npy(test_filepath, mock_data, data_purpose)

        mock_hstack.assert_called_once_with(mock_data)
        mock_save.assert_called_once_with(
            test_filepath,
            mock_hstack.return_value
        )

    @patch("seagrass.utils.np.save")
    @patch("seagrass.utils.np.hstack")
    def test_save_ml_data_npy_prediction(self, mock_hstack, mock_save):
        test_filepath = "test_file.npy"
        mock_data = Mock()
        data_purpose = "prediction"

        utils.save_ml_data_npy(test_filepath, mock_data, data_purpose)

        mock_hstack.assert_not_called()
        mock_save.assert_called_once_with(
            test_filepath,
            mock_data
        )

    @patch("seagrass.utils.np.save")
    @patch("seagrass.utils.np.hstack")
    def test_save_ml_data_npy_kwargs(self, mock_hstack, mock_save):
        test_filepath = "test_file.npy"
        mock_data = Mock()
        data_purpose = "prediction"

        test_kwargs = {
            "testInt": 1,
            "testBool": True,
            "testStr": "test",
            }

        utils.save_ml_data_npy(
            test_filepath, mock_data, data_purpose, **test_kwargs
        )

        mock_save.assert_called_once_with(
            test_filepath,
            mock_data,
            **test_kwargs
        )

    @patch("seagrass.utils.np.save")
    @patch("seagrass.utils.np.hstack")
    def test_save_ml_data_npy_invalid_filepaths(self, mock_hstack, mock_save):
        invalid_filepaths = [
            "test_file.path.py",
            "test_file.path.npy.",
            "test_file.path",
            "test_file",
        ]
        mock_data = Mock()
        data_purpose = "prediction"

        for filepath in invalid_filepaths:
            with self.subTest(invalid_filepath=filepath):
                with self.assertRaises(ValueError):
                    utils.save_ml_data_npy(filepath, mock_data, data_purpose)

    @patch("seagrass.utils.np.save")
    @patch("seagrass.utils.np.hstack")
    def test_save_ml_data_npy_invalid_data_purpose(
        self,
        mock_hstack,
        mock_save
    ):
        filepath = "test_file.path.npy"
        mock_data = Mock()
        invalid_data_purposes = [
            "test",
            "train",
            "predict",
            "valid",
        ]

        for data_purpose in invalid_data_purposes:
            with self.subTest(invalid_data_purpose=data_purpose):
                with self.assertRaises(ValueError):
                    utils.save_ml_data_npy(filepath, mock_data, data_purpose)

    @patch("seagrass.utils.pd.DataFrame")
    @patch("seagrass.utils.np.hstack")
    def test_save_ml_data_csv_training(self, mock_hstack, mock_df):
        test_filepath = "test_file.csv"
        mock_data = Mock()
        data_purpose = "training"

        utils.save_ml_data_csv(test_filepath, mock_data, data_purpose)

        mock_hstack.assert_called_once_with(mock_data)
        mock_df.assert_called_once_with(mock_hstack.return_value, columns=None)
        mock_df.return_value.to_csv.assert_called_once_with(
            test_filepath,
            index=False
        )

    @patch("seagrass.utils.pd.DataFrame")
    @patch("seagrass.utils.np.hstack")
    def test_save_ml_data_csv_prediction(self, mock_hstack, mock_df):
        test_filepath = "test_file.csv"
        mock_data = Mock()
        data_purpose = "prediction"

        utils.save_ml_data_csv(test_filepath, mock_data, data_purpose)

        mock_hstack.assert_not_called()
        mock_df.assert_called_once_with(mock_data, columns=None)
        mock_df.return_value.to_csv.assert_called_once_with(
            test_filepath,
            index=False
        )

    @patch("seagrass.utils.pd.DataFrame")
    @patch("seagrass.utils.np.hstack")
    def test_save_ml_data_csv_kwargs(self, mock_hstack, mock_df):
        test_filepath = "test_file.csv"
        mock_data = Mock()
        data_purpose = "prediction"
        test_cols = ["test1", "test2"]

        test_kwargs = {
            "testInt": 1,
            "testBool": True,
            "testStr": "test",
            }

        utils.save_ml_data_csv(
            test_filepath,
            mock_data,
            data_purpose,
            column_labels=test_cols,
            **test_kwargs
        )

        mock_df.assert_called_once_with(mock_data, columns=test_cols)
        mock_df.return_value.to_csv.assert_called_once_with(
            test_filepath,
            index=False,
            **test_kwargs
        )

    @patch("seagrass.utils.np.save")
    @patch("seagrass.utils.np.hstack")
    def test_save_ml_data_csv_invalid_filepaths(self, mock_hstack, mock_save):
        invalid_filepaths = [
            "test_file.path.py",
            "test_file.path.npy.",
            "test_file.path",
            "test_file",
        ]
        mock_data = Mock()
        data_purpose = "prediction"

        for filepath in invalid_filepaths:
            with self.subTest(invalid_filepath=filepath):
                with self.assertRaises(ValueError):
                    utils.save_ml_data_csv(filepath, mock_data, data_purpose)

    @patch("seagrass.utils.np.save")
    @patch("seagrass.utils.np.hstack")
    def test_save_ml_data_csv_invalid_data_purpose(
        self,
        mock_hstack,
        mock_save
    ):
        filepath = "test_file.path.npy"
        mock_data = Mock()
        invalid_data_purposes = [
            "test",
            "train",
            "predict",
            "valid",
        ]

        for data_purpose in invalid_data_purposes:
            with self.subTest(invalid_data_purpose=data_purpose):
                with self.assertRaises(ValueError):
                    utils.save_ml_data_csv(filepath, mock_data, data_purpose)

    @patch("seagrass.utils.rmtree")
    @patch("seagrass.utils.tarfile")
    @patch("seagrass.utils._make_data_structure_json")
    @patch("seagrass.utils.save_ml_data_csv")
    @patch("seagrass.utils._make_tmp_dir")
    def test_save_ml_data_modulos(
        self,
        mock__make_tmp_dir,
        mock_save_ml_data_csv,
        mock__make_data_structure_json,
        mock_tarfile,
        mock_rmtree
    ):
        test_filepath = "directory/test_file.tar"
        test_directory = "directory"
        test_csv_filepath = "directory/tmp/test_file.csv"
        test_csv_file = "test_file.csv"
        test_json_filepath = "directory/tmp/dataset_structure.json"
        test_json_file = "dataset_structure.json"
        test_data_purpose = "training"
        mock_data = Mock()

        mock_tar = mock_tarfile.open.return_value.__enter__.return_value
        mock__make_tmp_dir.return_value = "directory/tmp"

        utils.save_ml_data_modulos(test_filepath, mock_data, test_data_purpose)

        mock__make_tmp_dir.assert_called_once_with(test_directory)
        mock_save_ml_data_csv.assert_called_once_with(
            test_csv_filepath,  mock_data, test_data_purpose
        )
        mock__make_data_structure_json.assert_called_once_with(
            test_csv_file, test_json_filepath
        )

        mock_tarfile.open.assert_called_once_with(test_filepath, "w")
        mock_tarfile.open.return_value.__enter__.assert_called_once()
        mock_tar.add.assert_has_calls(
            [
                call(test_csv_filepath, arcname=test_csv_file),
                call(test_json_filepath, arcname=test_json_file)
            ]
        )
        mock_tarfile.open.return_value.__exit__.assert_called_once_with(
            None, None, None
        )

    @patch("seagrass.utils.rmtree")
    @patch("seagrass.utils.tarfile")
    @patch("seagrass.utils._make_data_structure_json")
    @patch("seagrass.utils.save_ml_data_csv")
    @patch("seagrass.utils._make_tmp_dir")
    def test_save_ml_data_modulos_delete_tmp_dir(
        self,
        mock__make_tmp_dir,
        mock_save_ml_data_csv,
        mock__make_data_structure_json,
        mock_tarfile,
        mock_rmtree
    ):
        test_filepath = "directory/test_file.tar"
        test_data_purpose = "training"
        mock_data = Mock()

        mock__make_tmp_dir.return_value = "directory/tmp"

        utils.save_ml_data_modulos(test_filepath, mock_data, test_data_purpose)

        mock_rmtree.assert_called_once_with("directory/tmp")

    @patch("seagrass.utils.rmtree")
    @patch("seagrass.utils.tarfile")
    @patch("seagrass.utils._make_data_structure_json")
    @patch("seagrass.utils.save_ml_data_csv")
    @patch("seagrass.utils._make_tmp_dir")
    def test_save_ml_data_modulos_kwargs(
        self,
        mock__make_tmp_dir,
        mock_save_ml_data_csv,
        mock__make_data_structure_json,
        mock_tarfile,
        mock_rmtree
    ):
        test_filepath = "directory/test_file.tar"
        test_data_purpose = "training"
        mock_data = Mock()

        mock__make_tmp_dir.return_value = "directory/tmp"

        test_kwargs = {
            "testInt": 1,
            "testBool": True,
            "testStr": "test",
            }

        utils.save_ml_data_modulos(
            test_filepath,
            mock_data,
            test_data_purpose,
            **test_kwargs
        )

        mock_save_ml_data_csv.assert_called_once_with(
            "directory/tmp/test_file.csv",
            mock_data,
            test_data_purpose,
            **test_kwargs
            )

        mock_rmtree.assert_called_once_with("directory/tmp")

    @patch("seagrass.utils.rmtree")
    @patch("seagrass.utils.tarfile")
    @patch("seagrass.utils._make_data_structure_json")
    @patch("seagrass.utils.save_ml_data_csv")
    @patch("seagrass.utils._make_tmp_dir")
    def test_save_ml_data_modulos_invalid_filepaths(
        self,
        mock__make_tmp_dir,
        mock_save_ml_data_csv,
        mock__make_data_structure_json,
        mock_tarfile,
        mock_rmtree
    ):
        invalid_filepaths = [
            "test_file.path.py",
            "test_file.path.npy.",
            "test_file.path",
            "test_file",
        ]
        mock_data = Mock()
        data_purpose = "training"

        for filepath in invalid_filepaths:
            with self.subTest(invalid_filepath=filepath):
                with self.assertRaises(ValueError):
                    utils.save_ml_data_modulos(
                        filepath, mock_data, data_purpose
                    )

    @patch("seagrass.utils.rmtree")
    @patch("seagrass.utils.tarfile")
    @patch("seagrass.utils._make_data_structure_json")
    @patch("seagrass.utils.save_ml_data_csv")
    @patch("seagrass.utils._make_tmp_dir")
    def test_save_ml_data_modulos_data_purpose(
        self,
        mock__make_tmp_dir,
        mock_save_ml_data_csv,
        mock__make_data_structure_json,
        mock_tarfile,
        mock_rmtree
    ):
        filepath = "test_file.path.tar"
        mock_data = Mock()
        invalid_data_purposes = [
            "test",
            "train",
            "predict",
            "valid",
        ]

        for data_purpose in invalid_data_purposes:
            with self.subTest(invalid_data_purpose=data_purpose):
                with self.assertRaises(ValueError):
                    utils.save_ml_data_modulos(
                        filepath, mock_data, data_purpose
                    )

    @patch("seagrass.utils.json")
    @patch("builtins.open")
    def test__make_data_structure_json(self, mock_open, mock_json):
        expected_data_structure = [
            {
                "type": "table",
                "path": "test_file.csv",
                "name": "test_file"
            },
            {
                "_version": "0.2"
            }
        ]

        test_csv_file = "test_file.csv"
        test_json_filepath = "directory/tmp/dataset_structure.json"
        utils._make_data_structure_json(test_csv_file, test_json_filepath)

        mock_open.assert_called_once_with(test_json_filepath, "w")
        mock_open.return_value.__enter__.assert_called_once()
        mock_json.dump.assert_called_once_with(
            expected_data_structure,
            mock_open.return_value.__enter__.return_value,
            indent=4
        )
        mock_open.return_value.__exit__.assert_called_once_with(
            None, None, None
        )

    @patch("seagrass.utils.os")
    def test__make_tmp_dir(self, mock_os):
        test_directory = "path/to/directory"
        tmp_dir = utils._make_tmp_dir(test_directory)

        self.assertEqual(tmp_dir, "path/to/directory/tmp")

    @patch("seagrass.utils.os")
    def test__make_tmp_dir_pwd(self, mock_os):
        test_directory = ""
        tmp_dir = utils._make_tmp_dir(test_directory)

        self.assertEqual(tmp_dir, "./tmp")

    @patch("seagrass.utils.os")
    def test__make_tmp_dir_mkdir(self, mock_os):
        mock_os.path.exists.return_value = True
        mock_os.path.isdir.return_value = False

        test_directory = "path/to/directory"
        utils._make_tmp_dir(test_directory)

        mock_os.path.exists.assert_called_once_with("path/to/directory/tmp")
        mock_os.path.isdir.assert_called_once_with("path/to/directory/tmp")
        mock_os.mkdir.assert_called_once_with("path/to/directory/tmp")

    @patch("seagrass.utils.extract_training_data_csv")
    @patch("seagrass.utils.extract_training_data_npy")
    def test_extract_training_data(
        self,
        mock_extract_training_data_npy,
        mock_extract_training_data_csv
    ):
        mock_X = Mock()
        mock_y = Mock()

        mock_extract_training_data_npy.return_value = (mock_X, mock_y)
        mock_extract_training_data_csv.return_value = (mock_X, mock_y)

        test_filepath_dict = {
            "test_file.path.npy": mock_extract_training_data_npy,
            "test_file.path.csv": mock_extract_training_data_csv,
        }

        for filepath, mock_function in test_filepath_dict.items():
            X, y = utils.extract_training_data(filepath)

            with self.subTest(filepath=filepath):
                mock_function.assert_called_once_with(
                    filepath,
                )

                self.assertEqual(X, mock_X)
                self.assertEqual(y, mock_y)

    @patch("seagrass.utils.extract_training_data_csv")
    @patch("seagrass.utils.extract_training_data_npy")
    def test_extract_training_data_invalid_filetypes(
        self,
        mock_extract_training_data_npy,
        mock_extract_training_data_csv
    ):

        invalid_filepaths = [
            "test_file.path.py",
            "test_file.path.npy.",
            "test_file.path",
            "test_file",
        ]

        for filepath in invalid_filepaths:
            with self.subTest(invalid_filepath=filepath):
                with self.assertRaises(ValueError):
                    utils.extract_training_data(filepath)

    @patch("seagrass.utils.np")
    def test_extract_training_data_npy(self, mock_np):
        test_filepath = "path/to/file"
        mock_np.load.return_value = np.array(
            (
                [1, 4, 7],
                [2, 5, 8],
                [3, 6, 9]
            )
        )

        expected_X = np.array(
            (
                [1, 4],
                [2, 5],
                [3, 6]
            )
        )

        expected_y = np.array(
            (
                [7],
                [8],
                [9]
            )
        )

        X, y = utils.extract_training_data_npy(test_filepath)

        assert_array_equal(X, expected_X)
        assert_array_equal(y, expected_y)

    @patch("seagrass.utils.pd")
    def test_extract_training_data_csv(self, mock_pd):
        expected_X = np.array(
            (
                [1, 4],
                [2, 5],
                [3, 6]
            )
        )

        expected_y = np.array(
            (
                [7],
                [8],
                [9]
            )
        )

        test_data = np.array(
            (
                [1, 4, 7],
                [2, 5, 8],
                [3, 6, 9]
            )
        )

        mock_pd.read_csv.return_value = pd.DataFrame(test_data)

        test_filepath = "path/to/file"
        X, y = utils.extract_training_data_csv(test_filepath)

        assert_array_equal(X, expected_X)
        assert_array_equal(y, expected_y)

    @patch("seagrass.utils.json")
    @patch("builtins.open")
    def test_make_json(self, mock_open, mock_json):
        test_json_filepath = "path/to/json"
        test_raster_filepath = "path/to/s2"
        test_ground_filepath = "path/to/ground/truth"
        test_raster_bands = [0, 1, 2, 3]
        test_raster_scale = 10000
        test_ground_truth_nodata = -9999
        test_ground_truth_nodata_threshold = None

        expected_output_dict = {
            "raster_filepath": test_raster_filepath,
            "ground_truth_filepath": test_ground_filepath,
            "raster_bands": test_raster_bands,
            "raster_scale": test_raster_scale,
            "ground_truth_nodata": test_ground_truth_nodata,
            "ground_truth_nodata_threshold": test_ground_truth_nodata_threshold
        }

        utils.make_json(
            test_json_filepath,
            test_raster_filepath,
            test_ground_filepath,
            test_raster_bands,
            test_raster_scale,
            test_ground_truth_nodata,
            test_ground_truth_nodata_threshold
        )

        mock_open.assert_called_once_with(test_json_filepath, "w")
        mock_open.return_value.__enter__.assert_called_once()
        mock_json.dump.assert_called_once_with(
            expected_output_dict,
            mock_open.return_value.__enter__.return_value,
            indent=4
        )
        mock_open.return_value.__exit__.assert_called_once_with(
            None, None, None
        )

    @patch("seagrass.utils.make_geocube")
    @patch("seagrass.utils.gpd")
    def test_shape_to_binary_raster(self, mock_gpd, mock_make_geocube):
        test_shp_filepath = "path/to/shp"
        test_out_dir = "path/to/out/dir"

        mock_geo_df = mock_gpd.read_file.return_value
        mock_geocube = mock_make_geocube.return_value

        utils.shape_to_binary_raster(test_shp_filepath, test_out_dir)
        print(mock_geocube.mock_calls)

        mock_gpd.read_file.assert_called_once_with(test_shp_filepath)
        mock_make_geocube.assert_called_once_with(
            vector_data=mock_geo_df,
            measurements=["data"],
            resolution=(-10, 10),
            fill=0,
        )

        mock_geocube.__getitem__.assert_called_once_with('data')
        mock_geocube.__getitem__.return_value.rio.to_raster.assert_called_once_with("path/to/out/dir/shp.tif")  # noqa: E501
