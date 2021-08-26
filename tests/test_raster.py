import unittest
from unittest.mock import patch, Mock
from xarray.testing import assert_equal

from seagrass import raster

import numpy as np
import xarray as xa


class TestRaster(unittest.TestCase):

    @patch("seagrass.raster.open_and_match_rasters")
    @patch("seagrass.raster.json")
    @patch("builtins.open")
    def test_open_from_json(
        self,
        mock_open,
        mock_json,
        mock_open_and_match_rasters
    ):
        test_json_filepath = "path/to/json"
        mock_json.load.return_value = {
            "raster_filepath": "path/to/raster",
            "ground_truth_filepath": "path/to/ground/truth",
            "raster_bands": [0, 1, 2, 3],
            "raster_scale": 10000,
            "ground_truth_nodata": -9999,
            "ground_truth_nodata_threshold": -1e6
        }

        mock_raster = Mock()
        mock_ground_truth = Mock()

        mock_open_and_match_rasters.return_value = (
            mock_raster, mock_ground_truth
        )

        raster_image, ground_truth = raster.open_from_json(test_json_filepath)

        mock_open.assert_called_once_with(test_json_filepath)
        mock_open.return_value.__enter__.assert_called_once()
        mock_json.load.assert_called_once_with(
            mock_open.return_value.__enter__.return_value
        )
        mock_open_and_match_rasters.assert_called_once_with(
            "path/to/raster",
            "path/to/ground/truth",
            10000
        )
        mock_open.return_value.__exit__.assert_called_once_with(
            None, None, None
        )

        self.assertEqual(raster_image, mock_raster)
        self.assertEqual(ground_truth, mock_ground_truth)

    @patch("seagrass.raster.open_and_match_rasters_mosaic")
    @patch("seagrass.raster.json")
    @patch("builtins.open")
    def test_open_from_json_mosaic(
        self,
        mock_open,
        mock_json,
        mock_open_and_match_rasters_mosaic
    ):
        test_json_filepath = "path/to/json"
        mock_json.load.return_value = {
            "raster_filepath": ["path/to/raster/1", "path/to/raster/2"],
            "ground_truth_filepath": "path/to/ground/truth",
            "raster_bands": [0, 1, 2, 3],
            "raster_scale": 10000,
            "ground_truth_nodata": -9999,
            "ground_truth_nodata_threshold": -1e6
        }

        mock_open_and_match_rasters_mosaic.return_value = (Mock(), Mock())

        raster_image, ground_truth = raster.open_from_json(test_json_filepath)

        mock_open_and_match_rasters_mosaic.assert_called_once_with(
            ["path/to/raster/1", "path/to/raster/2"],
            "path/to/ground/truth",
            [0, 1, 2, 3],
            10000,
            -9999,
            -1e6
        )

    @patch("seagrass.raster.open_and_match_rasters")
    @patch("seagrass.raster.json")
    @patch("builtins.open")
    def test_open_from_json_list_one_item(
        self,
        mock_open,
        mock_json,
        mock_open_and_match_rasters
    ):
        test_json_filepath = "path/to/json"
        mock_json.load.return_value = {
            "raster_filepath": ["path/to/raster"],
            "ground_truth_filepath": "path/to/ground/truth",
            "raster_bands": [0, 1, 2, 3],
            "raster_scale": 10000,
            "ground_truth_nodata": -9999,
            "ground_truth_nodata_threshold": -1e6
        }

        mock_open_and_match_rasters.return_value = (Mock(), Mock())

        raster_image, ground_truth = raster.open_from_json(test_json_filepath)

        mock_open_and_match_rasters.assert_called_once_with(
            "path/to/raster",
            "path/to/ground/truth",
            10000
        )

    @patch("seagrass.raster.open_and_match_rasters_mosaic")
    @patch("seagrass.raster.json")
    @patch("builtins.open")
    def test_open_from_json_list_dict_defaults(
        self,
        mock_open,
        mock_json,
        mock_open_and_match_rasters_mosaic
    ):
        test_json_filepath = "path/to/json"
        mock_json.load.return_value = {
            "raster_filepath": ["path/to/raster/1", "path/to/raster/2"],
            "ground_truth_filepath": "path/to/ground/truth",
        }

        mock_open_and_match_rasters_mosaic.return_value = (Mock(), Mock())

        raster_image, ground_truth = raster.open_from_json(test_json_filepath)

        mock_open_and_match_rasters_mosaic.assert_called_once_with(
            ["path/to/raster/1", "path/to/raster/2"],
            "path/to/ground/truth",
            None,
            1,
            None,
            None
        )

    @patch("seagrass.raster.return_mosaic_projected_ground_truth")
    @patch("seagrass.raster.create_raster_mosaic")
    def test_open_and_match_rasters_mosaic(
        self,
        mock_create_raster_mosaic,
        mock_return_mosaic_projected_ground_truth
    ):
        test_raster_filepath_list = ["path/to/raster/1", "path/to/raster/2"]
        test_ground_truth_filepath = "path/to/ground/truth"
        test_raster_bands = [0, 1, 2, 3]
        test_raster_scale = 10000
        test_ground_truth_nodata = -9999
        test_ground_truth_nodata_threshold = -1e6

        mock_raster_mosaic = Mock()
        mock_raster_transform = Mock()
        mock_ground_truth = Mock()

        mock_create_raster_mosaic.return_value = (
            mock_raster_mosaic, mock_raster_transform
        )
        mock_return_mosaic_projected_ground_truth.return_value = mock_ground_truth  # noqa: E501

        mosaic, ground_truth = raster.open_and_match_rasters_mosaic(
            test_raster_filepath_list,
            test_ground_truth_filepath,
            test_raster_bands,
            test_raster_scale,
            test_ground_truth_nodata,
            test_ground_truth_nodata_threshold
        )

        mock_create_raster_mosaic.assert_called_once_with(
            test_raster_filepath_list,
            test_ground_truth_filepath,
            test_raster_bands,
            test_raster_scale
        )

        mock_return_mosaic_projected_ground_truth.assert_called_once_with(
            test_ground_truth_filepath,
            mock_raster_transform,
            mock_raster_mosaic.shape,
            test_ground_truth_nodata,
            test_ground_truth_nodata_threshold
        )

        self.assertEqual(mosaic, mock_raster_mosaic)
        self.assertEqual(ground_truth, mock_ground_truth)

    @patch("seagrass.raster.rioxarray")
    def test_open_raster_image(self, mock_rioxarray):
        test_filepath = "path/to/raster"
        test_scale = 10000
        test_data = np.array(
            (
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            )
        )

        expected_raster = xa.DataArray(test_data)
        mock_rioxarray.open_rasterio.return_value = expected_raster

        out_raster = raster.open_raster_image(test_filepath, test_scale)

        mock_rioxarray.open_rasterio.assert_called_once_with(test_filepath)
        assert_equal(out_raster, expected_raster/10000)

    @patch("seagrass.raster.rioxarray")
    @patch("seagrass.raster.open_raster_image")
    def test_open_and_match_rasters(
        self,
        mock_open_raster_image,
        mock_rioxarray
    ):
        test_raster_filepath = "path/to/raster"
        test_ground_truth_filepath = "path/to/ground/truth"
        test_raster_scale = 10000

        mock_raster = mock_open_raster_image.return_value
        mock_ground_truth = mock_rioxarray.open_rasterio.return_value
        mock_reprojected_ground_truth = mock_ground_truth.rio.reproject_match.return_value  # noqa: E501

        raster_image, ground_truth = raster.open_and_match_rasters(
            test_raster_filepath,
            test_ground_truth_filepath,
            test_raster_scale
        )

        mock_open_raster_image.assert_called_once_with(
            test_raster_filepath,
            test_raster_scale
        )
        mock_rioxarray.open_rasterio.assert_called_once_with(
            test_ground_truth_filepath
        )
        mock_ground_truth.rio.reproject_match.assert_called_once_with(
            mock_raster
        )

        self.assertEqual(raster_image, mock_raster)
        self.assertEqual(ground_truth, mock_reprojected_ground_truth)
