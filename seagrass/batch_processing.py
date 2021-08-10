import numpy as np

from seagrass.mosaic import intersecting_tiles
from seagrass.raster import open_and_match_rasters
from seagrass.prepare import create_training_data


def create_training_data_batch(
    raster_file_list,
    ground_truth_filepath,
    mask=None,
    no_data_value=None,
    raster_bands=None
):

    intersecting_raster_list = intersecting_tiles(
        raster_file_list,
        ground_truth_filepath
    )

    for idx, raster_filepath in enumerate(intersecting_raster_list):
        raster, ground_truth = open_and_match_rasters(
            raster_filepath,
            ground_truth_filepath
        )

        if mask:
            ground_truth = ground_truth.where(
                mask(raster) == False,  # noqa: E712
                no_data_value
            )

        if idx == 0:
            features, targets = create_training_data(
                raster.values,
                ground_truth.values,
                no_data_value=no_data_value,
                s2_bands=raster_bands
            )

        else:
            X, y = create_training_data(
                raster.values,
                ground_truth.values,
                no_data_value=no_data_value,
                s2_bands=raster_bands
            )

            features = np.vstack([features, X])
            targets = np.vstack([targets, y])

    return features, targets
