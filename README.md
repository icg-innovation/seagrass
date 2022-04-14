# seagrass

A companion Python module (including Colab notebooks) for the University of Portsmouth seagrass project.

This package is still in development and is *highly* experimental, therefore the API is subject to frequent and significant change. Use with caution.

NOTE: This package currently holds no license and therefore cannot be copied, distributed, modified, or used for research outside the University of Portsmouth without express permission from the authors.

## Installation
To install the `seagrass` Python package, run the following command from your `bash` terminal:
```
pip install git+https://github.com/Max-FM/seagrass.git
```
or alternatively from the cloned package directory run:
```
pip install .
```
## Documentation (work in progress)
To generate html documentation pages locally, run the following command in your `bash` terminal (from the cloned package directory):
```
./generate_docs.sh
```
The documentation pages can then be found under `docs/build/html`.

## Import JSON structure (experimental)
If importing data using the `seagrass.raster.open_from_json` method, the input json file needs to abide by the following example structure:
```
{
    "raster_filepath": "path/to/s2/data", OR ["path/to/s2/data_1", "path/to/s2/data_2", etc],
    "ground_truth_filepath": "path/to/ground/truth/data",
    "raster_bands": [1, 2, 3, 4, etc],
    "raster_scale": 10000,
    "ground_truth_nodata": -9999,
    "ground_truth_nodata_threshold": -1e6
}
```

Currently accepted arguments are:

- `raster_filepath` (str or list, required): Satellite image geoTIFF filepath OR list of satellite image geoTIFF filepaths if creating a mosaic.
- `ground_truth_filepath` (str, required): Ground truth geoTIFF filepath.
- `raster_bands` (list, optional): List of integers corresponding to the desired satellite image bands to consider when importing. If not included then all bands are considered.
- `raster_scale` (int, optional): Value to divide the satellite image pixels by to obtain the true pixel values. Defaults to 10000 if not included.
- `ground_truth_nodata` (int, optional): Integer value representing pixels containing no data. Defaults to None if not included.
- `ground_truth_nodata_threshold` (int or float, optional): Pixels with values less than the threshold will instead be set equal to `ground_truth_nodata`. Defaults to None if not included.
