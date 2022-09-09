# setup python environment within this script to reference QGIS
import argparse
import os
import tempfile
from pathlib import Path

from qgis.core import QgsApplication, QgsProcessingFeedback
from qgis.gui import *
from qgis.PyQt.QtWidgets import *

qgs = QgsApplication([], False)
QgsApplication.setPrefixPath(os.environ.get("QGIS_PREFIX"), True)
QgsApplication.initQgis()
for alg in QgsApplication.processingRegistry().algorithms():
    print(alg.id(), "->", alg.displayName())

import processing
from processing.core.Processing import Processing

Processing.initialize()
feedback = QgsProcessingFeedback()


def _aspect(
    input_file: Path,
) -> Path:
    """Use QGIS to create an "aspect" file containing terrain orientation."""

    input_file = Path(input_file)
    output_file = Path(tempfile.gettempdir()) / f"{input_file.stem}_aspect.tif"

    if output_file.exists():
        return output_file

    proc = processing.run(
        "native:aspect",
        {
            "INPUT": input_file.as_posix(),
            "Z_FACTOR": 1,
            "OUTPUT": output_file.as_posix(),
        },
        feedback=feedback,
    )
    return Path(proc["OUTPUT"])


def _slope(input_file: Path) -> Path:
    """Use QGIS to create an "slope" file containing terrain slope."""

    input_file = Path(input_file)
    output_file = Path(tempfile.gettempdir()) / f"{input_file.stem}_slope.tif"

    if output_file.exists():
        return output_file

    # print('Calculating terrain "slope"')
    proc = processing.run(
        "native:slope",
        {
            "INPUT": input_file.as_posix(),
            "Z_FACTOR": 1,
            "OUTPUT": output_file.as_posix(),
        },
        feedback=feedback,
    )
    return Path(proc["OUTPUT"])


def _shadow(
    input_file: Path,
    day_of_year: int,
    hour_of_day: int,
    output_file: Path = None,
) -> Path:

    if (hour_of_day > 24) or (hour_of_day < 0) or (not isinstance(hour_of_day, int)):
        raise ValueError("hour_of_day must be and integer between 0 and 24.")

    if (day_of_year > 365) or (day_of_year < 0) or (not isinstance(day_of_year, int)):
        raise ValueError("day_of_year must be an integer between 0 and 365.")

    input_file = Path(input_file)
    if output_file is None:
        output_file = (
            input_file.parent
            / f"{input_file.stem}_shadow{day_of_year:03d}{hour_of_day:02d}.tif"
        )
    else:
        output_file = Path(output_file)

    if output_file.exists():
        # print(f'Shadow mask already exists in {output_file.stem}')
        return output_file

    aspect_file = _aspect(input_file)
    slope_file = _slope(input_file)

    # print(f'Calculating shadow mask in {output_file.stem}')
    proc = processing.run(
        "grass7:r.sun.incidout",
        {
            "elevation": input_file.as_posix(),
            "aspect": aspect_file.as_posix(),
            "aspect_value": 270,
            "slope": slope_file.as_posix(),
            "slope_value": 0,
            "linke": None,
            "albedo": None,
            "albedo_value": 0.2,
            "lat": None,
            "long": None,
            "coeff_bh": None,
            "coeff_dh": None,
            "horizon_basemap": None,
            "horizon_step": None,
            "day": day_of_year,
            "step": 0.5,
            "declination": None,
            "distance_step": 1,
            "npartitions": 1,
            "civil_time": None,
            "time": hour_of_day,
            "-p": False,
            "-m": False,
            "incidout": "TEMPORARY_OUTPUT",
            "beam_rad": "TEMPORARY_OUTPUT",
            "diff_rad": "TEMPORARY_OUTPUT",
            "refl_rad": "TEMPORARY_OUTPUT",
            "glob_rad": output_file.as_posix(),
            "GRASS_REGION_PARAMETER": None,
            "GRASS_REGION_CELLSIZE_PARAMETER": 0,
            "GRASS_RASTER_FORMAT_OPT": "",
            "GRASS_RASTER_FORMAT_META": "",
        },
        feedback=feedback,
    )
    return Path(proc["glob_rad"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a GeoTIFF terrain elevation file to get point-in-time shadows."
    )
    parser.add_argument(
        "input_file", help="The TIF/TIFF terrain file to calculate shadows for."
    )
    parser.add_argument(
        "day_of_year",
        help="The day of year to calculate shadow for.",
        type=int,
    )
    parser.add_argument(
        "hour_of_day",
        help="The hour of day to calculate shadow for.",
        type=int,
    )
    parser.add_argument(
        "--output_file",
        help="The TIF/TIFF terrain file to output the shadow mask into.",
        type=str,
        default=None,
        required=False,
    )

    args = parser.parse_args()

    print(
        _shadow(
            args.input_file,
            args.day_of_year,
            args.hour_of_day,
            args.output_file,
        )
    )
