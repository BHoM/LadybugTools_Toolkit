import argparse
import os
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


def _skyviewfactor(
    input_file: Path,
    output_file: Path = None,
    directions: int = 8,
    radius: float = 2000,
) -> Path:
    """Use QGIS to create a "sky-view" file containing terrain sky view factor."""
    input_file = Path(input_file)

    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_skyviewfactor.tif"
    else:
        output_file = Path(output_file)

    if output_file.exists():
        return output_file

    proc = processing.run(
        "saga:skyviewfactor",
        {
            "DEM": input_file.as_posix(),
            "VISIBLE": "TEMPORARY_OUTPUT",
            "SVF": output_file.as_posix(),
            "SIMPLE": "TEMPORARY_OUTPUT",
            "TERRAIN": "TEMPORARY_OUTPUT",
            "DISTANCE": "TEMPORARY_OUTPUT",
            "RADIUS": radius,
            "NDIRS": directions,
            "METHOD": 0,
            "DLEVEL": 3,
        },
        feedback=feedback,
    )
    return Path(proc["SVF"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a GeoTIFF terrain elevation file to get a sky view factor."
    )
    parser.add_argument(
        "input_file", help="The TIF/TIFF terrain file to calculate sky view factor for."
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="The TIF/TIFF terrain file to output the sky view factor into.",
        required=False,
    )
    parser.add_argument(
        "--directions",
        type=int,
        default=8,
        help="The number of directions to calculate sky view factor for.",
        required=False,
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2000.0,
        help="The distance to include in sky view factor calculation.",
        required=False,
    )
    args = parser.parse_args()

    print(
        _skyviewfactor(args.input_file, args.output_file, args.directions, args.radius)
    )
