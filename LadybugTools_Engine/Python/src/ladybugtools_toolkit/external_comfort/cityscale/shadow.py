import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def shadow(asc_file: Path, sun_altitude: float, sun_azimuth: float) -> np.ndarray:
    """Create a matrix of pixel values for the given topographic TIF-style image corresponding to
        the shadows cast by the sun at the given location.

    Args:
        asc_file (Path):
            The path to a TIF-style file containing topographic height data.
        sun_altitude (float):
            The altitude of the sun to cast shadows.
        sun_azimuth (float):
            The azimuth of the sun to cast shadows.

    Returns:
        np.ndarray:
            A pixel matrix from 0-1 representing proportion of sunlight received.
    """

    # set globals
    OSGEO4W_PATH = Path(r"C:\Program Files\QGIS 3.20.2\OSGeo4W.bat")

    input_file = Path(asc_file)
    output_file = (
        input_file.parent / f"{input_file.stem}_{sun_altitude}_{sun_azimuth}.tif"
    )

    # create command to generate shadow plot
    # https://gdal.org/programs/gdaldem.html
    cmd = f'gdaldem hillshade "{asc_file}" "{output_file}" -of GTiff -b 1 -z 40.0 -s 1.0 -az {sun_azimuth} -alt {sun_altitude} -alg Horn -compute_edges -combined'
    subprocess.Popen(
        f'"{OSGEO4W_PATH}" {cmd}', shell=True, stdout=subprocess.PIPE
    ).stdout.read()
    shadow_matrix = np.array(Image.open(output_file.as_posix())) / 255

    # delete shadow plot
    os.remove(output_file.as_posix())

    return np.where(shadow_matrix == 0, 1, shadow_matrix)
