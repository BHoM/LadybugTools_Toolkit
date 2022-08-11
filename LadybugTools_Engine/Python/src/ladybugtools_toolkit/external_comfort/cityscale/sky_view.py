import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def sky_view(
    asc_file: Path,
    n_directions: int = 16,
    view_radius: float = 5000,
) -> np.ndarray:
    """Calculate the sky-view-factor from the given topographic TIF-style image.

    Args:
        asc_file (Path):
            The path to a TIF-style file containing topographic height data.
        n_directions (int, optional):
            The number of directions in which to check for overshadowing elements. Defaults to 16.
        view_radius (float, optional):
            Teh radius around each pixel to check for overshadowing. Defaults to 5000.

    Returns:
        np.ndarray:
            A pixel matrix, from 0-1 representing proportion of sky visible.
    """

    # globals for this process
    SAGA = "C:/Program Files/QGIS 3.20.2/apps/saga"
    SAGA_MLB = "C:/Program Files/QGIS 3.20.2/apps/saga/modules"
    PATH = ";".join(
        [
            "C:/Program Files/QGIS 3.20.2/apps/qgis/bin",
            "C:/Program Files/QGIS 3.20.2/apps/grass/grass78/lib",
            "C:/Program Files/QGIS 3.20.2/apps/grass/grass78/bin",
            "C:/Program Files/QGIS 3.20.2/apps/qt5/bin",
            "C:/Program Files/QGIS 3.20.2/apps/Python39/Scripts",
            "C:/Program Files/QGIS 3.20.2/bin",
            "C:/WINDOWS/system32",
            "C:/WINDOWS",
            "C:/WINDOWS/system32/WBem",
            "C:/Program Files/QGIS 3.20.2/apps/saga",
            "C:/Program Files/QGIS 3.20.2/apps/saga/modules",
        ]
    )

    input_file = Path(asc_file)
    grid_file = input_file.with_suffix(".sgrd")
    output_file = input_file.parent / f"{input_file.stem}_SkyViewFactor.tif"

    if not output_file.exists():

        commands = " && ".join(
            [
                f"set SAGA={SAGA}",
                f"set SAGA_MLB={SAGA_MLB}",
                f"PATH={PATH}",
                f'call saga_cmd io_gdal 0 -TRANSFORM 1 -RESAMPLING 3 -GRIDS "{grid_file}" -FILES "{input_file}"',
                f'call saga_cmd ta_lighting "Sky View Factor" -DEM "{grid_file}" -RADIUS {view_radius} -NDIRS {n_directions} -METHOD 0 -DLEVEL 1.25 -SVF "{output_file}"',
            ]
        )

        # run commands
        subprocess.Popen(commands, shell=True, stdout=subprocess.PIPE).stdout.read()

        # remove files generated as part of this process
        os.remove(grid_file.as_posix())
        os.remove(input_file.with_suffix(".mgrd").as_posix())
        os.remove(input_file.with_suffix(".sdat").as_posix())
        os.remove(input_file.with_suffix(".prj").as_posix())

    # load resultant file into np array
    sky_view_matrix = np.array(Image.open(output_file.as_posix()))
    sky_view_matrix = np.interp(
        sky_view_matrix, [sky_view_matrix.min(), sky_view_matrix.max()], [0, 1]
    )

    return np.where(sky_view_matrix == 0, 1, sky_view_matrix)
