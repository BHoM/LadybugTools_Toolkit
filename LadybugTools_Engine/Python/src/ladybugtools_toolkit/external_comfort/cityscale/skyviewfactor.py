import subprocess
import tempfile
import uuid
from pathlib import Path

DIR = Path(__file__).parent


def skyviewfactor(
    tif_file: Path,
    output_file: Path = None,
    directions: int = 8,
    radius: float = 2000,
) -> Path:
    """Create a sky-view-factor tif file.

    Args:
        tif_file (Path):
            A raster elevation file.
        output_file (Path, optional):
            The name of the file to be created. Defaults to the input file, with "_skyviewfactor".
        directions (int, optional):
            The number of directions in which to calculate sky-view. Defaults to 8.
        radius (float, optional):
            The distance from each pixel to account for in sky-view. Defaults to 2000.

    Returns:
        Path:
            The path to the resultant TIF sky-view-factor file.

    """

    tif_file = Path(tif_file)

    run_command = f'"{(DIR / "_skyviewfactor.py").as_posix()}" "{tif_file.as_posix()}" --directions {directions} --radius {radius}'
    if output_file is not None:
        if output_file.suffix != ".tif":
            raise ValueError("output_file must have extension *.tif")
        output_file = Path(output_file)
        run_command += f' --output_file "{output_file.as_posix()}"'

    commands = [
        "@echo off",
        r'SET OSGEO4W_ROOT="C:\OSGeo4W"',
        r"call %OSGEO4W_ROOT%\bin\o4w_env.bat",
        "",
        r"path %PATH%;%OSGEO4W_ROOT%\apps\qgis\bin",
        r"path %PATH%;%OSGEO4W_ROOT%\apps\Qt5\bin",
        r"path %PATH%;%OSGEO4W_ROOT%\apps\Python39\Scripts",
        "",
        r"set QGIS_PREFIX=%OSGEO4W_ROOT%\apps\qgis",
        r"set PYTHONHOME=%OSGEO4W_ROOT%\apps\Python39",
        r"set PYTHONPATH=%OSGEO4W_ROOT%\apps\qgis\python;%OSGEO4W_ROOT%\apps\qgis\python\plugins;%PYTHONPATH%",
        r"set QT_QPA_PLATFORM_PLUGIN_PATH=%OSGEO4W_ROOT%\apps\Qt5\plugins\platforms",
        r"set QT_PLUGIN_PATH=%OSGEO4W_ROOT%\apps\qgis\qtplugins;%OSGEO4W_ROOT\apps\qt5\plugins",
        "",
        "set GDAL_FILENAME_IS_UTF8=YES",
        "set VSI_CACHE=TRUE",
        "set VSI_CACHE_SIZE=1000000",
        "",
        f"%PYTHONHOME%\\python.exe {run_command}",
    ]

    # write commands to bat file and run
    bat_temp = Path(tempfile.gettempdir()) / f"{str(uuid.uuid4())}.bat"
    with open(bat_temp, "w") as fp:
        fp.writelines("\n".join(commands))

    out = subprocess.Popen(
        bat_temp.as_posix(), shell=True, stdout=subprocess.PIPE
    ).stdout.read()

    return Path(out.decode("ascii").strip())
