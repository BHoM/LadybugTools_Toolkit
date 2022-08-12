import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

DIR = Path(__file__).parent


def shadow(
    tif_file: Path,
    date_time: datetime,
    output_file: Path = None,
) -> Path:
    """Create a point-in-time radiation plot for a given tif/tiff terrain elevation file.

    Args:
        tif_file (Path):
            A raster elevation file.
        date_time (datetime):
            A datetime at which to run the calculation.
        output_file (Path, optional):
            The name of the file to be created. Defaults to the input file, with the day-hour appended.

    Returns:
        Path:
            The path to the resultant shadow file.
    """

    tif_file = Path(tif_file)

    run_command = f'"{(DIR / "_shadow.py").as_posix()}" "{tif_file.as_posix()}" {date_time.timetuple().tm_yday} {int(date_time.hour + date_time.minute / 60)}'
    if output_file is not None:
        output_file = Path(output_file)
        if output_file.suffix != ".tif":
            raise ValueError("output_file must have extension *.tif")
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
