"""Root for the bhom subpackage."""

from pathlib import Path  # pylint: disable=E0401

from win32api import HIWORD, LOWORD, GetFileVersionInfo

BHOM_ASSEMBLIES_DIRECTORY = Path("C:/ProgramData/BHoM/Assemblies")
BHOM_DIRECTORY = Path("C:/ProgramData/BHoM")
BHOM_LOG_FOLDER = Path("C:/ProgramData/BHoM/Logs")

PYTHON_CODE_DIRECTORY = Path("C:/ProgramData/BHoM/Extensions/PythonCode")
PYTHON_ENVIRONMENTS_DIRECTORY = Path(
    "C:/ProgramData/BHoM/Extensions/PythonEnvironments"
)

TOOLKIT_NAME = "LadybugTools_Toolkit"

_file_version_ms = GetFileVersionInfo(
    (BHOM_ASSEMBLIES_DIRECTORY / "BHoM.dll").as_posix(), "\\"
)["FileVersionMS"]

BHOM_VERSION = f"{HIWORD(_file_version_ms)}.{LOWORD(_file_version_ms)}"
