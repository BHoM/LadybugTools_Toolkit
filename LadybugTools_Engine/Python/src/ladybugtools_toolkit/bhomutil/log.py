import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from win32api import HIWORD, LOWORD, GetFileVersionInfo


def _ui_name() -> str:
    """Return the name of the current Python BHoM toolkit."""
    return "LadybugTools_Toolkit"


def _bhom_dir() -> Path:
    """Return the root BHoM directory as a Path object."""
    return Path(r"C:\ProgramData\BHoM")


def _log_folder() -> Path:
    """Return the BHoM Analytics logging directory as a Path object (and create it if it doesnt
    already exist)."""
    log_directory = _bhom_dir() / "Logs"
    log_directory.mkdir(exist_ok=True, parents=True)
    return log_directory


def _bhom_version() -> str:
    """Return the version of BHoM installed (using the BHoM.dll in the root BHoM directory."""
    bhom_dll: Path = _bhom_dir() / "Assemblies" / "BHoM.dll"
    info = GetFileVersionInfo(
        bhom_dll.as_posix(), "\\"
    )  # pylint: disable=[no-name-in-module]
    ms = info["FileVersionMS"]
    ls = info["FileVersionLS"]
    return f"{HIWORD(ms)}.{LOWORD(ms)}.{HIWORD(ls)}.{LOWORD(ls)}"  # pylint: disable=[no-name-in-module]


def _ticks(dt: datetime = datetime.utcnow(), short: bool = False) -> int:
    """Python implementation of C# DateTime.UtcNow.Ticks."""
    _ticks = (dt - datetime(1, 1, 1)).total_seconds()

    if short:
        return int(_ticks)
    return int(_ticks * (10**7))


def _console_logger() -> logging.Logger:
    """Create/return a console logger, for outputting messages to the the console."""
    logger = logging.getLogger(f"{_ui_name()}_Console")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    # stream_handler.setFormatter(formatter)
    stream_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(stream_format)
    stream_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)

    return logger


def _file_logger() -> logging.Logger:
    """Create/return a file logger, for outputting BHoM analytics to a file."""
    logger = logging.getLogger(f"{_ui_name()}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    file_handler = RotatingFileHandler(
        _log_folder() / f"Usage_{_ui_name()}_{_ticks(short=True)}.log",
        encoding="utf-8",
        mode="a",
        delay=True,  # wait until all logs collected before writing
        maxBytes=25 * 1024 * 1024,  # 25mb max before file overwritten
        backupCount=0,
    )
    file_format = logging.Formatter("%(message)s")
    file_handler.setFormatter(file_format)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)

    return logger
