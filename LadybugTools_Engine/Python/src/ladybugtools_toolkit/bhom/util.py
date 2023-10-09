"""General utility functions."""
import configparser
from datetime import datetime
from pathlib import Path

import psutil
from win32api import HIWORD, LOWORD, GetFileVersionInfo


def process_chain() -> str:
    """Return the process chain.

    This method determines the chain of processes that led to the current instance of Python being run.
    """

    chain = []
    proc = psutil.Process()
    while proc is not None:
        chain.append(Path(proc.name()).stem)
        proc = proc.parent()
    seen = set()
    seen_add = seen.add
    return " > ".join(reversed([x for x in chain if not (x in seen or seen_add(x))]))


def csharp_ticks(date_time: datetime = datetime.utcnow(), short: bool = False) -> int:
    """Python implementation of C# DateTime.UtcNow.Ticks.

    Args:
        date_time (datetime, optional): The datetime to convert to ticks. Defaults to datetime.utcnow().
        short (bool, optional): Whether to return the short ticks. Defaults to False.

    Returns:
        int: The ticks.
    """

    _ticks = (date_time - datetime(1, 1, 1)).total_seconds()

    if short:
        return int(_ticks)

    return int(_ticks * (10**7))


def bhom_version() -> str:
    """Return the BHoM version."""

    current_dir = Path(__file__).parent

    # load defaults from config.ini
    config = configparser.ConfigParser()
    config.read(current_dir / "config.ini")

    assemblies_directory = config.get("DEFAULT", "BHoMAssembliesDirectory")

    version_info = GetFileVersionInfo(
        (Path(assemblies_directory) / "BHoM.dll").as_posix(), "\\"
    )
    most_significant = version_info["FileVersionMS"]

    return f"{HIWORD(most_significant)}.{LOWORD(most_significant)}"


def toolkit_name() -> str:
    """Return the toolkit name."""

    current_dir = Path(__file__).parent

    config = configparser.ConfigParser()
    config.read(current_dir / "config.ini")

    return config.get("DEFAULT", "ToolkitName")


def log_folder() -> Path:
    """Return the log folder."""

    current_dir = Path(__file__).parent

    config = configparser.ConfigParser()
    config.read(current_dir / "config.ini")

    _log_folder = Path(config.get("DEFAULT", "BHoMLogFolder"))
    _log_folder.mkdir(parents=True, exist_ok=True)
    return _log_folder
