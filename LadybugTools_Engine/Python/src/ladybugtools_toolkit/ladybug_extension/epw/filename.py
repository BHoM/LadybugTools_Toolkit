from pathlib import Path

from ladybug.epw import EPW


def filename(epw: EPW, include_extension: bool = False) -> str:
    """Get the filename of the given EPW.

    Args:
        epw (EPW):
            An EPW object.
        include_extension (bool, optional):
            Set to True to include the file extension. Defaults to False.

    Returns:
        string:
            The name of the EPW file.
    """

    if include_extension:
        return Path(epw.file_path).name

    return Path(epw.file_path).stem
