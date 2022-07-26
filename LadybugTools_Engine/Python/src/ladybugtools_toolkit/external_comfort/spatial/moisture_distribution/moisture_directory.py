from pathlib import Path
from typing import Union


def moisture_directory(simulation_directory: Union[Path, str]) -> Path:
    """Get/create the moisture directory for a spatial simulation.

    Args:
        simulation_directory (Union[Path, str]): The associated simulation directory

    Returns:
        Path: The path to the moisture directory
    """

    if not (simulation_directory / "moisture").exists():
        raise FileNotFoundError(
            f'No "moisture" directory exists in {simulation_directory}'
        )

    return simulation_directory / "moisture"
