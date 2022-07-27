from pathlib import Path
from typing import List, Union

from ladybugtools_toolkit.external_comfort.spatial.moisture_distribution.moisture_directory import (
    moisture_directory,
)
from ladybugtools_toolkit.external_comfort.spatial.moisture_distribution.moisture_source import (
    MoistureSource,
)


def load_moisture_sources(
    simulation_directory: Union[Path, str]
) -> List[MoistureSource]:
    """Get/create the moisture directory for a spatial simulation.

    Args:
        simulation_directory (Union[Path, str]):
            The associated simulation directory.

    Returns:
        List[MoistureSource]:
            A list of moisture sources in the simulation directory.
    """

    return MoistureSource.from_json(
        moisture_directory(simulation_directory) / "moisture_sources.json"
    )
