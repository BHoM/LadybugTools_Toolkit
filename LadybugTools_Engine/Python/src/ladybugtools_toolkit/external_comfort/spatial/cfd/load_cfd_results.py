from pathlib import Path
from typing import List

import pandas as pd


def load_moisture_sources(simulation_directory: Path) -> List[MoistureSource]:
    """Get/create the moisture directory for a spatial simulation.

    Args:
        simulation_directory (Path):
            The associated simulation directory.

    Returns:
        List[MoistureSource]:
            A list of moisture sources in the simulation directory.
    """

    return MoistureSource.from_json(
        moisture_directory(simulation_directory) / "moisture_sources.json"
    )
