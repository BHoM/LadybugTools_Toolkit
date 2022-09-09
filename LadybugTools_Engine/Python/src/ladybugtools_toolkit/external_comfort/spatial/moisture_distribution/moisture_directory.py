from pathlib import Path


def moisture_directory(simulation_directory: Path) -> Path:
    """Get the moisture directory for a spatial simulation.

    Args:
        simulation_directory (Path):
            The associated simulation directory

    Returns:
        Path:
            The path to the moisture directory
    """

    if not (simulation_directory / "moisture").exists():
        raise FileNotFoundError(
            f'No "moisture" directory exists in {simulation_directory}. For this method to work, '
            + 'you need a moisture directory containing a JSON file named "moisture_sources.json". The file contains a list of '
            + 'MoistureSource objects in the format [{"id": "name_of_moisture_source", '
            + '"magnitude": 0.3, "point_indices": [0, 1, 2], "decay_function": "linear", '
            + '"schedule": []}]'
        )

    return simulation_directory / "moisture"
