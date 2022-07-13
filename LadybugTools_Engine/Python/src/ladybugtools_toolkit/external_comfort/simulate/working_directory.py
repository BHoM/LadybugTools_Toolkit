import getpass
from pathlib import Path

from honeybee.config import folders as hb_folders
from honeybee.model import Model


def working_directory(model: Model) -> Path:
    """Get the working directory (where simulation results will be stored) for the given model.

    Args:
        model (Model): A honeybee Model.

    Returns:
         Path: The simulation directory associated with the given model.
    """

    hb_folders.default_simulation_folder = f"C:/Users/{getpass.getuser()}/simulation"

    working_directory: Path = (
        Path(hb_folders.default_simulation_folder) / model.identifier
    )
    working_directory.mkdir(parents=True, exist_ok=True)

    return working_directory
