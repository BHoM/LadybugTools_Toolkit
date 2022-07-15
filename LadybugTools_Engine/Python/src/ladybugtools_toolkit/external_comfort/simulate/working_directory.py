import getpass
from pathlib import Path

from honeybee.config import folders as hb_folders
from honeybee.model import Model


def working_directory(model: Model, create: bool = False) -> Path:
    """Get the working directory (where simulation results will be stored) for the given model, and
        create it if it doesnt already exist.

    Args:
        model (Model): A honeybee Model.
        create (bool, optional): Set to True to create the directory. Default is False.

    Returns:
        Path: The simulation directory associated with the given model.
    """

    hb_folders.default_simulation_folder = f"C:/Users/{getpass.getuser()}/simulation"

    working_directory: Path = (
        Path(hb_folders.default_simulation_folder) / model.identifier
    )
    if create:
        working_directory.mkdir(parents=True, exist_ok=True)

    return working_directory
