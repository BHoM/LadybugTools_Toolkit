import getpass
from pathlib import Path

from honeybee.config import folders as hb_folders
from honeybee.model import Model
from ladybug.epw import EPW

from ...ladybug_extension.epw import equality as epw_eq
from ..model import equality as model_eq

hb_folders.default_simulation_folder = f"C:/Users/{getpass.getuser()}/simulation"


def surface_temperature_results_exist(model: Model, epw: EPW) -> bool:
    """Check whether results already exist for this configuration of model and EPW.

    Args:
        model (Model): The model to check for.
        epw (EPW): The EPW to check for.

    Returns:
        bool: True if the model and EPW have already been simulated, False otherwise.
    """
    working_directory: Path = (
        Path(hb_folders.default_simulation_folder) / model.identifier
    )

    # Try to load existing HBJSON file and check that it matches
    try:
        existing_model = Model.from_hbjson(
            (
                working_directory
                / "annual_irradiance"
                / f"{working_directory.stem}.hbjson"
            ).as_posix()
        )
        models_match = model_eq(model, existing_model, include_identifier=True)
    except (FileNotFoundError, AssertionError):
        return False

    # Try to load existing EPW file and check that it matches
    try:
        existing_epw = EPW((working_directory / Path(epw.file_path).name).as_posix())
        epws_match = epw_eq(epw, existing_epw, include_header=True)
    except (FileNotFoundError, AssertionError):
        return False

    # Check that the output files necessary to reload exist
    results_exist = (working_directory / "run" / "eplusout.sql").exists()

    return all([epws_match, models_match, results_exist])
