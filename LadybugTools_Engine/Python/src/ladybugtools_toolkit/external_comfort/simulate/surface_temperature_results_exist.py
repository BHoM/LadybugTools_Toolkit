from pathlib import Path

from honeybee.model import Model
from ladybug.epw import EPW

from ...ladybug_extension.epw import equality as epw_eq
from ..model import equality as model_eq
from .working_directory import working_directory as wd


def surface_temperature_results_exist(model: Model, epw: EPW = None) -> bool:
    """Check whether results already exist for this configuration of model and EPW.

    Args:
        model (Model): The model to check for.
        epw (EPW): The EPW to check for.  Currently unused.

    Returns:
        bool: True if the model and EPW have already been simulated, False otherwise.
    """
    working_directory = wd(model, False)

    # Try to load existing HBJSON file and check that it matches
    try:
        existing_model = Model.from_hbjson(
            (working_directory / f"{working_directory.stem}.hbjson").as_posix()
        )
        if not model_eq(model, existing_model, include_identifier=True):
            return False
    except (FileNotFoundError, AssertionError):
        return False

    # Try to load existing EPW file and check that it matches
    try:
        existing_epw = EPW((working_directory / Path(epw.file_path).name).as_posix())
        if not epw_eq(epw, existing_epw, include_header=True):
            return False
    except (FileNotFoundError, AssertionError):
        return False

    # Check that the output files necessary to reload exist
    if not (working_directory / "run" / "eplusout.sql").exists():
        return False

    return True
