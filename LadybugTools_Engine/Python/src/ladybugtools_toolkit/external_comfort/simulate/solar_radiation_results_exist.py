from pathlib import Path

from honeybee.model import Model
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.model.equality import equality as model_eq
from ladybugtools_toolkit.external_comfort.simulate.working_directory import (
    working_directory as wd,
)
from ladybugtools_toolkit.ladybug_extension.epw.equality import equality as epw_eq
from ladybugtools_toolkit.ladybug_extension.epw.filename import filename


from python_toolkit.bhom.analytics import analytics


@analytics
def solar_radiation_results_exist(model: Model, epw: EPW = None) -> bool:
    """Check whether results already exist for this configuration of model and EPW.

    Args:
        model (Model): The model to check for.
        epw (EPW): The EPW to check for.

    Returns:
        bool: True if the model and EPW have already been simulated, False otherwise.
    """
    working_directory = wd(model, False)

    # Try to load existing HBJSON file and check that it matches
    try:
        existing_model = Model.from_hbjson(
            (
                working_directory
                / "annual_irradiance"
                / f"{working_directory.stem}.hbjson"
            ).as_posix()
        )
        if not model_eq(model, existing_model, include_identifier=True):
            return False
    except (FileNotFoundError, AssertionError):
        return False

    # Try to load existing EPW file and check that it matches
    try:
        existing_epw = EPW((working_directory / filename(epw, True)).as_posix())
        if not epw_eq(epw, existing_epw, include_header=True):
            return False
    except (FileNotFoundError, AssertionError):
        return False

    # Check that the output files necessary to reload exist
    if not all(
        [
            (
                working_directory / "annual_irradiance/results/direct/UNSHADED.ill"
            ).exists(),
            (
                working_directory / "annual_irradiance/results/total/UNSHADED.ill"
            ).exists(),
        ]
    ):
        return False

    return True
