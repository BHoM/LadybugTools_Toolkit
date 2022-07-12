import getpass
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.model import Model
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug.wea import Wea
from lbt_recipes.recipe import Recipe, RecipeSettings

from ...honeybee_extension.results import load_ill, make_annual
from ...ladybug_extension.datacollection import from_series
from .solar_radiation_results_exist import solar_radiation_results_exist


def solar_radiation(model: Model, epw: EPW) -> Tuple[Path]:
    """Run Radiance on a model and return the results.

    Args:
        model (Model): A honeybee Model to be run through EnergyPlus.
        epw (EPW): An EPW object to be used for the simulation.

    Returns:
        total_irradiance, direct_irradiance (Tuple[Path]): A tuple containing paths to the total and direct insolation ILL files.
    """

    os.environ["HOME"] = f"C:\\Users\\{getpass.getuser()}"
    hb_folders.default_simulation_folder = f"C:/Users/{getpass.getuser()}/simulation"

    working_directory: Path = (
        Path(hb_folders.default_simulation_folder) / model.identifier
    )
    working_directory.mkdir(parents=True, exist_ok=True)

    print("- Simulating annual irradiance")
    wea = Wea.from_epw_file(epw.file_path)

    recipe = Recipe("annual-irradiance")
    recipe.input_value_by_name("model", model)
    recipe.input_value_by_name("wea", wea)
    recipe_settings = RecipeSettings()
    _ = recipe.run(
        settings=recipe_settings,
        radiance_check=True,
        queenbee_path=r"C:\Program Files\ladybug_tools\python\Scripts\queenbee.exe",
    )

    # save EPW to working directory
    epw.save(working_directory / Path(epw.file_path).name)

    total_irradiance = (
        working_directory / "annual_irradiance/results/total/UNSHADED.ill"
    )
    direct_irradiance = (
        working_directory / "annual_irradiance/results/direct/UNSHADED.ill"
    )

    return total_irradiance, direct_irradiance
