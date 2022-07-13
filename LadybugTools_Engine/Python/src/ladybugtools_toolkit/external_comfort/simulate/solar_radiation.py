from pathlib import Path
from typing import Dict

from honeybee.model import Model
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug.wea import Wea
from lbt_recipes.recipe import Recipe, RecipeSettings

from .solar_radiation_results_exist import solar_radiation_results_exist
from .solar_radiation_results_load import solar_radiation_results_load
from .working_directory import working_directory as wd


def solar_radiation(model: Model, epw: EPW) -> Dict[str, HourlyContinuousCollection]:
    """Run Radiance on a model and return the results.

    Args:
        model (Model): A honeybee Model to be run through Radiance.
        epw (EPW): An EPW object to be used for the simulation.

    Returns:
         Dict[str, HourlyContinuousCollection]: A dictionary containing radiation-related collections.
    """

    working_directory = wd(model)

    total_irradiance = (
        working_directory / "annual_irradiance/results/total/UNSHADED.ill"
    )
    direct_irradiance = (
        working_directory / "annual_irradiance/results/direct/UNSHADED.ill"
    )

    if solar_radiation_results_exist(model, epw):
        print(f"[{model.identifier}] - Loading annual irradiance")
        return solar_radiation_results_load(total_irradiance, direct_irradiance)

    print(f"[{model.identifier}] - Simulating annual irradiance")
    wea = Wea.from_epw_file(epw.file_path)

    recipe = Recipe("annual-irradiance")
    recipe.input_value_by_name("model", model)
    recipe.input_value_by_name("wea", wea)
    recipe.input_value_by_name("north", 0)
    recipe.input_value_by_name("timestep", 1)
    recipe.input_value_by_name("output-type", "solar")
    recipe.input_value_by_name("grid-filter", "UNSHADED")
    recipe_settings = RecipeSettings()
    _ = recipe.run(
        settings=recipe_settings,
        radiance_check=True,
        queenbee_path=r"C:\Program Files\ladybug_tools\python\Scripts\queenbee.exe",
    )

    return solar_radiation_results_load(total_irradiance, direct_irradiance)
