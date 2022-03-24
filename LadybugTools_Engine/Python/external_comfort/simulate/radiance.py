import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import getpass
from pathlib import Path
from typing import Dict

import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.model import Model
from honeybee_extension.results.load_ill import load_ill
from honeybee_extension.results.make_annual import make_annual
from honeybee_radiance.config import folders as hbr_folders
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug.wea import Wea
from ladybug_extension.datacollection.from_series import from_series
from lbt_recipes.recipe import Recipe, RecipeSettings

USERNAME = getpass.getuser()

ladybug_tools_folder = Path(f"C:/Users/{USERNAME}/ladybug_tools")

hb_folders.default_simulation_folder = f"C:/Users/{USERNAME}/simulation"
hb_folders._python_exe_path = (ladybug_tools_folder / "python/python.exe").as_posix()
hb_folders._python_package_path = (
    ladybug_tools_folder / "python/Lib/site-packages"
).as_posix()
hb_folders._python_scripts_path = (ladybug_tools_folder / "python/Scripts").as_posix()

QUEENBEE_EXE = (ladybug_tools_folder / "python/Scripts/queenbee.exe").as_posix()

hbr_folders.radiance_path = (ladybug_tools_folder / "radiance").as_posix()

if not (Path(hbr_folders.radiance_path) / "bin/rtrace.exe").exists():
    raise FileNotFoundError(
        f"Radiance binaries not found in {hbr_folders.radiance_path}. Ensure that the Radiance installation is located in this directory."
    )


def radiance(model: Model, epw: EPW) -> Dict[str, HourlyContinuousCollection]:
    """Run Radiance on a model and return the results.

    Args:
        model (Model): A honeybee Model to be run through EnergyPlus.
        epw (EPW): An EPW object to be used for the simulation.

    Returns:
        A dictionary containing radiation values.
    """

    working_directory = (
        Path(hb_folders.default_simulation_folder) / f"{model.identifier}"
    )
    working_directory.mkdir(exist_ok=True, parents=True)

    # # TODO - Uncomment below post testing to stop reloading results from already run sim!
    # total_directory = working_directory / "annual_irradiance/results/total"
    # direct_directory = working_directory / "annual_irradiance/results/direct"
    # # TODO - END OF TODO!

    wea = Wea.from_epw_file(epw.file_path)

    recipe = Recipe("annual-irradiance")
    recipe.input_value_by_name("model", model)
    recipe.input_value_by_name("wea", wea)
    recipe_settings = RecipeSettings()
    results = recipe.run(settings=recipe_settings, radiance_check=True)

    total_directory = Path(results) / "annual_irradiance/results/total"
    direct_directory = Path(results) / "annual_irradiance/results/direct"

    unshaded_total = (
        make_annual(load_ill(total_directory / "UNSHADED.ill"))
        .fillna(0)
        .sum(axis=1)
        .rename("GlobalHorizontalRadiation (Wh/m2)")
    )
    unshaded_direct = (
        make_annual(load_ill(direct_directory / "UNSHADED.ill"))
        .fillna(0)
        .sum(axis=1)
        .rename("DirectNormalRadiation (Wh/m2)")
    )
    unshaded_diffuse = (unshaded_total - unshaded_direct).rename(
        "DiffuseHorizontalRadiation (Wh/m2)"
    )

    d = {
        "unshaded_direct_radiation": from_series(unshaded_direct),
        "unshaded_diffuse_radiation": from_series(unshaded_diffuse),
        "shaded_direct_radiation": from_series(
            pd.Series(
                [0] * 8760,
                index=unshaded_total.index,
                name="DirectNormalRadiation (Wh/m2)",
            )
        ),
        "shaded_diffuse_radiation": from_series(
            pd.Series(
                [0] * 8760,
                index=unshaded_total.index,
                name="DiffuseHorizontalRadiation (Wh/m2)",
            )
        ),
    }
    return d
