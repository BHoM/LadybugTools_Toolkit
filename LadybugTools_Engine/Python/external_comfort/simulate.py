import getpass
import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.model import Model
from honeybee_energy.config import folders as hbe_folders
from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
from honeybee_energy.simulation.parameter import (RunPeriod, ShadowCalculation,
                                                  SimulationControl,
                                                  SimulationOutput,
                                                  SimulationParameter)
from honeybee_extension.results.load_ill import load_ill
from honeybee_extension.results.load_sql import load_sql
from honeybee_extension.results.make_annual import make_annual
from honeybee_radiance.config import folders as hbr_folders
from ladybug.epw import EPW, AnalysisPeriod, HourlyContinuousCollection
from ladybug.wea import Wea
from ladybug_extension.datacollection.from_series import from_series
from lbt_recipes.recipe import Recipe, RecipeSettings

from external_comfort.ground_temperature import energyplus_strings

USERNAME = getpass.getuser()

ladybug_tools_folder = Path(f"C:/Users/{USERNAME}/ladybug_tools")

hb_folders.default_simulation_folder = f"C:/Users/{USERNAME}/simulation"
hb_folders._python_exe_path = (ladybug_tools_folder / "python/python.exe").as_posix()
hb_folders._python_package_path = (
    ladybug_tools_folder / "python/Lib/site-packages"
).as_posix()
hb_folders._python_scripts_path = (ladybug_tools_folder / "python/Scripts").as_posix()

hbe_folders.openstudio_path = (ladybug_tools_folder / "openstudio/bin").as_posix()
hbe_folders.energyplus_path = (
    ladybug_tools_folder / "openstudio/EnergyPlus"
).as_posix()
hbe_folders.honeybee_openstudio_gem_path = (
    ladybug_tools_folder / "resources/measures/honeybee_openstudio_gem/lib"
).as_posix()

QUEENBEE_EXE = (ladybug_tools_folder / "python/Scripts/queenbee.exe").as_posix()

hbr_folders.radiance_path = (ladybug_tools_folder / "radiance").as_posix()

if not (Path(hbr_folders.radiance_path) / "bin/rtrace.exe").exists():
    raise FileNotFoundError(
        f"Radiance binaries not found in {hbr_folders.radiance_path}. Ensure that the Radiance installation is located in this directory."
    )



if not (Path(hbe_folders.openstudio_path) / "openstudio.exe").exists():
    raise FileNotFoundError(
        f"openstudio.exe not found in {hbe_folders.openstudio_path}. Ensure that the Openstudio installation is located in this directory."
    )

if not Path(hbe_folders.honeybee_openstudio_gem_path).exists():
    raise FileNotFoundError(
        f"honeybee_openstudio_gem measures not found in {hbe_folders.honeybee_openstudio_gem_path}. Ensure that a Ladyubg-tools installation has been completed installation is located in this directory."
    )


def energyplus(model: Model, epw: EPW) -> Dict[str, HourlyContinuousCollection]:
    """Run EnergyPlus on a model and return the results.

    Args:
        model (Model): A honeybee Model to be run through EnergyPlus.
        epw (EPW): An EPW object to be used for the simulation.

    Returns:
        A dictionary containing ground and shade (below and above) surface temperature values.
    """

    working_directory = (
        Path(hb_folders.default_simulation_folder) / f"{model.identifier}"
    )
    working_directory.mkdir(exist_ok=True, parents=True)

    # Write model JSON
    model_dict = model.to_dict(triangulate_sub_faces=True)
    model_json = working_directory / f"{model.identifier}.hbjson"
    with open(model_json, "w") as fp:
        json.dump(model_dict, fp)

    # Write simulation parameter JSON
    sim_output = SimulationOutput(
        outputs=["Surface Outside Face Temperature"],
        include_sqlite=True,
        summary_reports=None,
        include_html=False,
    )

    sim_control = SimulationControl(
        do_zone_sizing=False,
        do_system_sizing=False,
        do_plant_sizing=False,
        run_for_sizing_periods=False,
        run_for_run_periods=True,
    )
    sim_period = RunPeriod.from_analysis_period(
        AnalysisPeriod(), start_day_of_week="Monday"
    )
    shadow_calc = ShadowCalculation(
        solar_distribution="FullExteriorWithReflections",
        calculation_method="PolygonClipping",
        calculation_update_method="Timestep",
    )
    sim_par = SimulationParameter(
        output=sim_output,
        simulation_control=sim_control,
        shadow_calculation=shadow_calc,
        terrain_type="Country",
        run_period=sim_period,
        timestep=10,
    )
    sim_par_dict = sim_par.to_dict()
    sim_par_json = working_directory / "simulation_parameter.json"
    with open(sim_par_json, "w") as fp:
        json.dump(sim_par_dict, fp)

    # Create OpenStudio workflow
    osw = to_openstudio_osw(
        working_directory.as_posix(),
        model_json.as_posix(),
        sim_par_json.as_posix(),
        additional_measures=None,
        epw_file=epw.file_path,
    )

    # Convert workflow to IDF file
    _, idf = run_osw(osw, silent=False)

    # Add ground temperature strings to IDF
    with open(idf, "r") as fp:
        temp = fp.readlines()
    with open(idf, "w") as fp:
        fp.writelines(temp + [energyplus_strings(epw)])

    # Simulate IDF
    sql, _, _, _, _ = run_idf(idf, epw.file_path, silent=False)

    # Remove files no longer needed (to save on space)
    output_directory = Path(sql).parent
    for file in output_directory.glob("*"):
        if file.suffix not in [".sql", ".err"]:
            os.remove(file)

    # Return results
    df = load_sql(sql)
    d = {
        "shaded_below_temperature": from_series(
            df.filter(regex="GROUND_ZONE_UP_SHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
        ),
        "unshaded_below_temperature": from_series(
            df.filter(regex="GROUND_ZONE_UP_UNSHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
        ),
        "shaded_above_temperature": from_series(
            df.filter(regex="SHADE_ZONE_DOWN").droplevel([0, 1, 2], axis=1).squeeze()
        ),
        "unshaded_above_temperature": epw.sky_temperature,
    }
    return d

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

