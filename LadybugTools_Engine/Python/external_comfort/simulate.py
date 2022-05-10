import getpass
import json
import os
import time
from pathlib import Path
from typing import Dict

import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.model import Model
from honeybee_energy.config import folders as hbe_folders
from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
from honeybee_energy.simulation.parameter import (
    RunPeriod,
    ShadowCalculation,
    SimulationControl,
    SimulationOutput,
    SimulationParameter,
)
from honeybee_extension.results import load_ill, load_sql, make_annual
from honeybee_radiance.config import folders as hbr_folders
from ladybug.epw import EPW, AnalysisPeriod, HourlyContinuousCollection
from ladybug.wea import Wea
from ladybug_extension.datacollection import from_series
from lbt_recipes.recipe import Recipe, RecipeSettings

from external_comfort.ground_temperature import energyplus_strings

USERNAME = getpass.getuser()

"""
Where this code is run and IT policies modify the "HOME" environment variable, 
this part is essential to make sure that HOME is accessible via the Honeybee/
Queenbee configuration.
"""
os.environ["HOME"] = f"C:\\Users\\{USERNAME}"

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
    time.sleep(1)
    working_directory = _working_directory(model, epw)

    if _do_energyplus_results_exist(model, epw):
        print("- Loading previously simulated surface temperature results")
        sql = (working_directory / "run" / "eplusout.sql").as_posix()
    else:
        print("- Simulating surface temperatures")
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
    return {
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


def radiance(model: Model, epw: EPW) -> Dict[str, HourlyContinuousCollection]:
    """Run Radiance on a model and return the results.

    Args:
        model (Model): A honeybee Model to be run through EnergyPlus.
        epw (EPW): An EPW object to be used for the simulation.

    Returns:
        A dictionary containing radiation values.
    """

    working_directory = _working_directory(model, epw)

    if _do_radiance_results_exist(model, epw):
        print("- Loading previously simulated annual irradiance")
        total_directory = working_directory / "annual_irradiance/results/total"
        direct_directory = working_directory / "annual_irradiance/results/direct"
    else:
        print("- Simulating annual irradiance")
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

    # Remove files no longer needed (to save on space)
    for file in list((working_directory / "annual_irradiance").glob("**/*")):
        if file.is_file():
            if file.suffix not in [".txt", ".ill", ".hbjson"]:
                os.remove(file)

    return {
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


def _working_directory(model: Model, epw: EPW) -> Path:
    """Return the working directory for the radiance and energyplus simulations.

    Args:
        model (Model): A HB model to be run through Radiance and Energyplus.
        epw (EPW): An EPW object to be used for the simulation.

    Returns:
        Path: The path where the simulation results will be stored.
    """
    working_directory = (
        Path(hb_folders.default_simulation_folder) / f"{model.identifier}"
    )
    working_directory.mkdir(exist_ok=True, parents=True)

    # Write EPW to working directory, and stopping if the lockfile exists
    lockfile = working_directory / "epw.lock"
    if not lockfile.exists():
        with open(lockfile, "w+") as fp:
            pass
        epw.save(working_directory / Path(epw.file_path).name)

    return working_directory


def _do_radiance_results_exist(model: Model, epw: EPW) -> bool:
    """Check whether results already exist for this configuration of model and EPW.

    Args:
        model (Model): The model to check for.
        epw (EPW): The EPW to check for.

    Returns:
        bool: True if the model and EPW have already been simulated, False otherwise.
    """
    working_directory = _working_directory(model, epw)

    # Try to load existing HBJSON file
    try:
        existing_model_path = working_directory / f"{working_directory.stem}.hbjson"
        existing_model = Model.from_hbjson(existing_model_path)
    except (FileNotFoundError, AssertionError) as e:
        return False

    # Check that identifiers match
    model_identifiers_match = existing_model.identifier == model.identifier

    # Check that the directories exist
    directories_exist = (
        working_directory / "annual_irradiance" / "results" / "direct" / "UNSHADED.ill"
    ).exists() and (
        working_directory / "annual_irradiance" / "results" / "total" / "UNSHADED.ill"
    ).exists()

    # Check that the EPW file is the same
    existing_epw_filename = list(working_directory.glob("*.epw"))[0].name
    epws_match = existing_epw_filename == Path(epw.file_path).name

    # Check the HBJSON materials match
    existing_ground_material = (
        existing_model.rooms[0].faces[-1].properties.energy.construction.materials[0]
    )
    existing_shade_material = (
        existing_model.rooms[2].faces[0].properties.energy.construction.materials[0]
    )
    proposed_ground_material = (
        model.rooms[0].faces[-1].properties.energy.construction.materials[0]
    )
    proposed_shade_material = (
        model.rooms[2].faces[0].properties.energy.construction.materials[0]
    )
    materials_match = (existing_ground_material == proposed_ground_material) and (
        existing_shade_material == proposed_shade_material
    )

    match_found = all(
        [model_identifiers_match, directories_exist, epws_match, materials_match]
    )

    if match_found:
        return True
    else:
        return False


def _do_energyplus_results_exist(model: Model, epw: EPW) -> bool:
    """Check whether results already exist for this configuration of model and EPW.

    Args:
        model (Model): The model to check for.
        epw (EPW): The EPW to check for.

    Returns:
        bool: True if the model and EPW have already been simulated, False otherwise.
    """
    working_directory = _working_directory(model, epw)

    # Try to load existing HBJSON file
    try:
        existing_model_path = working_directory / f"{working_directory.stem}.hbjson"
        existing_model = Model.from_hbjson(existing_model_path)
    except (FileNotFoundError, AssertionError) as e:
        return False

    # Check that identifiers match
    model_identifiers_match = existing_model.identifier == model.identifier

    # Check that the files exist
    file_exist = (working_directory / "run" / "eplusout.sql").exists()

    # Check that the EPW file is the same
    existing_epw_filename = list(working_directory.glob("*.epw"))[0].name
    epws_match = existing_epw_filename == Path(epw.file_path).name

    # Check the HBJSON materials match
    existing_ground_material = (
        existing_model.rooms[0].faces[-1].properties.energy.construction.materials[0]
    )
    existing_shade_material = (
        existing_model.rooms[2].faces[0].properties.energy.construction.materials[0]
    )
    proposed_ground_material = (
        model.rooms[0].faces[-1].properties.energy.construction.materials[0]
    )
    proposed_shade_material = (
        model.rooms[2].faces[0].properties.energy.construction.materials[0]
    )
    materials_match = (existing_ground_material == proposed_ground_material) and (
        existing_shade_material == proposed_shade_material
    )

    match_found = all(
        [model_identifiers_match, file_exist, epws_match, materials_match]
    )

    if match_found:
        return True
    else:
        return False
