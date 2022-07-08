import getpass
import json
import os
import shutil
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
from honeybee_radiance.config import folders as hbr_folders
from ladybug.epw import EPW, AnalysisPeriod, HourlyContinuousCollection
from ladybug.wea import Wea
from lbt_recipes.recipe import Recipe, RecipeSettings

from ..honeybee_extension.results import load_ill, load_sql, make_annual
from ..ladybug_extension.datacollection import from_series
from ..ladybug_extension.epw import _epw_equality
from .ground_temperature import energyplus_strings
from .model import _model_equality

"""
Where this code is run and IT policies modify the "HOME" environment variable, 
this part is essential to make sure that HOME is accessible via the Honeybee/
Queenbee configuration.
"""
USERNAME = getpass.getuser()
os.environ["HOME"] = f"C:\\Users\\{USERNAME}"

QUEENBEE_EXE = "C:/ProgramData/BHoM/Extensions/PythonEnvironments/LadybugTools_Toolkit/Scripts/queenbee.exe"
PYTHON_EXE = (
    "C:/ProgramData/BHoM/Extensions/PythonEnvironments/LadybugTools_Toolkit/python.exe"
)
PYTHON_PACKAGES = "C:/ProgramData/BHoM/Extensions/PythonEnvironments/LadybugTools_Toolkit/Lib/site-packages"
PYTHON_SCRIPTS = (
    "C:/ProgramData/BHoM/Extensions/PythonEnvironments/LadybugTools_Toolkit/scripts"
)

ladybug_tools_folder = Path(f"C:/Users/{USERNAME}/ladybug_tools")

hb_folders.default_simulation_folder = f"C:/Users/{USERNAME}/simulation"
SIMULATION_DIRECTORY = Path(hb_folders.default_simulation_folder)

Path(hb_folders.default_simulation_folder).mkdir(parents=True, exist_ok=True)
hb_folders._python_exe_path = PYTHON_EXE
hb_folders._python_package_path = PYTHON_PACKAGES
hb_folders._python_scripts_path = PYTHON_SCRIPTS

hbe_folders.openstudio_path = (ladybug_tools_folder / "openstudio/bin").as_posix()
hbe_folders.energyplus_path = (
    ladybug_tools_folder / "openstudio/EnergyPlus"
).as_posix()
hbe_folders.honeybee_openstudio_gem_path = (
    ladybug_tools_folder / "resources/measures/honeybee_openstudio_gem/lib"
).as_posix()
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
        f"honeybee_openstudio_gem measures not found in {hbe_folders.honeybee_openstudio_gem_path}. Ensure that a Ladybug-tools installation has been completed and this directory exists (with contents!)"
    )


def energyplus(model: Model, epw: EPW) -> Dict[str, HourlyContinuousCollection]:
    """Run EnergyPlus on a model and return the results.

    Args:
        model (Model): A honeybee Model to be run through EnergyPlus.
        epw (EPW): An EPW object to be used for the simulation.

    Returns:
        A dictionary containing ground and shade (below and above) surface temperature values.
    """

    working_directory: Path = SIMULATION_DIRECTORY / model.identifier
    working_directory.mkdir(parents=True, exist_ok=True)
    sql_path = working_directory / "run" / "eplusout.sql"

    if not _energyplus_results_exist(model, epw):
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
        with open(idf, "a") as fp:
            fp.writelines([energyplus_strings(epw)])
        # TODO - Replace this part with a proper "Other Side Boundary Condition" for
        # hourly ground temperatures, once possible in HB-energy
        # (https://github.com/ladybug-tools/honeybee-energy/issues/407)

        # Simulate IDF
        _, _, _, _, _ = run_idf(idf, epw.file_path, silent=False)

        # Remove unneeded files
        _tidy_energyplus_results(working_directory)

        # save EPW to working directory
        epw.save(working_directory / Path(epw.file_path).name)
    else:
        print("- Loading surface temperatures")

    # Return results
    df = load_sql(sql_path)
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

    working_directory: Path = SIMULATION_DIRECTORY / model.identifier
    working_directory.mkdir(parents=True, exist_ok=True)
    total_directory = working_directory / "annual_irradiance/results/total"
    direct_directory = working_directory / "annual_irradiance/results/direct"

    if not _radiance_results_exist(model, epw):
        print("- Simulating annual irradiance")
        wea = Wea.from_epw_file(epw.file_path)

        recipe = Recipe("annual-irradiance")
        recipe.input_value_by_name("model", model)
        recipe.input_value_by_name("wea", wea)
        recipe_settings = RecipeSettings()
        _ = recipe.run(
            settings=recipe_settings, radiance_check=True, queenbee_path=QUEENBEE_EXE
        )

        # save EPW to working directory
        epw.save(working_directory / Path(epw.file_path).name)
    else:
        print("- Loading annual irradiance")

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

    # Remove unneeded files
    _tidy_radiance_results(working_directory)

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


def _radiance_results_exist(model: Model, epw: EPW) -> bool:
    """Check whether results already exist for this configuration of model and EPW.

    Args:
        model (Model): The model to check for.
        epw (EPW): The EPW to check for.

    Returns:
        bool: True if the model and EPW have already been simulated, False otherwise.
    """
    working_directory: Path = SIMULATION_DIRECTORY / model.identifier

    # Try to load existing HBJSON file and check that it matches
    try:
        existing_model = Model.from_hbjson(
            (
                working_directory
                / "annual_irradiance"
                / f"{working_directory.stem}.hbjson"
            ).as_posix()
        )
        models_match = _model_equality(model, existing_model, include_identifier=True)
        # if models_match:
        #     print("models match")
    except (FileNotFoundError, AssertionError) as e:
        return False

    # Try to load existing EPW file and check that it matches
    try:
        existing_epw = EPW((working_directory / Path(epw.file_path).name).as_posix())
        epws_match = _epw_equality(epw, existing_epw, include_header=True)
        # if epws_match:
        #     print("epw match")
    except (FileNotFoundError, AssertionError) as e:
        return False

    # Check that the output files necessary to reload exist
    results_exist = all(
        [
            (
                working_directory / "annual_irradiance/results/direct/UNSHADED.ill"
            ).exists(),
            (
                working_directory / "annual_irradiance/results/total/UNSHADED.ill"
            ).exists(),
        ]
    )
    radiance_results_exist = all([epws_match, models_match, results_exist])
    # if radiance_results_exist:
    #     print("radiance_results_exist")

    return radiance_results_exist


def _energyplus_results_exist(model: Model, epw: EPW) -> bool:
    """Check whether results already exist for this configuration of model and EPW.

    Args:
        model (Model): The model to check for.
        epw (EPW): The EPW to check for.

    Returns:
        bool: True if the model and EPW have already been simulated, False otherwise.
    """
    working_directory: Path = SIMULATION_DIRECTORY / model.identifier

    # Try to load existing HBJSON file and check that it matches
    try:
        existing_model = Model.from_hbjson(
            (
                working_directory
                / "annual_irradiance"
                / f"{working_directory.stem}.hbjson"
            ).as_posix()
        )
        models_match = _model_equality(model, existing_model, include_identifier=True)
    except (FileNotFoundError, AssertionError) as e:
        return False

    # Try to load existing EPW file and check that it matches
    try:
        existing_epw = EPW((working_directory / Path(epw.file_path).name).as_posix())
        epws_match = _epw_equality(epw, existing_epw, include_header=True)
    except (FileNotFoundError, AssertionError) as e:
        return False

    # Check that the output files necessary to reload exist
    results_exist = (working_directory / "run" / "eplusout.sql").exists()

    energyplus_results_exist = all([epws_match, models_match, results_exist])

    if energyplus_results_exist:
        return True
    else:
        return False


def _tidy_radiance_results(working_directory: Path) -> None:
    """Tidy up an radiance results folder created using hb-radiance.

    Args:
        working_directory (Path): The working directory to tidy.
    """
    # Find all files in directory
    all_files = working_directory.glob("**/*")
    # Remove files and folders
    folders_to_delete = [
        working_directory / "annual_irradiance/resources",
        working_directory / "annual_irradiance/__logs__",
        working_directory / "annual_irradiance/__params",
        working_directory / "annual_irradiance/initial_results",
        working_directory / "annual_irradiance/metrics",
        working_directory / "annual_irradiance/model",
    ]
    files_to_delete = [
        "__inputs__.json",
        "annual_irradiance_raytracing.done",
        "annual_irradiance_inputs.json",
    ]
    files_of_type_to_delete = [".wea", ".done"]

    for file in all_files:
        if (file.suffix in files_of_type_to_delete) or (file.name in files_to_delete):
            os.remove(file)
    for folder in folders_to_delete:
        shutil.rmtree(folder, ignore_errors=True)

    return None


def _tidy_energyplus_results(working_directory: Path) -> None:
    """Tidy up an energyplus results folder created using hb-energy.

    Args:
        working_directory (Path): The working directory to tidy.
    """

    # Find all files in directory
    all_files = working_directory.glob("**/*")
    # Remove files and folders
    files_to_delete = [
        "run_workflow.bat",
        "workflow.osw",
        "out.osw",
        "simulation_parameter.json",
        "in.osm",
        "measure_attributes.json",
        "run.log",
        "sqlite.err",
        "started.job",
        "data_point.zip",
        "eplusout.audit",
        "eplusout.bnd",
        "eplusout.eio",
        "eplusout.end",
        "eplusout.eso",
        "eplusout.mdd",
        "eplusout.mtd",
        "eplusout.rdd",
        "eplusout.shd",
        "eplustbl.htm",
        "finished.job",
        "in.bat",
        "in.idf",
    ]
    for file in all_files:
        if file.name in files_to_delete:
            os.remove(file)
    (working_directory / "generated_files").rmdir()
    (working_directory / "reports").rmdir()

    return None
