import sys
sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import pandas as pd

import json
import os
from pathlib import Path
from typing import Dict, List

from lbt_recipes.recipe import Recipe, RecipeSettings
from honeybee.model import Model
from ladybug.epw import EPW, HourlyContinuousCollection, AnalysisPeriod
from ladybug.wea import Wea
from honeybee_energy.simulation.parameter import (
    RunPeriod,
    ShadowCalculation,
    SimulationControl,
    SimulationOutput,
    SimulationParameter,
)
from ladybug.datatype.temperature import Temperature
from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
import numpy as np
from ladybug_comfort.collection.solarcal import HorizontalSolarCal
from ladybug_comfort.parameter.solarcal import SolarCalParameter

from honeybee_extension.results import load_sql, load_ill, _make_annual
from ladybug_extension.datacollection import from_series, to_series
from external_comfort.ground_temperature import energyplus_ground_temperature_strings
from external_comfort.material import MATERIALS
import getpass
from pathlib import Path

from honeybee.config import folders as hb_folders
from honeybee_energy.config import folders as hbe_folders
from honeybee_radiance.config import folders as hbr_folders

USERNAME = getpass.getuser()

ladybug_tools_folder = Path(f"C:/Users/{USERNAME}/ladybug_tools")

hb_folders.default_simulation_folder = f"C:/Users/{USERNAME}/simulation"
hb_folders._python_exe_path = (ladybug_tools_folder / "python/python.exe").as_posix()
hb_folders._python_package_path = (ladybug_tools_folder / "python/Lib/site-packages").as_posix()
hb_folders._python_scripts_path = (ladybug_tools_folder / "python/Scripts").as_posix()

QUEENBEE_EXE = (ladybug_tools_folder / "python/Scripts/queenbee.exe").as_posix()

hbe_folders.openstudio_path = (ladybug_tools_folder / "openstudio/bin").as_posix()
hbe_folders.energyplus_path = (ladybug_tools_folder / "openstudio/EnergyPlus").as_posix()
hbe_folders.honeybee_openstudio_gem_path = (ladybug_tools_folder / "resources/measures/honeybee_openstudio_gem/lib").as_posix()

hbr_folders.radiance_path = (ladybug_tools_folder / "radiance").as_posix()

assert (Path(hbe_folders.openstudio_path) / "openstudio.exe").exists(), \
    f"openstudio.exe not found in {hbe_folders.openstudio_path}. Ensure that the Openstudio installation is located in this directory."

assert Path(hbe_folders.honeybee_openstudio_gem_path).exists(), \
    f"honeybee_openstudio_gem measures not found in {hbe_folders.honeybee_openstudio_gem_path}. Ensure that a Ladyubg-tools installation has been completed installation is located in this directory."

assert (Path(hbr_folders.radiance_path) / "bin/rtrace.exe").exists(), \
    f"Radiance binaries not found in {hbr_folders.radiance_path}. Ensure that the Radiance installation is located in this directory."

def _run_energyplus(
    model: Model, epw: EPW
) -> Dict[str, HourlyContinuousCollection]:
    """Run EnergyPlus on a model and return the results.

    Args:
        model (Model): A honeybee Model to be run through EnergyPlus.
        epw (EPW): An EPW object to be used for the simulation.

    Returns:
        A dictionary containing ground and shade (below and above) surface temperature values.
    """

    working_directory = Path(hb_folders.default_simulation_folder) / f"{model.identifier}"
    working_directory.mkdir(exist_ok=True, parents=True)

    # TODO - Uncomment below post testing to stop reloading results from already run sim!
    sql = (working_directory / "run/eplusout.sql").as_posix()
    # TODO - END OF TODO!

    # # Write model JSON
    # model_dict = model.to_dict(triangulate_sub_faces=True)
    # model_json = working_directory / f"{model.identifier}.hbjson"
    # with open(model_json, "w") as fp:
    #     json.dump(model_dict, fp)

    # # Write simulation parameter JSON
    # sim_output = SimulationOutput(
    #     outputs=["Surface Outside Face Temperature"],
    #     include_sqlite=True,
    #     summary_reports=None,
    #     include_html=False,
    # )

    # sim_control = SimulationControl(
    #     do_zone_sizing=False,
    #     do_system_sizing=False,
    #     do_plant_sizing=False,
    #     run_for_sizing_periods=False,
    #     run_for_run_periods=True,
    # )
    # sim_period = RunPeriod.from_analysis_period(
    #     AnalysisPeriod(), start_day_of_week="Monday"
    # )
    # shadow_calc = ShadowCalculation(
    #     solar_distribution="FullExteriorWithReflections",
    #     calculation_method="PolygonClipping",
    #     calculation_update_method="Timestep",
    # )
    # sim_par = SimulationParameter(
    #     output=sim_output,
    #     simulation_control=sim_control,
    #     shadow_calculation=shadow_calc,
    #     terrain_type="Country",
    #     run_period=sim_period,
    #     timestep=10,
    # )
    # sim_par_dict = sim_par.to_dict()
    # sim_par_json = working_directory / "simulation_parameter.json"
    # with open(sim_par_json, "w") as fp:
    #     json.dump(sim_par_dict, fp)

    # # Create OpenStudio workflow
    # osw = to_openstudio_osw(
    #     working_directory.as_posix(),
    #     model_json.as_posix(),
    #     sim_par_json.as_posix(),
    #     additional_measures=None,
    #     epw_file=epw.file_path,
    # )

    # # Convert workflow to IDF file
    # _, idf = run_osw(osw, silent=False)

    # # Add ground temperature strings to IDF
    # with open(idf, "r") as fp:
    #     temp = fp.readlines()
    # with open(idf, "w") as fp:
    #     fp.writelines(temp + [energyplus_ground_temperature_strings(epw)])

    # # Simulate IDF
    # sql, _, _, _, _ = run_idf(idf, epw.file_path, silent=False)

    # # Remove files no longer needed (to save on space)
    # output_directory = Path(sql).parent
    # for file in output_directory.glob("*"):
    #     if file.suffix not in [".sql", ".err"]:
    #         os.remove(file)

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


def _run_radiance(model: Model, epw: EPW) -> Dict[str, HourlyContinuousCollection]:
    """Run Radiance on a model and return the results.

    Args:
        model (Model): A honeybee Model to be run through EnergyPlus.
        epw (EPW): An EPW object to be used for the simulation.

    Returns:
        A dictionary containing radiation values.
    """

    working_directory = Path(hb_folders.default_simulation_folder) / f"{model.identifier}"
    working_directory.mkdir(exist_ok=True, parents=True)

    # TODO - Uncomment below post testing to stop reloading results from already run sim!
    total_directory = working_directory / "annual_irradiance/results/total"
    direct_directory = working_directory / "annual_irradiance/results/direct"
    # TODO - END OF TODO!

    # wea = Wea.from_epw_file(epw.file_path)

    # recipe = Recipe("annual-irradiance")
    # recipe.input_value_by_name("model", model)
    # recipe.input_value_by_name("wea", wea)
    # recipe_settings = RecipeSettings()
    # results = recipe.run(settings=recipe_settings, radiance_check=True)

    # total_directory = Path(results) / "annual_irradiance/results/total"
    # direct_directory = Path(results) / "annual_irradiance/results/direct"
    
    unshaded_total = _make_annual(load_ill(total_directory / "UNSHADED.ill")).fillna(0).sum(axis=1).rename("GlobalHorizontalRadiation (Wh/m2)")
    unshaded_direct = _make_annual(load_ill(direct_directory / "UNSHADED.ill")).fillna(0).sum(axis=1).rename("DirectNormalRadiation (Wh/m2)")
    unshaded_diffuse = (unshaded_total - unshaded_direct).rename("DiffuseHorizontalRadiation (Wh/m2)")

    d = {
        "unshaded_direct_radiation": from_series(unshaded_direct),
        "unshaded_diffuse_radiation": from_series(unshaded_diffuse),
        "shaded_direct_radiation": from_series(pd.Series([0] * 8760, index=unshaded_total.index, name="DirectNormalRadiation (Wh/m2)")),
        "shaded_diffuse_radiation": from_series(pd.Series([0] * 8760, index=unshaded_total.index, name="DiffuseHorizontalRadiation (Wh/m2)")),
    }
    return d


def _radiant_temperature_from_collections(collections: List[HourlyContinuousCollection], view_factors: List[float]) -> HourlyContinuousCollection:
    assert len(collections) == len(view_factors), \
        "The number of collections and view factors must be the same."
    assert sum(view_factors) == 1, \
        "The sum of view factors must be 1."
    
    mrt_series = np.power((np.power(pd.concat([to_series(i) for i in collections], axis=1) + 273.15, 4) * view_factors).sum(axis=1), 0.25) - 273.15
    mrt_series.name = "Temperature (C)"
    return from_series(mrt_series)


def _mean_radiant_temperature_from_surfaces(
    surface_temperatures: List[float], view_factors: List[float]
) -> float:
    """Calculate Mean Radiant Temperature from a list of surface temperature and view factors to those surfaces.

    Args:
        surface_temperatures (List[float]): A list of surface temperatures.
        view_factors (List[float]): A list of view-factors (one per surface)

    Returns:
        float: A value describing resultant radiant temperature.
    """
    resultant_temperature = 0
    for i, temp in enumerate(surface_temperatures):
        temperature_kelvin = temp + 273.15
        resultant_temperature = (
            resultant_temperature + np.pow(temperature_kelvin, 4) * view_factors[i]
        )
    mean_radiant_temperature_kelvin = np.pow(resultant_temperature, 0.25)
    mean_radiant_temperature = mean_radiant_temperature_kelvin - 273.15
    return mean_radiant_temperature


def _convert_radiation_to_mean_radiant_temperature(
    epw: EPW,
    surface_temperature: HourlyContinuousCollection,
    direct_radiation: HourlyContinuousCollection,
    diffuse_radiation: HourlyContinuousCollection,
) -> HourlyContinuousCollection:
    """Using the SolarCal method, convert surrounding surface temperature and direct/diffuse radiation into mean radiant temperature.

    Args:
        epw (EPW): A ladybug EPW object.
        surface_temperature (HourlyContinuousCollection): A ladybug surface temperature data collection.
        direct_radiation (HourlyContinuousCollection): A ladybug radiation data collection representing direct solar radiation.
        diffuse_radiation (HourlyContinuousCollection): A ladybug radiation data collection representing diffuse solar radiation.

    Returns:
        HourlyContinuousCollection: A ladybug mean radiant temperature data collection.
    """
    fract_body_exp = 0
    ground_reflectivity = 0

    if not isinstance(surface_temperature.header.data_type, Temperature):
        surface_temperature.header.data_type = Temperature

    solar_body_par = SolarCalParameter()
    solar_mrt_obj = HorizontalSolarCal(
        epw.location,
        direct_radiation,
        diffuse_radiation,
        surface_temperature,
        fract_body_exp,
        ground_reflectivity,
        solar_body_par,
    )

    mrt = solar_mrt_obj.mean_radiant_temperature

    return mrt

if __name__ == "__main__":
    pass