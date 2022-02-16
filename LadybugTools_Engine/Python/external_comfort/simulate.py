import json
import os
from pathlib import Path
from typing import Dict, List

from lbt_recipes.recipe import Recipe
from honeybee.model import Model
from honeybee.config import folders as hb_folders
from ladybug.epw import EPW, HourlyContinuousCollection, AnalysisPeriod
from ladybug.wea import Wea
from honeybee_energy.simulation.parameter import (
    RunPeriod,
    ShadowCalculation,
    SimulationControl,
    SimulationOutput,
    SimulationParameter,
)
from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
import numpy as np
from ladybug_comfort.collection.solarcal import HorizontalSolarCal
from ladybug_comfort.parameter.solarcal import SolarCalParameter


from honeybee_extension.results import load_sql, load_ill, _make_annual
from ladybug_extension.datacollection import from_series
from external_comfort import QUEENBEE_EXE
from external_comfort.ground_temperature import energyplus_ground_temperature_strings

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
        fp.writelines(temp + [energyplus_ground_temperature_strings(epw)])

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
        "shaded_ground_temperature": from_series(
            df.filter(regex="GROUND_ZONE_UP_SHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
        ),
        "unshaded_ground_temperature": from_series(
            df.filter(regex="GROUND_ZONE_UP_UNSHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
        ),
        "shade_temperature": from_series(
            df.filter(regex="SHADE_ZONE_DOWN").droplevel([0, 1, 2], axis=1).squeeze()
        ),
    }


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

    wea = Wea.from_epw_file(epw.file_path)

    recipe = Recipe("annual-irradiance")
    recipe.input_value_by_name("model", model)
    recipe.input_value_by_name("wea", wea)
    # recipe.input_value_by_name("radiance-parameters", "-ab 2 -ad 128 -lw 2e-05")
    # recipe.default_project_folder = str(working_directory)

    results = recipe.run(queenbee_path=QUEENBEE_EXE)

    total_directory = Path(results) / "annual_irradiance/results/total"
    direct_directory = Path(results) / "annual_irradiance/results/direct"

    unshaded_total = _make_annual(load_ill(total_directory / "UNSHADED.ill"))
    unshaded_direct = _make_annual(load_ill(direct_directory / "UNSHADED.ill"))
    unshaded_diffuse = unshaded_total - unshaded_direct

    unshaded_total = (
        _make_annual(load_ill(total_directory / "UNSHADED.ill"))
        .fillna(0)
        .sum(axis=1)
        .rename("GlobalHorizontalRadiation (Wh/m2)")
    )

    unshaded_direct = (
        _make_annual(load_ill(direct_directory / "SHADED.ill"))
        .fillna(0)
        .sum(axis=1)
        .rename("DirectNormalRadiation (Wh/m2)")
    )

    unshaded_diffuse = (unshaded_total - unshaded_direct).rename(
        "DiffuseHorizontalRadiation (Wh/m2)"
    )

    return {
        "unshaded_direct_radiation": from_series(unshaded_direct),
        "unshaded_diffuse_radiation": from_series(unshaded_diffuse),
    }


def _mean_radiant_temperature_from_surfaces(
    temperatures: List[float], view_factors: List[float]
) -> float:
    """Calculate Mean Radiant Temperature from a list of surface temperature and view factors to those surfaces.

    Args:
        temperatures (List[float]): A list of surface temperatures.
        view_factors (List[float]): A list of view-factors (one per surface)

    Returns:
        float: A value describing resultant radiant temperature.
    """
    resultant_temperature = 0
    for i, temp in enumerate(temperatures):
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