import json
from typing import Dict

from honeybee.model import Model
from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
from honeybee_energy.simulation.parameter import (RunPeriod, ShadowCalculation,
                                                  SimulationControl,
                                                  SimulationOutput,
                                                  SimulationParameter)
from ladybug.epw import EPW, AnalysisPeriod, HourlyContinuousCollection
from ladybugtools_toolkit import analytics
from ladybugtools_toolkit.external_comfort.ground_temperature.eplus_otherside_coefficient import \
    eplus_otherside_coefficient
from ladybugtools_toolkit.external_comfort.simulate.surface_temperature_results_exist import \
    surface_temperature_results_exist
from ladybugtools_toolkit.external_comfort.simulate.surface_temperature_results_load import \
    surface_temperature_results_load
from ladybugtools_toolkit.external_comfort.simulate.working_directory import \
    working_directory as wd
from ladybugtools_toolkit.ladybug_extension.epw.filename import filename


@analytics
def surface_temperature(
    model: Model, epw: EPW
) -> Dict[str, HourlyContinuousCollection]:
    """Run EnergyPlus on a model and return the results.

    Args:
        model (Model): A honeybee Model to be run through EnergyPlus.
        epw (EPW): An EPW object to be used for the simulation.

    Returns:
        Dict[str, HourlyContinuousCollection]: A dictionary containing surface temperature-related
            collections.
    """

    working_directory = wd(model, True)

    sql_path = working_directory / "run" / "eplusout.sql"

    if surface_temperature_results_exist(model, epw):
        print(f"[{model.identifier}] - Loading surface temperature")
        return surface_temperature_results_load(sql_path, epw)

    epw.save((working_directory / filename(epw, True)).as_posix())

    print(f"[{model.identifier}] - Simulating surface temperature")

    # Write model JSON
    model_dict = model.to_dict(triangulate_sub_faces=True)
    model_json = working_directory / f"{model.identifier}.hbjson"
    with open(model_json, "w", encoding="utf-8") as fp:
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
    with open(sim_par_json, "w", encoding="utf-8") as fp:
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
    with open(idf, "r", encoding="utf-8") as fp:
        idf_string = fp.read()
    idf_string = idf_string.replace(
        "Ground,                                 !- Outside Boundary Condition",
        "OtherSideCoefficients,                  !- Outside Boundary Condition",
    )
    idf_string = idf_string.replace(
        ",                                       !- Outside Boundary Condition Object",
        "GroundTemperature,                      !- Outside Boundary Condition Object",
    )
    idf_string += f"\n\n{eplus_otherside_coefficient(epw)}"
    with open(idf, "w", encoding="utf-8") as fp:
        idf_string = fp.write(idf_string)

    # Simulate IDF
    _, _, _, _, _ = run_idf(idf, epw.file_path, silent=False)

    return surface_temperature_results_load(sql_path, epw)
