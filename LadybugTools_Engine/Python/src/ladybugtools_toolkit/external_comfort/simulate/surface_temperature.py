import getpass
import json
from pathlib import Path

from honeybee.config import folders as hb_folders
from honeybee.model import Model
from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
from honeybee_energy.simulation.parameter import (
    RunPeriod,
    ShadowCalculation,
    SimulationControl,
    SimulationOutput,
    SimulationParameter,
)
from ladybug.epw import EPW, AnalysisPeriod

from ..ground_temperature import energyplus_strings_otherside_coefficient

hb_folders.default_simulation_folder = f"C:/Users/{getpass.getuser()}/simulation"


def surface_temperature(model: Model, epw: EPW) -> Path:
    """Run EnergyPlus on a model and return the results.

    Args:
        model (Model): A honeybee Model to be run through EnergyPlus.
        epw (EPW): An EPW object to be used for the simulation.

    Returns:
        A dictionary containing ground and shade (below and above) surface temperature values.
    """

    working_directory: Path = (
        Path(hb_folders.default_simulation_folder) / model.identifier
    )
    working_directory.mkdir(parents=True, exist_ok=True)
    sql_path = working_directory / "run" / "eplusout.sql"

    print("- Simulating surface temperatures")
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
    idf_string += f"\n\n{energyplus_strings_otherside_coefficient(epw)}"
    with open(idf, "w", encoding="utf-8") as fp:
        idf_string = fp.write(idf_string)

    # Simulate IDF
    _, _, _, _, _ = run_idf(idf, epw.file_path, silent=False)

    # save EPW to working directory
    epw.save(working_directory / Path(epw.file_path).name)

    return sql_path
