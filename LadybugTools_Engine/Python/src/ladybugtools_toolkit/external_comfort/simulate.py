from __future__ import annotations

import contextlib
import getpass
import io
import itertools
import json
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import FunctionType
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.model import Model
from honeybee_energy.material.opaque import EnergyMaterial, EnergyMaterialVegetation
from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
from honeybee_energy.simulation.parameter import (
    RunPeriod,
    ShadowCalculation,
    SimulationControl,
    SimulationOutput,
    SimulationParameter,
)
from ladybug.epw import EPW, AnalysisPeriod, HourlyContinuousCollection
from ladybug.wea import Wea
from ladybug_comfort.collection.solarcal import (
    HorizontalRefSolarCal,
    HorizontalSolarCal,
    OutdoorSolarCal,
)
from ladybug_comfort.parameter.solarcal import SolarCalParameter
from lbt_recipes.recipe import Recipe, RecipeSettings
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..bhomutil.analytics import CONSOLE_LOGGER
from ..bhomutil.bhom_object import BHoMObject
from ..bhomutil.encoder import (
    BHoMEncoder,
    fix_bhom_jsondict,
    inf_dtype_to_inf_str,
    inf_str_to_inf_dtype,
)
from ..helpers import evaporative_cooling_effect, sanitise_string
from ..honeybee_extension.results import load_ill, load_res, load_sql, make_annual
from ..ladybug_extension.analysis_period import describe_analysis_period
from ..ladybug_extension.datacollection import (
    average,
    collection_from_series,
    collection_to_series,
)
from ..ladybug_extension.epw import epw_to_dataframe
from ..ladybug_extension.epw import equality as epw_eq
from ..ladybug_extension.epw import get_filename
from . import QUEENBEE_PATH
from .ground_temperature import energyplus_strings
from .material import OpaqueMaterial, OpaqueVegetationMaterial, material_from_dict
from .model import create_model
from .model import equality as model_eq
from .utci.calculate import utci


def working_directory(model: Model, create: bool = False) -> Path:
    """Get the working directory (where simulation results will be stored) for the given model, and
        create it if it doesn't already exist.

    Args:
        model (Model): A honeybee Model.
        create (bool, optional): Set to True to create the directory. Default is False.

    Returns:
        Path: The simulation directory associated with the given model.
    """

    hb_folders.default_simulation_folder = f"C:/Users/{getpass.getuser()}/simulation"

    working_dir: Path = Path(hb_folders.default_simulation_folder) / model.identifier
    if create:
        working_dir.mkdir(parents=True, exist_ok=True)

    return working_dir


def simulation_id(
    epw_file: Path,
    ground_material: Union[EnergyMaterial, EnergyMaterialVegetation],
    shade_material: Union[EnergyMaterial, EnergyMaterialVegetation],
) -> str:
    """Create an ID for a simulation.

    Args:
        epw_file (Path):
            The path to an EPW file.
        ground_material (Union[OpaqueMaterial, OpaqueVegetationMaterial]):
            A material object.
        shade_material (Union[OpaqueMaterial, OpaqueVegetationMaterial]):
            A material object.

    Returns:
        str:
            An ID for the siulation run using this combination of inputs.
    """

    epw_id = sanitise_string(Path(epw_file).stem)
    ground_material_id = sanitise_string(ground_material.identifier)
    shade_material_id = sanitise_string(shade_material.identifier)
    id_string = f"{epw_id}__{ground_material_id}__{shade_material_id}"
    if len(id_string) > 100:
        warnings.warn(
            "simulation ID would be longer than 100 characters. In order for this to work it needs to be shortened. As such it might make things break if we try to reload this configuration in the future!"
        )
        id_string = id_string[:100]
    return id_string


def surface_temperature(
    model: Model, epw: EPW
) -> Dict[str, HourlyContinuousCollection]:
    """Run EnergyPlus on a model and return the results.

    Args:
        model (Model): A honeybee Model to be run through EnergyPlus.
        epw (EPW): An EPW object to be used for the simulation.

    Returns:
        Dict[str, LBTHourlyContinuousCollection]: A dictionary containing
            surface temperature-related collections.
    """

    wd = working_directory(model, True)

    sql_path = wd / "run" / "eplusout.sql"

    if surface_temperature_results_exist(model, epw):
        CONSOLE_LOGGER.info(f"[{model.identifier}] - Loading surface temperature")
        return surface_temperature_results_load(sql_path, epw)

    epw.save((wd / get_filename(epw, True)).as_posix())

    CONSOLE_LOGGER.info(f"[{model.identifier}] - Simulating surface temperature")

    # Write model JSON
    model_dict = model.to_dict(triangulate_sub_faces=True)
    model_json = wd / f"{model.identifier}.hbjson"
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
    sim_par_json = wd / "simulation_parameter.json"
    with open(sim_par_json, "w", encoding="utf-8") as fp:
        json.dump(sim_par_dict, fp)

    # Create OpenStudio workflow
    osw = to_openstudio_osw(
        wd.as_posix(),
        model_json.as_posix(),
        sim_par_json.as_posix(),
        additional_measures=None,
        epw_file=epw.file_path,
    )

    # Convert workflow to IDF file
    _, idf = run_osw(osw, silent=False)

    # Add ground monthly ground temperature strings to IDF
    with open(idf, "r", encoding="utf-8") as fp:
        idf_string = fp.read()
    idf_string += f"\n\n{energyplus_strings(epw)}"
    with open(idf, "w", encoding="utf-8") as fp:
        idf_string = fp.write(idf_string)

    # Simulate IDF
    _, _, _, _, _ = run_idf(idf, epw.file_path, silent=False)

    return surface_temperature_results_load(sql_path, epw)


def surface_temperature_results_load(
    sql_path: Path,
    epw: EPW,
) -> Dict[str, HourlyContinuousCollection]:
    """Load results from the surface temperature simulation.

    Args:
        sql_path (Path): An SQL file containing EnergyPlus results.
        epw (EPW): An EPW file. Required to get the temperature of the
            sky for an unshaded case.

    Returns:
        Dict[str, HourlyContinuousCollection]: A dictionary containing
            surface temperature-related collections.
    """

    df = load_sql(sql_path)

    return {
        "ShadedDownTemperature": collection_from_series(
            df.filter(regex="GROUND_ZONE_UP_SHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
            .rename("Ground Temperature (C)")
        ),
        "UnshadedDownTemperature": collection_from_series(
            df.filter(regex="GROUND_ZONE_UP_UNSHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
            .rename("Ground Temperature (C)")
        ),
        "ShadedUpTemperature": collection_from_series(
            df.filter(regex="SHADE_ZONE_DOWN")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
            .rename("Sky Temperature (C)")
        ),
        "UnshadedUpTemperature": epw.sky_temperature,
    }


def surface_temperature_results_exist(model: Model, epw: EPW = None) -> bool:
    """Check whether results already exist for this configuration of model and EPW.

    Args:
        model (Model): The model to check for.
        epw (EPW): The EPW to check for.  Currently unused.

    Returns:
        bool: True if the model and EPW have already been simulated, False otherwise.
    """
    wd = working_directory(model, False)

    # Try to load existing HBJSON file and check that it matches
    try:
        existing_model = Model.from_hbjson(
            (wd / f"{model.identifier}.hbjson").as_posix()
        )
        if not model_eq(model, existing_model, include_identifier=True):
            return False
    except (FileNotFoundError, AssertionError):
        return False

    # Try to load existing EPW file and check that it matches
    try:
        existing_epw = EPW((wd / get_filename(epw, True)).as_posix())
        if not epw_eq(epw, existing_epw, include_header=True):
            return False
    except (FileNotFoundError, AssertionError):
        return False

    # Check that the output files necessary to reload exist
    if not (wd / "run" / "eplusout.sql").exists():
        return False

    return True


def radiant_temperature(
    collections: List[HourlyContinuousCollection], view_factors: List[float] = None
) -> HourlyContinuousCollection:
    """Calculate the MRT from a list of surface temperature collections, and view
        factors to each of those surfaces.

    Args:
        collections (List[HourlyContinuousCollection]): A list of hourly continuous collections.
        view_factors (List[float]): A list of view factors to each of the collections. If None, then all input collections are weighted equally.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of the effective radiant
            temperature.
    """

    if view_factors is None:
        view_factors = [1 / len(collections)] * len(collections)
    if len(collections) != len(view_factors):
        raise ValueError("The number of collections and view factors must be the same.")
    if sum(view_factors) != 1:
        raise ValueError("The sum of view factors must be 1.")

    mrt_series = (
        np.power(
            (
                np.power(
                    pd.concat([collection_to_series(i) for i in collections], axis=1)
                    + 273.15,
                    4,
                )
                * view_factors
            ).sum(axis=1),
            0.25,
        )
        - 273.15
    )
    mrt_series.name = "Radiant Temperature (C)"
    return collection_from_series(mrt_series)


def direct_sun_hours(
    model: Model,
    epw: EPW,
    analysis_period: AnalysisPeriod,
) -> pd.Series:
    """Simulate the number of direct sun hours for a HB model for a given time.

    Args:
        model (Model):
            A Honeybee model containing sensor grid.
        epw (EPW):
            An epw file object.
        analysis_period (AnalysisPeriod, optional):
            An AnalsysiPeriod, including timestep to simulate.

    Returns:
        pd.Series:
            A series containing per-sensor sunlight hours.
    """

    working_dir = working_directory(model, True)
    res_dir = working_dir / "sunlight_hours"
    res_dir.mkdir(parents=True, exist_ok=True)
    results_file: Path = (
        res_dir
        / f"{describe_analysis_period(analysis_period, save_path=True, include_timestep=True)}.res"
    )

    if results_file.exists():
        CONSOLE_LOGGER.info(
            f"[{model.identifier}] - Loading Direct Sun Hours (hours) for {describe_analysis_period(analysis_period, save_path=False, include_timestep=True)}"
        )
        return load_res(results_file).squeeze()

    CONSOLE_LOGGER.info(
        f"[{model.identifier}] - Simulating direct sun hours for {describe_analysis_period(analysis_period, save_path=False, include_timestep=True)}"
    )
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        wea = Wea.from_epw_file(
            epw.file_path, timestep=analysis_period.timestep
        ).filter_by_analysis_period(analysis_period)
    grid_name = model.properties.radiance.sensor_grids[0].identifier

    # based on time of year being simulated - adjust tree/vegetation porosity

    recipe = Recipe("direct-sun-hours")
    recipe.input_value_by_name("model", model)
    recipe.input_value_by_name("wea", wea)
    recipe.input_value_by_name("north", 0)
    recipe.input_value_by_name("timestep", analysis_period.timestep)
    recipe.input_value_by_name("grid-filter", grid_name)
    recipe_settings = RecipeSettings()
    _ = recipe.run(
        settings=recipe_settings,
        radiance_check=True,
        queenbee_path=QUEENBEE_PATH,
    )

    # move results to proper location
    pts_file = working_dir / "direct_sun_hours" / "model" / "grid" / f"{grid_name}.pts"
    res_file = (
        working_dir / "direct_sun_hours" / "results" / "cumulative" / f"{grid_name}.res"
    )

    shutil.copy(pts_file, res_dir / "sensors.pts")
    shutil.copy(res_file, results_file)

    return load_res(results_file).squeeze()


def solar_radiation(model: Model, epw: EPW) -> Dict[str, HourlyContinuousCollection]:
    """Run Radiance on a model and return the results.

    Args:
        model (Model): A honeybee Model to be run through Radiance.
        epw (EPW): An EPW object to be used for the simulation.

    Returns:
        Dict[str, LBTHourlyContinuousCollection]: A dictionary containing radiation-related
            collections.
    """

    wd = working_directory(model, True)

    if solar_radiation_results_exist(model, epw):
        CONSOLE_LOGGER.info(f"[{model.identifier}] - Loading annual irradiance")
        return solar_radiation_results_load(model)

    epw.save((wd / get_filename(epw, True)).as_posix())

    CONSOLE_LOGGER.info(f"[{model.identifier}] - Simulating annual irradiance")
    wea = Wea.from_epw_file(epw.file_path)

    recipe = Recipe("annual-irradiance")
    recipe.input_value_by_name("model", model)
    recipe.input_value_by_name("wea", wea)
    recipe.input_value_by_name("north", 0)
    recipe.input_value_by_name("timestep", 1)
    recipe.input_value_by_name("output-type", "solar")
    recipe_settings = RecipeSettings()
    _ = recipe.run(
        settings=recipe_settings,
        radiance_check=True,
        queenbee_path=QUEENBEE_PATH,
    )

    return solar_radiation_results_load(model)


def solar_radiation_results_load(model: Model) -> Dict[str, HourlyContinuousCollection]:
    """Load results from the solar radiation simulation.

    Args:
        model (Model): A honeybee Model to be run through Radiance.

    Returns:
        Dict[str, HourlyContinuousCollection]: A dictionary containing
            radiation-related collections.
    """

    wd = working_directory(model, False)

    shaded_down_direct_irradiance = collection_from_series(
        make_annual(load_ill(wd / "annual_irradiance/results/direct/SHADED_DOWN.ill"))
        .squeeze()
        .fillna(0)
        .rename("Irradiance (W/m2)")
    )
    shaded_up_direct_irradiance = collection_from_series(
        make_annual(load_ill(wd / "annual_irradiance/results/direct/SHADED_UP.ill"))
        .squeeze()
        .fillna(0)
        .rename("Irradiance (W/m2)")
    )
    unshaded_down_direct_irradiance = collection_from_series(
        make_annual(load_ill(wd / "annual_irradiance/results/direct/UNSHADED_DOWN.ill"))
        .squeeze()
        .fillna(0)
        .rename("Irradiance (W/m2)")
    )
    unshaded_up_direct_irradiance = collection_from_series(
        make_annual(load_ill(wd / "annual_irradiance/results/direct/UNSHADED_UP.ill"))
        .squeeze()
        .fillna(0)
        .rename("Irradiance (W/m2)")
    )
    shaded_down_total_irradiance = collection_from_series(
        make_annual(load_ill(wd / "annual_irradiance/results/total/SHADED_DOWN.ill"))
        .squeeze()
        .fillna(0)
        .rename("Irradiance (W/m2)")
    )
    shaded_up_total_irradiance = collection_from_series(
        make_annual(load_ill(wd / "annual_irradiance/results/total/SHADED_UP.ill"))
        .squeeze()
        .fillna(0)
        .rename("Irradiance (W/m2)")
    )
    unshaded_down_total_irradiance = collection_from_series(
        make_annual(load_ill(wd / "annual_irradiance/results/total/UNSHADED_DOWN.ill"))
        .squeeze()
        .fillna(0)
        .rename("Irradiance (W/m2)")
    )
    unshaded_up_total_irradiance = collection_from_series(
        make_annual(load_ill(wd / "annual_irradiance/results/total/UNSHADED_UP.ill"))
        .squeeze()
        .fillna(0)
        .rename("Irradiance (W/m2)")
    )
    shaded_down_diffuse_irradiance = (
        shaded_down_total_irradiance - shaded_down_direct_irradiance
    )
    shaded_up_diffuse_irradiance = (
        shaded_up_total_irradiance - shaded_up_direct_irradiance
    )
    unshaded_down_diffuse_irradiance = (
        unshaded_down_total_irradiance - unshaded_down_direct_irradiance
    )
    unshaded_up_diffuse_irradiance = (
        unshaded_up_total_irradiance - unshaded_up_direct_irradiance
    )

    # load data and return in dict
    return {
        "ShadedDownDiffuseIrradiance": shaded_down_diffuse_irradiance,
        "ShadedDownDirectIrradiance": shaded_down_direct_irradiance,
        "ShadedDownTotalIrradiance": shaded_down_total_irradiance,
        "ShadedUpDiffuseIrradiance": shaded_up_diffuse_irradiance,
        "ShadedUpDirectIrradiance": shaded_up_direct_irradiance,
        "ShadedUpTotalIrradiance": shaded_up_total_irradiance,
        "UnshadedDownDiffuseIrradiance": unshaded_down_diffuse_irradiance,
        "UnshadedDownDirectIrradiance": unshaded_down_direct_irradiance,
        "UnshadedDownTotalIrradiance": unshaded_down_total_irradiance,
        "UnshadedUpDiffuseIrradiance": unshaded_up_diffuse_irradiance,
        "UnshadedUpDirectIrradiance": unshaded_up_direct_irradiance,
        "UnshadedUpTotalIrradiance": unshaded_up_total_irradiance,
    }


def solar_radiation_results_exist(model: Model, epw: EPW = None) -> bool:
    """Check whether results already exist for this configuration of model and EPW.

    Args:
        model (Model): The model to check for.
        epw (EPW): The EPW to check for.

    Returns:
        bool: True if the model and EPW have already been simulated, False otherwise.
    """
    wd = working_directory(model, False)

    # Try to load existing HBJSON file and check that it matches
    try:
        existing_model = Model.from_hbjson(
            (wd / "annual_irradiance" / f"{model.identifier}.hbjson").as_posix()
        )
        if not model_eq(model, existing_model, include_identifier=True):
            return False
    except (FileNotFoundError, AssertionError):
        return False

    # Try to load existing EPW file and check that it matches
    try:
        existing_epw = EPW((wd / get_filename(epw, True)).as_posix())
        if not epw_eq(epw, existing_epw, include_header=True):
            return False
    except (FileNotFoundError, AssertionError):
        return False

    # Check that the output files necessary to reload exist
    if not all(
        [
            (wd / "annual_irradiance/results/direct/SHADED_DOWN.ill").exists(),
            (wd / "annual_irradiance/results/direct/SHADED_UP.ill").exists(),
            (wd / "annual_irradiance/results/direct/UNSHADED_DOWN.ill").exists(),
            (wd / "annual_irradiance/results/direct/UNSHADED_UP.ill").exists(),
            (wd / "annual_irradiance/results/total/SHADED_DOWN.ill").exists(),
            (wd / "annual_irradiance/results/total/SHADED_UP.ill").exists(),
            (wd / "annual_irradiance/results/total/UNSHADED_DOWN.ill").exists(),
            (wd / "annual_irradiance/results/total/UNSHADED_UP.ill").exists(),
        ]
    ):
        return False

    return True


@dataclass(init=True, repr=True, eq=True)
class SimulationResult(BHoMObject):
    """An object containing all the results of a mean radiant temperature
        simulation.

    Args:
        epw_file (str):
            An epw file path.
        ground_material (Union[LBTEnergyMaterialVegetation, LBTEnergyMaterial]):
            A surface material for the ground zones topmost face.
        shade_material (Union[LBTEnergyMaterialVegetation, LBTEnergyMaterial]):
            A surface material for the shade zones faces.
        identifier (str, optional):
            A unique identifier for the model. Defaults to None which will
            generate a unique identifier. This is useful for testing purposes!

    Returns:
        SimulationResult:
            An object containing all the results of a mean radiant
            temperature simulation.
    """

    EpwFile: Path = field(repr=False, compare=True)
    GroundMaterial: Union[OpaqueVegetationMaterial, OpaqueMaterial] = field(
        repr=False, compare=True
    )
    ShadeMaterial: Union[OpaqueVegetationMaterial, OpaqueMaterial] = field(
        repr=False, compare=True
    )
    Identifier: str = field(repr=True, compare=True, default=None)

    ShadedDownTemperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    ShadedUpTemperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    ShadedRadiantTemperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    ShadedLongwaveMeanRadiantTemperatureDelta: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    ShadedShortwaveMeanRadiantTemperatureDelta: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    ShadedMeanRadiantTemperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )

    UnshadedDownTemperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    UnshadedUpTemperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    UnshadedRadiantTemperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    UnshadedLongwaveMeanRadiantTemperatureDelta: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    UnshadedShortwaveMeanRadiantTemperatureDelta: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    UnshadedMeanRadiantTemperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )

    _t: str = field(
        init=False,
        repr=False,
        compare=True,
        default="BH.oM.LadybugTools.SimulationResult",
    )

    def __post_init__(self):
        self.EpwFile = Path(self.EpwFile).resolve()

        if self.Identifier is None:
            self.Identifier = simulation_id(
                self.EpwFile,
                self.GroundMaterial.to_lbt(),
                self.ShadeMaterial.to_lbt(),
            )

        # wrap methods within this class
        super().__post_init__()

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> SimulationResult:
        """Create this object from a dictionary."""

        if not isinstance(
            dictionary["GroundMaterial"], (OpaqueMaterial, OpaqueVegetationMaterial)
        ):
            dictionary["GroundMaterial"] = material_from_dict(
                dictionary["GroundMaterial"]
            )

        if not isinstance(
            dictionary["ShadeMaterial"], (OpaqueMaterial, OpaqueVegetationMaterial)
        ):
            dictionary["ShadeMaterial"] = material_from_dict(
                dictionary["ShadeMaterial"]
            )

        # handle objects
        for simulated_result in [
            "ShadedDownTemperature",
            "ShadedUpTemperature",
            "ShadedRadiantTemperature",
            "ShadedLongwaveMeanRadiantTemperatureDelta",
            "ShadedShortwaveMeanRadiantTemperatureDelta",
            "ShadedMeanRadiantTemperature",
            "UnshadedDownTemperature",
            "UnshadedUpTemperature",
            "UnshadedRadiantTemperature",
            "UnshadedLongwaveMeanRadiantTemperatureDelta",
            "UnshadedShortwaveMeanRadiantTemperatureDelta",
            "UnshadedMeanRadiantTemperature",
        ]:
            if dictionary[simulated_result] is None:
                continue
            if isinstance(dictionary[simulated_result], HourlyContinuousCollection):
                continue
            if isinstance(dictionary[simulated_result], dict):
                if "type" in dictionary[simulated_result].keys():
                    dictionary[simulated_result] = HourlyContinuousCollection.from_dict(
                        dictionary[simulated_result]
                    )
                else:
                    dictionary[simulated_result] = None

        return cls(
            EpwFile=dictionary["EpwFile"],
            GroundMaterial=dictionary["GroundMaterial"],
            ShadeMaterial=dictionary["ShadeMaterial"],
            Identifier=dictionary["Identifier"],
            ShadedDownTemperature=dictionary["ShadedDownTemperature"],
            ShadedUpTemperature=dictionary["ShadedUpTemperature"],
            ShadedRadiantTemperature=dictionary["ShadedRadiantTemperature"],
            ShadedLongwaveMeanRadiantTemperatureDelta=dictionary[
                "ShadedLongwaveMeanRadiantTemperatureDelta"
            ],
            ShadedShortwaveMeanRadiantTemperatureDelta=dictionary[
                "ShadedShortwaveMeanRadiantTemperatureDelta"
            ],
            ShadedMeanRadiantTemperature=dictionary["ShadedMeanRadiantTemperature"],
            UnshadedDownTemperature=dictionary["UnshadedDownTemperature"],
            UnshadedUpTemperature=dictionary["UnshadedUpTemperature"],
            UnshadedRadiantTemperature=dictionary["UnshadedRadiantTemperature"],
            UnshadedLongwaveMeanRadiantTemperatureDelta=dictionary[
                "UnshadedLongwaveMeanRadiantTemperatureDelta"
            ],
            UnshadedShortwaveMeanRadiantTemperatureDelta=dictionary[
                "UnshadedShortwaveMeanRadiantTemperatureDelta"
            ],
            UnshadedMeanRadiantTemperature=dictionary["UnshadedMeanRadiantTemperature"],
        )

    @classmethod
    def from_json(cls, json_string: str) -> SimulationResult:
        """Create this object from a JSON string."""

        dictionary = inf_str_to_inf_dtype(
            json.loads(json_string, object_hook=fix_bhom_jsondict)
        )
        return cls.from_dict(dictionary)

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as it's dictionary equivalent."""
        dictionary = {}
        for k, v in self.__dict__.items():
            if isinstance(getattr(self, k), FunctionType):
                continue
            dictionary[k] = v
        dictionary["_t"] = self._t
        return dictionary

    def to_json(self) -> str:
        """Return this object as it's JSON string equivalent."""
        return json.dumps(inf_dtype_to_inf_str(self.to_dict()), cls=BHoMEncoder)

    @property
    def epw(self) -> EPW:
        """Return the EPW opject loaded from the associated EPW file."""
        return EPW(self.EpwFile)

    @property
    def model(self) -> Model:
        """Return the model object created from the associated materials."""
        return create_model(
            self.GroundMaterial.to_lbt(), self.ShadeMaterial.to_lbt(), self.Identifier
        )

    @property
    def epw_file(self) -> Path:
        """Handy accessor using proper Python naming style."""
        return self.EpwFile

    @property
    def ground_material(self) -> Union[OpaqueMaterial, OpaqueVegetationMaterial]:
        """Handy accessor using proper Python naming style."""
        return self.GroundMaterial

    @property
    def shade_material(self) -> Union[OpaqueMaterial, OpaqueVegetationMaterial]:
        """Handy accessor using proper Python naming style."""
        return self.ShadeMaterial

    @property
    def identifier(self) -> str:
        """Handy accessor using proper Python naming style."""
        return self.Identifier

    @property
    def shaded_down_temperature(self) -> HourlyContinuousCollection:
        """Handy accessor using proper Python naming style."""
        return self.ShadedDownTemperature

    @property
    def shaded_up_temperature(self) -> HourlyContinuousCollection:
        """Handy accessor using proper Python naming style."""
        return self.ShadedUpTemperature

    @property
    def shaded_radiant_temperature(self) -> HourlyContinuousCollection:
        """Handy accessor using proper Python naming style."""
        return self.ShadedRadiantTemperature

    @property
    def shaded_longwave_mean_radiant_temperature_delta(
        self,
    ) -> HourlyContinuousCollection:
        """Handy accessor using proper Python naming style."""
        return self.ShadedLongwaveMeanRadiantTemperatureDelta

    @property
    def shaded_shortwave_mean_radiant_temperature_delta(
        self,
    ) -> HourlyContinuousCollection:
        """Handy accessor using proper Python naming style."""
        return self.ShadedShortwaveMeanRadiantTemperatureDelta

    @property
    def shaded_mean_radiant_temperature(self) -> HourlyContinuousCollection:
        """Handy accessor using proper Python naming style."""
        return self.ShadedMeanRadiantTemperature

    @property
    def unshaded_down_temperature(self) -> HourlyContinuousCollection:
        """Handy accessor using proper Python naming style."""
        return self.UnshadedDownTemperature

    @property
    def unshaded_up_temperature(self) -> HourlyContinuousCollection:
        """Handy accessor using proper Python naming style."""
        return self.UnshadedUpTemperature

    @property
    def unshaded_radiant_temperature(self) -> HourlyContinuousCollection:
        """Handy accessor using proper Python naming style."""
        return self.UnshadedRadiantTemperature

    @property
    def unshaded_longwave_mean_radiant_temperature_delta(
        self,
    ) -> HourlyContinuousCollection:
        """Handy accessor using proper Python naming style."""
        return self.UnshadedLongwaveMeanRadiantTemperatureDelta

    @property
    def unshaded_shortwave_mean_radiant_temperature_delta(
        self,
    ) -> HourlyContinuousCollection:
        """Handy accessor using proper Python naming style."""
        return self.UnshadedShortwaveMeanRadiantTemperatureDelta

    @property
    def unshaded_mean_radiant_temperature(self) -> HourlyContinuousCollection:
        """Handy accessor using proper Python naming style."""
        return self.UnshadedMeanRadiantTemperature

    def is_run(self) -> bool:
        """Helper method to determine whether this object is populated with simulation results."""
        variables = [
            "ShadedDownTemperature",
            "ShadedUpTemperature",
            "ShadedRadiantTemperature",
            "ShadedLongwaveMeanRadiantTemperatureDelta",
            "ShadedShortwaveMeanRadiantTemperatureDelta",
            "ShadedMeanRadiantTemperature",
            "UnshadedDownTemperature",
            "UnshadedUpTemperature",
            "UnshadedRadiantTemperature",
            "UnshadedLongwaveMeanRadiantTemperatureDelta",
            "UnshadedShortwaveMeanRadiantTemperatureDelta",
            "UnshadedMeanRadiantTemperature",
        ]
        if any(getattr(self, i) is None for i in variables):
            return False
        return True

    def run(self) -> SimulationResult:
        """Run this SimulationResult and return the populated object."""

        # create object to populate
        sim_res = SimulationResult(
            self.EpwFile, self.GroundMaterial, self.ShadeMaterial
        )

        # create referenced objects
        epw = self.epw
        model = self.model

        # calculate surrounding surface temperatures
        results = surface_temperature(model, epw)

        # populate simulated variables
        for k, v in results.items():
            setattr(sim_res, k, v)

        # calculate other variables
        sim_res.ShadedRadiantTemperature = radiant_temperature(
            [
                sim_res.ShadedDownTemperature,
                sim_res.ShadedUpTemperature,
            ],
            [0.5, 0.5],
        )
        sim_res.UnshadedRadiantTemperature = radiant_temperature(
            [
                sim_res.UnshadedDownTemperature,
                sim_res.UnshadedUpTemperature,
            ],
            [0.5, 0.5],
        )

        # calculate MRT
        params = SolarCalParameter()

        shaded_cal = OutdoorSolarCal(
            location=epw.location,
            direct_normal_solar=epw.direct_normal_radiation,
            diffuse_horizontal_solar=epw.diffuse_horizontal_radiation,
            horizontal_infrared=epw.horizontal_infrared_radiation_intensity,
            surface_temperatures=sim_res.ShadedRadiantTemperature,
            floor_reflectance=model.faces_by_identifier(["GROUND_ZONE_UP_SHADED"])[
                0
            ].properties.energy.construction.outside_solar_reflectance,
            sky_exposure=0,
            fraction_body_exposed=0,
            solarcal_body_parameter=params,
        )
        sim_res.ShadedMeanRadiantTemperature = shaded_cal.mean_radiant_temperature
        sim_res.ShadedShortwaveMeanRadiantTemperatureDelta = (
            shaded_cal.shortwave_mrt_delta
        )
        sim_res.ShadedLongwaveMeanRadiantTemperatureDelta = (
            shaded_cal.longwave_mrt_delta
        )

        unshaded_cal = OutdoorSolarCal(
            location=epw.location,
            direct_normal_solar=epw.direct_normal_radiation,
            diffuse_horizontal_solar=epw.diffuse_horizontal_radiation,
            horizontal_infrared=epw.horizontal_infrared_radiation_intensity,
            surface_temperatures=sim_res.UnshadedDownTemperature,
            floor_reflectance=model.faces_by_identifier(["GROUND_ZONE_UP_SHADED"])[
                0
            ].properties.energy.construction.outside_solar_reflectance,
            sky_exposure=1,
            fraction_body_exposed=1,
            solarcal_body_parameter=params,
        )
        sim_res.UnshadedMeanRadiantTemperature = unshaded_cal.mean_radiant_temperature
        sim_res.UnshadedShortwaveMeanRadiantTemperatureDelta = (
            unshaded_cal.shortwave_mrt_delta
        )
        sim_res.UnshadedLongwaveMeanRadiantTemperatureDelta = (
            unshaded_cal.longwave_mrt_delta
        )

        return sim_res

    def to_dataframe(
        self, include_epw: bool = False, include_epw_additional: bool = False
    ) -> pd.DataFrame:
        """Create a Pandas DataFrame from this object.

        Args:
            include_epw (bool, optional):
                Set to True to include the dataframe for the EPW file also.
            include_epw_additional (bool, optional): Set to True to also include calculated
                values such as sun position along with EPW. Only includes if include_epw is
                True also.

        Returns:
            pd.DataFrame: A Pandas DataFrame with this objects properties.
        """

        if not self.is_run():
            raise AttributeError(f'This {self.__class__.__name__} has not been "run"!')

        obj_series = []
        for k, v in self.to_dict().items():
            if isinstance(v, HourlyContinuousCollection):
                series = collection_to_series(v)
                category = "Shaded" if k.lower().startswith("shaded") else "Unshaded"
                obj_series.append(
                    series.rename((self.Identifier, category, f"{series.name} ({k})"))
                )
        obj_df = pd.concat(obj_series, axis=1)

        if include_epw:
            return pd.concat(
                [
                    epw_to_dataframe(self.epw, include_epw_additional),
                    obj_df,
                ],
                axis=1,
            )

        return obj_df

    def ranked_mitigations(
        self,
        n_steps: int = 8,
        analysis_period: AnalysisPeriod = None,
        comfort_limits: Tuple[float] = (9, 26),
        evaporative_cooling_effectiveness: float = 0.7,
    ) -> pd.DataFrame:
        """Determine the relative impact of different measures to adjust UTCI.

        Args:
            n_steps (int, optional):
                The number of steps to calculate per variable (n_steps**3 is the number of calculations run). Defaults to 8.
            analysis_period (AnalysisPeriod, optional):
                A period to apply to results. Defaults to None.
            comfort_limits (Tuple[float], optional):
                Optional limits to set what is considered comfortable. Defaults to (9, 26).

        Returns:
            pd.DataFrame:
                A table of relative UTCI impact proportions.
        """

        # TODO - break method out into parts - namely basic inputs and single Series output, referenced from elsewhere

        if analysis_period is None:
            analysis_period = AnalysisPeriod()

        # get comfort limits as single values
        comfort_low = min(comfort_limits)
        comfort_mid = np.mean(comfort_limits)
        comfort_high = max(comfort_limits)

        # construct dataframe containing inputs to this process
        epw = self.epw
        atm = collection_to_series(epw.atmospheric_station_pressure).rename("atm")
        dbt = collection_to_series(epw.dry_bulb_temperature).rename("dbt")
        rh = collection_to_series(epw.relative_humidity).rename("rh")
        ws = collection_to_series(epw.wind_speed).rename("ws")
        mrt_unshaded = collection_to_series(self.UnshadedMeanRadiantTemperature).rename(
            "mrt_unshaded"
        )
        mrt_shaded = collection_to_series(self.ShadedMeanRadiantTemperature).rename(
            "mrt_shaded"
        )
        utci_unshaded = utci(dbt, rh, mrt_unshaded, ws).rename("utci_unshaded")
        df = pd.concat(
            [atm, dbt, rh, ws, mrt_unshaded, mrt_shaded, utci_unshaded], axis=1
        )

        # filter by analysis period
        df = df.iloc[list(analysis_period.hoys_int)]

        # get comfort mask for baseline
        df["utci_unshaded_comfortable"] = df.utci_unshaded.between(
            comfort_low, comfort_high
        )

        # get distance from comfortable (midpoint) for each timestep in baseline
        df["utci_unshaded_distance_from_comfortable_midpoint"] = (
            df.utci_unshaded - comfort_mid
        )
        df[
            "utci_unshaded_distance_from_comfortable"
        ] = df.utci_unshaded_distance_from_comfortable_midpoint.where(
            ~df.utci_unshaded_comfortable, 0
        )

        # get possible values for shade/shelter/evapclg
        shading_proportions = np.linspace(1, 0, n_steps)
        wind_shelter_proportions = np.linspace(0, 1, n_steps)
        evap_clg_proportions = np.linspace(0, 1, n_steps)

        def _temp(
            dbt,
            rh,
            atm,
            ws,
            mrt_unshaded,
            mrt_shaded,
            utci_unshaded_distance_from_comfortable_midpoint,
            name,
        ):
            # create feasible ranges of values
            dbts, rhs = np.array(
                [
                    evaporative_cooling_effect(
                        dbt, rh, evap_x * evaporative_cooling_effectiveness, atm
                    )
                    for evap_x in evap_clg_proportions
                ]
            ).T
            wss = ws * wind_shelter_proportions
            mrts = np.interp(shading_proportions, [0, 1], [mrt_unshaded, mrt_shaded])

            # create all possible combinations of inputs
            dbts, rhs, mrts, wss = np.array(
                list(itertools.product(dbts, rhs, mrts, wss))
            ).T
            utcis = utci([dbts], [rhs], [mrts], [wss])[0]
            shad_props, _, windshlt_props, evapclg_props = np.array(
                list(
                    itertools.product(
                        shading_proportions,
                        shading_proportions,
                        wind_shelter_proportions,
                        evap_clg_proportions,
                    )
                )
            ).T

            # reshape matrix
            mtx = pd.DataFrame(
                [
                    dbts,
                    rhs,
                    mrts,
                    wss,
                    utcis,
                    shad_props,
                    windshlt_props,
                    evapclg_props,
                ],
                index=[
                    "dbt",
                    "rh",
                    "mrt",
                    "ws",
                    "utci",
                    "shade",
                    "wind shelter",
                    "evaporative cooling",
                ],
            ).T

            # get comfort mask for current timestep
            mtx["utci_comfortable"] = mtx.utci.between(comfort_low, comfort_high)

            # get distance from comfortable (midpoint) for current timestep
            mtx["utci_distance_from_comfortable_midpoint"] = mtx.utci - comfort_mid
            mtx[
                "utci_distance_from_comfortable"
            ] = mtx.utci_distance_from_comfortable_midpoint.where(
                ~mtx.utci_comfortable, 0
            )

            # determine whether comfort has improved, and drop rows where it hasnt
            mtx["comfort_improved"] = abs(
                mtx.utci_distance_from_comfortable_midpoint
            ) < abs(utci_unshaded_distance_from_comfortable_midpoint)
            mtx = mtx[mtx.comfort_improved]

            # sort by distance to comfort midpoint, to get optimal conditions
            mtx["utci_distance_from_comfortable_midpoint_absolute"] = abs(
                mtx.utci_distance_from_comfortable_midpoint
            )
            mtx = mtx.sort_values("utci_distance_from_comfortable_midpoint_absolute")

            # normalise variables
            temp = (
                mtx[["shade", "wind shelter", "evaporative cooling"]]
                / mtx[["shade", "wind shelter", "evaporative cooling"]].sum(axis=0)
            ).reset_index(drop=True)

            # get topmost 25%
            ranks = temp.head(int(len(temp) / 4)).mean(axis=0)
            ranks = ranks / ranks.sum()
            ranks.name = name

            # TODO - include "do nothing" as an option for ranking!

            return ranks

        tqdm.pandas(
            desc=f"Calculating ranked beneficial impact of comfort mitigation measures for {epw}"
        )
        return df.progress_apply(
            lambda row: _temp(
                row.dbt,
                row.rh,
                row.atm,
                row.ws,
                row.mrt_unshaded,
                row.mrt_shaded,
                row.utci_unshaded_distance_from_comfortable_midpoint,
                row.name,
            ),
            axis=1,
        )

    @staticmethod
    def plot_ranked_mitigations(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
        raise NotImplementedError("Not yet done, also, should probably go elsewhere!")
