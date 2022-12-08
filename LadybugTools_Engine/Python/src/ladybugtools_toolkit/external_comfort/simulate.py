from __future__ import annotations

import contextlib
import getpass
import io
import itertools
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
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
from ladybug.epw import EPW, AnalysisPeriod, HourlyContinuousCollection
from ladybug.wea import Wea
from ladybug_comfort.collection.solarcal import HorizontalSolarCal
from ladybug_comfort.parameter.solarcal import SolarCalParameter
from lbt_recipes.recipe import Recipe, RecipeSettings
from tqdm import tqdm

from ..bhomutil.analytics import CONSOLE_LOGGER
from ..bhomutil.bhom_object import BHoMObject, bhom_dict_to_dict
from ..helpers import sanitise_string
from ..honeybee_extension.results import load_ill, load_res, load_sql, make_annual
from ..ladybug_extension.analysis_period import describe as describe_analysis_period
from ..ladybug_extension.datacollection import from_series, to_series
from ..ladybug_extension.epw import equality as epw_eq
from ..ladybug_extension.epw import filename, to_dataframe
from . import QUEENBEE_PATH
from .ground_temperature import eplus_otherside_coefficient
from .material import OpaqueMaterial, OpaqueVegetationMaterial, material_from_dict
from .model import create_model
from .model import equality as model_eq
from .moisture import evaporative_cooling_effect
from .utci import utci


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
    ground_material: Union[OpaqueMaterial, OpaqueVegetationMaterial],
    shade_material: Union[OpaqueMaterial, OpaqueVegetationMaterial],
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
    return f"{epw_id}__{ground_material_id}__{shade_material_id}"


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

    epw.save((wd / filename(epw, True)).as_posix())

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
        "shaded_below_temperature": from_series(
            df.filter(regex="GROUND_ZONE_UP_SHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
            .rename("Ground Temperature (C)")
        ),
        "unshaded_below_temperature": from_series(
            df.filter(regex="GROUND_ZONE_UP_UNSHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
            .rename("Ground Temperature (C)")
        ),
        "shaded_above_temperature": from_series(
            df.filter(regex="SHADE_ZONE_DOWN")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
            .rename("Sky Temperature (C)")
        ),
        "unshaded_above_temperature": epw.sky_temperature,
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
        existing_epw = EPW((wd / filename(epw, True)).as_posix())
        if not epw_eq(epw, existing_epw, include_header=True):
            return False
    except (FileNotFoundError, AssertionError):
        return False

    # Check that the output files necessary to reload exist
    if not (wd / "run" / "eplusout.sql").exists():
        return False

    return True


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

    total_irradiance = wd / "annual_irradiance/results/total/UNSHADED.ill"
    direct_irradiance = wd / "annual_irradiance/results/direct/UNSHADED.ill"

    if solar_radiation_results_exist(model, epw):
        CONSOLE_LOGGER.info(f"[{model.identifier}] - Loading annual irradiance")
        return solar_radiation_results_load(total_irradiance, direct_irradiance)

    epw.save((wd / filename(epw, True)).as_posix())

    CONSOLE_LOGGER.info(f"[{model.identifier}] - Simulating annual irradiance")
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
        queenbee_path=QUEENBEE_PATH,
    )

    return solar_radiation_results_load(total_irradiance, direct_irradiance)


def solar_radiation_results_load(
    total_irradiance: Path, direct_irradiance: Path
) -> Dict[str, HourlyContinuousCollection]:
    """Load results from the solar radiation simulation.

    Args:
        total_irradiance (Path): An ILL file containing total irradiance.
        direct_irradiance (Path): An ILL file containing direct irradiance.

    Returns:
        Dict[str, HourlyContinuousCollection]: A dictionary containing
            radiation-related collections.
    """

    unshaded_total = (
        make_annual(load_ill(total_irradiance))
        .fillna(0)
        .sum(axis=1)
        .rename("GlobalHorizontalRadiation (Wh/m2)")
    )
    unshaded_direct = (
        make_annual(load_ill(direct_irradiance))
        .fillna(0)
        .sum(axis=1)
        .rename("DirectNormalRadiation (Wh/m2)")
    )
    unshaded_diffuse = (unshaded_total - unshaded_direct).rename(
        "DiffuseHorizontalRadiation (Wh/m2)"
    )
    _shaded = pd.Series(
        [0] * len(unshaded_total), index=unshaded_total.index, name="base"
    )

    return {
        "unshaded_direct_radiation": from_series(unshaded_direct),
        "unshaded_diffuse_radiation": from_series(unshaded_diffuse),
        "unshaded_total_radiation": from_series(unshaded_total),
        "shaded_direct_radiation": from_series(
            _shaded.rename("DirectNormalRadiation (Wh/m2)")
        ),
        "shaded_diffuse_radiation": from_series(
            _shaded.rename("DiffuseHorizontalRadiation (Wh/m2)")
        ),
        "shaded_total_radiation": from_series(
            _shaded.rename("GlobalHorizontalRadiation (Wh/m2)")
        ),
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
        existing_epw = EPW((wd / filename(epw, True)).as_posix())
        if not epw_eq(epw, existing_epw, include_header=True):
            return False
    except (FileNotFoundError, AssertionError):
        return False

    # Check that the output files necessary to reload exist
    if not all(
        [
            (wd / "annual_irradiance/results/direct/UNSHADED.ill").exists(),
            (wd / "annual_irradiance/results/total/UNSHADED.ill").exists(),
        ]
    ):
        return False

    return True


def longwave_radiant_temperature(
    collections: List[HourlyContinuousCollection], view_factors: List[float]
) -> HourlyContinuousCollection:
    """Calculate the LW MRT from a list of surface temperature collections, and view
        factors to each of those surfaces.

    Args:
        collections (List[HourlyContinuousCollection]): A list of hourly continuous collections.
        view_factors (List[float]): A list of view factors to each of the collections.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of the effective radiant
            temperature.
    """

    if len(collections) != len(view_factors):
        raise ValueError("The number of collections and view factors must be the same.")
    if sum(view_factors) != 1:
        raise ValueError("The sum of view factors must be 1.")

    mrt_series = (
        np.power(
            (
                np.power(
                    pd.concat([to_series(i) for i in collections], axis=1) + 273.15,
                    4,
                )
                * view_factors
            ).sum(axis=1),
            0.25,
        )
        - 273.15
    )
    mrt_series.name = "Radiant Temperature (C)"
    return from_series(mrt_series)


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

    epw_file: Path = field(repr=True, compare=True)
    ground_material: Union[OpaqueVegetationMaterial, OpaqueMaterial] = field(
        repr=True, compare=True
    )
    shade_material: Union[OpaqueVegetationMaterial, OpaqueMaterial] = field(
        repr=True, compare=True
    )
    identifier: str = field(repr=True, compare=True, default=None)

    shaded_above_temperature: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    shaded_below_temperature: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    shaded_diffuse_radiation: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    shaded_direct_radiation: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    shaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    shaded_mean_radiant_temperature: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    shaded_shortwave_mean_radiant_temperature: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    shaded_total_radiation: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    unshaded_above_temperature: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    unshaded_below_temperature: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    unshaded_diffuse_radiation: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    unshaded_direct_radiation: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    unshaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    unshaded_mean_radiant_temperature: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    unshaded_shortwave_mean_radiant_temperature: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )
    unshaded_total_radiation: HourlyContinuousCollection = field(
        repr=False,
        init=True,
        compare=False,
        default=None,
    )

    _t: str = field(
        init=False,
        repr=False,
        compare=True,
        default="BH.oM.LadybugTools.SimulationResult",
    )

    def __post_init__(self):

        self.epw_file = Path(self.epw_file).absolute()

        if self.identifier is None:
            self.identifier = simulation_id(
                self.epw_file,
                self.ground_material.to_lbt(),
                self.shade_material.to_lbt(),
            )

        # wrap methods within this class
        super().__post_init__()

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> SimulationResult:
        """Create this object from a dictionary."""

        sanitised_dict = bhom_dict_to_dict(dictionary)
        sanitised_dict.pop("_t", None)

        # handle objects
        for material in ("ground_material", "shade_material"):
            if isinstance(sanitised_dict[material], dict):
                sanitised_dict[material] = material_from_dict(sanitised_dict[material])

        for simulated_result in [
            "shaded_above_temperature",
            "shaded_below_temperature",
            "shaded_diffuse_radiation",
            "shaded_direct_radiation",
            "shaded_longwave_mean_radiant_temperature",
            "shaded_mean_radiant_temperature",
            "shaded_shortwave_mean_radiant_temperature",
            "shaded_total_radiation",
            "unshaded_above_temperature",
            "unshaded_below_temperature",
            "unshaded_diffuse_radiation",
            "unshaded_direct_radiation",
            "unshaded_longwave_mean_radiant_temperature",
            "unshaded_mean_radiant_temperature",
            "unshaded_shortwave_mean_radiant_temperature",
            "unshaded_total_radiation",
        ]:
            if isinstance(sanitised_dict[simulated_result], dict):
                if "type" in sanitised_dict[simulated_result].keys():
                    sanitised_dict[
                        simulated_result
                    ] = HourlyContinuousCollection.from_dict(
                        sanitised_dict[simulated_result]
                    )
                else:
                    sanitised_dict[simulated_result] = None

        return cls(
            epw_file=sanitised_dict["epw_file"],
            ground_material=sanitised_dict["ground_material"],
            shade_material=sanitised_dict["shade_material"],
            identifier=sanitised_dict["identifier"],
            shaded_above_temperature=sanitised_dict["shaded_above_temperature"],
            shaded_below_temperature=sanitised_dict["shaded_below_temperature"],
            shaded_diffuse_radiation=sanitised_dict["shaded_diffuse_radiation"],
            shaded_direct_radiation=sanitised_dict["shaded_direct_radiation"],
            shaded_longwave_mean_radiant_temperature=sanitised_dict[
                "shaded_longwave_mean_radiant_temperature"
            ],
            shaded_mean_radiant_temperature=sanitised_dict[
                "shaded_mean_radiant_temperature"
            ],
            shaded_shortwave_mean_radiant_temperature=sanitised_dict[
                "shaded_shortwave_mean_radiant_temperature"
            ],
            shaded_total_radiation=sanitised_dict["shaded_total_radiation"],
            unshaded_above_temperature=sanitised_dict["unshaded_above_temperature"],
            unshaded_below_temperature=sanitised_dict["unshaded_below_temperature"],
            unshaded_diffuse_radiation=sanitised_dict["unshaded_diffuse_radiation"],
            unshaded_direct_radiation=sanitised_dict["unshaded_direct_radiation"],
            unshaded_longwave_mean_radiant_temperature=sanitised_dict[
                "unshaded_longwave_mean_radiant_temperature"
            ],
            unshaded_mean_radiant_temperature=sanitised_dict[
                "unshaded_mean_radiant_temperature"
            ],
            unshaded_shortwave_mean_radiant_temperature=sanitised_dict[
                "unshaded_shortwave_mean_radiant_temperature"
            ],
            unshaded_total_radiation=sanitised_dict["unshaded_total_radiation"],
        )

    @classmethod
    def from_json(cls, json_string: str) -> SimulationResult:
        """Create this object from a JSON string."""

        dictionary = json.loads(json_string)

        return cls.from_dict(dictionary)

    @property
    def epw(self) -> EPW:
        """Return the EPW opject loaded from the associated EPW file."""
        return EPW(self.epw_file)

    @property
    def model(self) -> Model:
        """Return the model object created from the associated materials."""
        return create_model(
            self.ground_material.to_lbt(), self.shade_material.to_lbt(), self.identifier
        )

    def is_run(self) -> bool:
        """Helper method to determine whether this object is populated with simulation results."""
        variables = [
            "shaded_above_temperature",
            "shaded_below_temperature",
            "shaded_diffuse_radiation",
            "shaded_direct_radiation",
            "shaded_longwave_mean_radiant_temperature",
            "shaded_mean_radiant_temperature",
            "shaded_shortwave_mean_radiant_temperature",
            "shaded_total_radiation",
            "unshaded_above_temperature",
            "unshaded_below_temperature",
            "unshaded_diffuse_radiation",
            "unshaded_direct_radiation",
            "unshaded_longwave_mean_radiant_temperature",
            "unshaded_mean_radiant_temperature",
            "unshaded_shortwave_mean_radiant_temperature",
            "unshaded_total_radiation",
        ]
        if any(getattr(self, i) is None for i in variables):
            return False
        return True

    def run(self) -> SimulationResult:
        """Run this SimulationResult and return the populated object."""

        # create object to populate
        sim_res = SimulationResult(
            self.epw_file, self.ground_material, self.shade_material
        )

        # create referenced objects
        epw = self.epw
        model = self.model

        # get surface temperature and radiation simulation results
        solar_radiation_results = solar_radiation(model, epw)
        surface_temperature_results = surface_temperature(model, epw)

        # populate simulated variables
        sim_res.unshaded_direct_radiation = solar_radiation_results[
            "unshaded_direct_radiation"
        ]
        sim_res.unshaded_diffuse_radiation = solar_radiation_results[
            "unshaded_diffuse_radiation"
        ]
        sim_res.unshaded_total_radiation = solar_radiation_results[
            "unshaded_total_radiation"
        ]
        sim_res.shaded_direct_radiation = solar_radiation_results[
            "shaded_direct_radiation"
        ]
        sim_res.shaded_diffuse_radiation = solar_radiation_results[
            "shaded_diffuse_radiation"
        ]
        sim_res.shaded_total_radiation = solar_radiation_results[
            "shaded_total_radiation"
        ]

        sim_res.shaded_below_temperature = surface_temperature_results[
            "shaded_below_temperature"
        ]
        sim_res.unshaded_below_temperature = surface_temperature_results[
            "unshaded_below_temperature"
        ]
        sim_res.shaded_above_temperature = surface_temperature_results[
            "shaded_above_temperature"
        ]
        sim_res.unshaded_above_temperature = surface_temperature_results[
            "unshaded_above_temperature"
        ]

        # calculate other variables
        sim_res.shaded_longwave_mean_radiant_temperature = longwave_radiant_temperature(
            [
                sim_res.shaded_below_temperature,
                sim_res.shaded_above_temperature,
            ],
            [0.5, 0.5],
        )
        sim_res.unshaded_longwave_mean_radiant_temperature = (
            longwave_radiant_temperature(
                [
                    sim_res.unshaded_below_temperature,
                    sim_res.unshaded_above_temperature,
                ],
                [0.5, 0.5],
            )
        )

        # calculate MRT for shaded/unshaded
        solar_body_par = SolarCalParameter()

        fract_body_exp = 1  # represents fully exposed human
        ground_reflectivity = 0  # 0 as Radiance includes ground reflectivity

        _shaded = HorizontalSolarCal(
            epw.location,
            sim_res.shaded_direct_radiation,
            sim_res.shaded_diffuse_radiation,
            sim_res.shaded_longwave_mean_radiant_temperature,
            fract_body_exp,
            ground_reflectivity,
            solar_body_par,
        )
        _unshaded = HorizontalSolarCal(
            epw.location,
            sim_res.unshaded_direct_radiation,
            sim_res.unshaded_diffuse_radiation,
            sim_res.unshaded_longwave_mean_radiant_temperature,
            fract_body_exp,
            ground_reflectivity,
            solar_body_par,
        )

        sim_res.shaded_shortwave_mean_radiant_temperature = _shaded.mrt_delta
        sim_res.unshaded_shortwave_mean_radiant_temperature = _unshaded.mrt_delta

        sim_res.shaded_mean_radiant_temperature = _shaded.mean_radiant_temperature
        sim_res.unshaded_mean_radiant_temperature = _unshaded.mean_radiant_temperature

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
                series = to_series(v)
                category = "Shaded" if k.lower().startswith("shaded") else "Unshaded"
                obj_series.append(
                    series.rename((self.identifier, category, series.name))
                )
        obj_df = pd.concat(obj_series, axis=1)

        if include_epw:
            return pd.concat(
                [
                    to_dataframe(self.epw, include_epw_additional),
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
    ) -> pd.DataFrame:

        # TODO - break method out into parts - namely basic inputs and single Series output, referenced from elsewhere

        if analysis_period is None:
            analysis_period = AnalysisPeriod()

        # calculate point-in-time baseline UTCI
        df = self.to_dataframe(True, True).droplevel(0, axis=1)
        df[("Unshaded", "Universal Thermal Climate Index (C)")] = utci(
            df["EPW"]["Dry Bulb Temperature (C)"],
            df["EPW"]["Relative Humidity (%)"],
            df["Unshaded"]["Mean Radiant Temperature (C)"],
            df["EPW"]["Wind Speed (m/s)"],
        )

        # create range of proportions from which to apply comfort mitigation measures
        shading_proportions = np.linspace(1, 0, n_steps)
        wind_shelter_proportions = np.linspace(0, 1, n_steps)
        evap_clg_proportions = np.linspace(0, 1, n_steps)

        def _temp(row: pd.Series, idx: Any, comfort_limits: Tuple) -> pd.Series:
            dbt = row["EPW"]["Dry Bulb Temperature (C)"]
            atm = row["EPW"]["Atmospheric Station Pressure (Pa)"]
            rh = row["EPW"]["Relative Humidity (%)"]
            ws = row["EPW"]["Wind Speed (m/s)"]
            mrt_shaded = row["Shaded"]["Mean Radiant Temperature (C)"]
            mrt_unshaded = row["Unshaded"]["Mean Radiant Temperature (C)"]
            openfield_utci = row["Unshaded"]["Universal Thermal Climate Index (C)"]

            # if (openfield_utci > min(comfort_limits)) & (
            #     openfield_utci < max(comfort_limits)
            # ):
            #     return pd.Series(
            #         index=["shade", "wind shelter", "evaporative cooling"],
            #         data=np.nan,
            #         name=idx,
            #     )

            # create feasible ranges of values
            dbts, rhs = np.array(
                [
                    evaporative_cooling_effect(dbt, rh, evap_x * 0.7, atm)
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

            # get the distance to comfortable, based on the comfort limits
            mtx["comfortable_distance"] = abs(
                np.where(
                    utcis < min(comfort_limits),
                    -(min(comfort_limits) - utcis),
                    np.where(
                        utcis > max(comfort_limits), utcis - max(comfort_limits), 0
                    ),
                )
            )

            # sort values in df based on distance to comfortable - and get the top 25% of results
            sorted_mtx = mtx.sort_values("comfortable_distance").head(int(len(mtx) / 4))

            # normalise rank
            ranks = sorted_mtx.mean()[["shade", "wind shelter", "evaporative cooling"]]
            ranks = ranks / ranks.sum()
            ranks.name = idx

            return ranks

        ranked_mitigation = []
        for idx, row in tqdm(list(df.iloc[list(analysis_period.hoys_int)].iterrows())):
            ranked_mitigation.append(_temp(row, idx, comfort_limits))

        return pd.concat(ranked_mitigation, axis=1).T
