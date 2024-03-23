"""Module for the external comfort package, handling simulation of shaded/
unshaded surface temperatures in an abstract "openfield" condition."""
# pylint: disable=E0401
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

# pylint: enable=E0401
from caseconverter import pascalcase
import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.model import Model
from honeybee_energy.dictutil import dict_to_material
from honeybee_energy.material.opaque import EnergyMaterial, EnergyMaterialVegetation
from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
from honeybee_energy.simulation.parameter import (
    ShadowCalculation,
    SimulationControl,
    SimulationOutput,
    SimulationParameter,
)
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug.futil import nukedir
from ladybug_comfort.collection.solarcal import OutdoorSolarCal, SolarCalParameter

from ..bhom.logging import CONSOLE_LOGGER
from ..bhom.to_bhom import (
    hourlycontinuouscollection_to_bhom,
    material_to_bhom,
)
from ..honeybee_extension.results import load_sql
from ..ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)
from ..ladybug_extension.epw import epw_to_dataframe
from ..ladybug_extension.epw import equality as epw_equality
from ..ladybug_extension.groundtemperature import energyplus_strings
from .model import create_model, get_ground_reflectance, model_equality
from ..helpers import convert_keys_to_snake_case, sanitise_string
from .material import Materials


def simulation_id(
    epw_file: Path,
    ground_material: EnergyMaterial | EnergyMaterialVegetation,
    shade_material: EnergyMaterial | EnergyMaterialVegetation,
) -> str:
    """Create an ID for a simulation.

    Args:
        epw_file (Path): The path to the EPW file.
        ground_material (EnergyMaterial | EnergyMaterialVegetation): The ground material.
        shade_material (EnergyMaterial | EnergyMaterialVegetation): The shade material.

    Returns:
        str: The simulation ID.
    """

    epw_id = sanitise_string(epw_file.stem)
    ground_material_id = sanitise_string(ground_material.identifier)
    shade_material_id = sanitise_string(shade_material.identifier)
    return f"{epw_id}__{ground_material_id}__{shade_material_id}"


def simulation_directory(model: Model) -> Path:
    """Get the working directory (where simulation results will be stored) for the given model, and
        create it if it doesn't already exist.

    Args:
        model (Model): A honeybee Model.

    Returns:
        Path: The simulation directory associated with the given model.
    """

    working_dir: Path = Path(hb_folders.default_simulation_folder) / model.identifier
    working_dir.mkdir(parents=True, exist_ok=True)

    return working_dir


def simulate_surface_temperatures(
    model: Model, epw_file: Path, remove_dir: bool = False
) -> dict[str, HourlyContinuousCollection]:
    """Simulate surface temperatures for a Honeybee Model and return the
        resulting SQL results file path.

    Args:
        model (Model): A honeybee Model.
        epw_file (Path): The path to an EPW file.
        remove_dir (bool, optional): Set to True to remove the simulation
            directory

    Returns:
        dict[str, HourlyContinuousCollection]: Surface temperature results.
    """

    if not isinstance(model, Model):
        raise ValueError("model must be a Honeybee Model.")

    epw_file = Path(epw_file)
    if not epw_file.exists():
        raise ValueError("epw_file must be a valid file path.")
    epw = EPW(epw_file)

    sim_dir = simulation_directory(model)

    # does the epw file already exist in the sim dir
    epws_match = False
    existing_epws = list(sim_dir.glob("*.epw"))
    if len(existing_epws) >= 1:
        for existing_epw in existing_epws:
            if epw_equality(epw, EPW(existing_epw), include_header=True):
                # print(
                #     f"{epw} is the same as {EPW(existing_epw)} ({existing_epw.relative_to(sim_dir)})"
                # )
                epws_match = True
            else:
                existing_epw.unlink()
    saved_epw = (sim_dir / epw_file.name).as_posix()
    epw.save(saved_epw)

    # do the models match
    models_match = False
    existing_models = list(sim_dir.glob("*.hbjson"))
    if len(existing_models) >= 1:
        for existing_model in existing_models:
            if model_equality(model, Model.from_hbjson(existing_model)):
                models_match = True
            else:
                existing_model.unlink()

    # does the sql_path exist
    sql_path = sim_dir / "run" / "eplusout.sql"
    sql_exists = sql_path.exists()

    # check for existing results and reload if they exist
    if not all(
        [
            sql_exists,
            models_match,
            epws_match,
        ]
    ):
        CONSOLE_LOGGER.info(f"Simulating {model.identifier}")
        model_json = sim_dir / f"{model.identifier}.hbjson"
        with open(model_json, "w", encoding="utf-8") as fp:
            json.dump(model.to_dict(triangulate_sub_faces=True), fp)

        sim_par = SimulationParameter(
            output=SimulationOutput(
                outputs=["Surface Outside Face Temperature"],
                include_sqlite=True,
                summary_reports=None,
                include_html=False,
            ),
            simulation_control=SimulationControl(
                do_zone_sizing=False,
                do_system_sizing=False,
                do_plant_sizing=False,
                run_for_sizing_periods=False,
                run_for_run_periods=True,
            ),
            shadow_calculation=ShadowCalculation(
                solar_distribution="FullExteriorWithReflections",
                calculation_method="PolygonClipping",
                calculation_update_method="Periodic",
                maximum_figures=200,
            ),
            terrain_type="Country",
            timestep=10,
        )
        sim_par_json = sim_dir / "simulation_parameter.json"
        with open(sim_par_json, "w", encoding="utf-8") as fp:
            json.dump(sim_par.to_dict(), fp)

        osw = to_openstudio_osw(
            sim_dir.as_posix(),
            model_json.as_posix(),
            sim_par_json.as_posix(),
            additional_measures=None,
            epw_file=epw.file_path,
        )

        _, idf = run_osw(osw, silent=True)

        with open(idf, "r", encoding="utf-8") as fp:
            idf_string = fp.read()
        idf_string += f"\n\n{energyplus_strings(epw)}"

        with open(idf, "w", encoding="utf-8") as fp:
            idf_string = fp.write(idf_string)

        run_idf(idf, epw.file_path, silent=False)

    else:
        CONSOLE_LOGGER.info(f"Reloading {model.identifier}")

    df = load_sql(sql_path)

    if remove_dir:
        nukedir(sim_dir, rmdir=True)

    return {
        "shaded_down_temperature": collection_from_series(
            df.filter(regex="GROUND_ZONE_UP_SHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
            .rename("Ground Temperature (C)")
        ),
        "unshaded_down_temperature": collection_from_series(
            df.filter(regex="GROUND_ZONE_UP_UNSHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
            .rename("Ground Temperature (C)")
        ),
        "shaded_up_temperature": collection_from_series(
            df.filter(regex="SHADE_ZONE_DOWN")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
            .rename("Sky Temperature (C)")
        ),
        "unshaded_up_temperature": epw.sky_temperature,
    }


def radiant_temperature(
    collections: list[HourlyContinuousCollection], view_factors: list[float] = None
) -> HourlyContinuousCollection:
    """Calculate the MRT from a list of surface temperature collections, and view
        factors to each of those surfaces.

    Args:
        collections (List[HourlyContinuousCollection]):
            A list of hourly continuous collections.
        view_factors (List[float]):
            A list of view factors to each of the collections.
            If None, then all input collections are weighted equally.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of the effective radiant temperature.
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


_ATTRIBUTES = [
    "shaded_down_temperature",
    "shaded_up_temperature",
    "unshaded_down_temperature",
    "unshaded_up_temperature",
    "shaded_radiant_temperature",
    "shaded_longwave_mean_radiant_temperature_delta",
    "shaded_shortwave_mean_radiant_temperature_delta",
    "shaded_mean_radiant_temperature",
    "unshaded_radiant_temperature",
    "unshaded_longwave_mean_radiant_temperature_delta",
    "unshaded_shortwave_mean_radiant_temperature_delta",
    "unshaded_mean_radiant_temperature",
]


@dataclass(init=True, repr=True, eq=True)
class SimulationResult:
    """_"""

    epw_file: Path
    ground_material: EnergyMaterial | EnergyMaterialVegetation
    shade_material: EnergyMaterial | EnergyMaterialVegetation
    identifier: str = None

    shaded_down_temperature: HourlyContinuousCollection = None
    shaded_up_temperature: HourlyContinuousCollection = None

    unshaded_down_temperature: HourlyContinuousCollection = None
    unshaded_up_temperature: HourlyContinuousCollection = None

    shaded_radiant_temperature: HourlyContinuousCollection = None
    shaded_longwave_mean_radiant_temperature_delta: HourlyContinuousCollection = None
    shaded_shortwave_mean_radiant_temperature_delta: HourlyContinuousCollection = None
    shaded_mean_radiant_temperature: HourlyContinuousCollection = None

    unshaded_radiant_temperature: HourlyContinuousCollection = None
    unshaded_longwave_mean_radiant_temperature_delta: HourlyContinuousCollection = None
    unshaded_shortwave_mean_radiant_temperature_delta: HourlyContinuousCollection = None
    unshaded_mean_radiant_temperature: HourlyContinuousCollection = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.identifier})"

    def __post_init__(self):
        """_"""

        # validation
        if not isinstance(self.epw_file, (Path, str)):
            raise ValueError("epw_file must be a Path or str.")
        self.epw_file = Path(self.epw_file).resolve()
        if not self.epw_file.exists():
            raise ValueError("epw_file does not exist.")

        if isinstance(self.ground_material, Materials):
            self.ground_material = self.ground_material.value
        if isinstance(self.shade_material, Materials):
            self.shade_material = self.shade_material.value

        if not isinstance(
            self.ground_material, (EnergyMaterial, EnergyMaterialVegetation)
        ):
            raise ValueError(
                "ground_material must be an EnergyMaterial or EnergyMaterialVegetation."
            )
        if not isinstance(
            self.shade_material, (EnergyMaterial, EnergyMaterialVegetation)
        ):
            raise ValueError(
                "shade_material must be an EnergyMaterial or EnergyMaterialVegetation."
            )

        if self.identifier is None:
            self.identifier = simulation_id(
                self.epw_file, self.ground_material, self.shade_material
            )

        for attr in _ATTRIBUTES:
            if not isinstance(
                getattr(self, attr), (HourlyContinuousCollection, type(None))
            ):
                raise ValueError(
                    f"{attr} must be either an HourlyContinuousCollection, or None."
                )

        # run simulation and populate object with results if not already done
        _epw = EPW(self.epw_file)
        _model = create_model(
            identifier=self.identifier,
            ground_material=self.ground_material,
            shade_material=self.shade_material,
        )

        if not all(
            [
                self.shaded_down_temperature,
                self.unshaded_down_temperature,
                self.shaded_up_temperature,
                self.unshaded_up_temperature,
            ]
        ):
            results = simulate_surface_temperatures(
                model=_model,
                epw_file=self.epw_file,
                remove_dir=not bool(self.identifier),
            )
            for k, v in results.items():
                if isinstance(getattr(self, k), HourlyContinuousCollection):
                    continue
                setattr(self, k, v)

        # calculate other variables
        self.shaded_radiant_temperature = radiant_temperature(
            [
                self.shaded_down_temperature,
                self.shaded_up_temperature,
            ],
        )
        self.unshaded_radiant_temperature = radiant_temperature(
            [
                self.unshaded_down_temperature,
                self.unshaded_up_temperature,
            ],
        )

        # calculate MRT
        params = SolarCalParameter()
        shaded_cal = OutdoorSolarCal(
            location=_epw.location,
            direct_normal_solar=_epw.direct_normal_radiation,
            diffuse_horizontal_solar=_epw.diffuse_horizontal_radiation,
            horizontal_infrared=_epw.horizontal_infrared_radiation_intensity,
            surface_temperatures=self.shaded_radiant_temperature,
            floor_reflectance=get_ground_reflectance(_model),
            sky_exposure=0,
            fraction_body_exposed=0,
            solarcal_body_parameter=params,
        )
        unshaded_cal = OutdoorSolarCal(
            location=_epw.location,
            direct_normal_solar=_epw.direct_normal_radiation,
            diffuse_horizontal_solar=_epw.diffuse_horizontal_radiation,
            horizontal_infrared=_epw.horizontal_infrared_radiation_intensity,
            surface_temperatures=self.unshaded_down_temperature,
            floor_reflectance=get_ground_reflectance(_model),
            sky_exposure=1,
            fraction_body_exposed=1,
            solarcal_body_parameter=params,
        )
        for shadedness, cal in list(
            zip(*[["shaded", "unshaded"], [shaded_cal, unshaded_cal]])
        ):
            for var in [
                "mean_radiant_temperature",
                "shortwave_mrt_delta",
                "longwave_mrt_delta",
            ]:
                setattr(
                    self,
                    f"{shadedness}_{var.replace('mrt', 'mean_radiant_temperature')}",
                    getattr(cal, var),
                )

        # add some accessors for collections as series
        # TODO - replace with @property matehods for each attribute to save on overwritten valeus being carried throught downstream operaitons
        for attr in _ATTRIBUTES:
            setattr(self, f"{attr}_series", collection_to_series(getattr(self, attr)))

    def to_dict(self) -> dict[str, Any]:
        """Convert this object to a dictionary."""
        ground_material_dict = self.ground_material.to_dict()
        shade_material_dict = self.shade_material.to_dict()

        attr_dict = {}
        for attr in _ATTRIBUTES:
            if getattr(self, attr):
                attr_dict[attr] = getattr(self, attr).to_dict()

        d = {
            **{
                "type": "SimulationResult",
                "epw_file": self.epw_file.as_posix(),
                "ground_material": ground_material_dict,
                "shade_material": shade_material_dict,
                "identifier": self.identifier,
            },
            **attr_dict,
        }

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimulationResult":
        """Create this object from a dictionary."""
        if isinstance(d["ground_material"], dict):
            d["ground_material"] = dict_to_material(d["ground_material"])

        if isinstance(d["shade_material"], dict):
            d["shade_material"] = dict_to_material(d["shade_material"])

        for attr in _ATTRIBUTES:
            if d.get(attr, None):
                if isinstance(d[attr], dict):
                    d[attr] = HourlyContinuousCollection.from_dict(d[attr])
            else:
                d[attr] = None

        return cls(
            epw_file=d["epw_file"],
            ground_material=d["ground_material"],
            shade_material=d["shade_material"],
            identifier=d["identifier"],
            shaded_down_temperature=d["shaded_down_temperature"],
            shaded_up_temperature=d["shaded_up_temperature"],
            unshaded_down_temperature=d["unshaded_down_temperature"],
            unshaded_up_temperature=d["unshaded_up_temperature"],
            shaded_radiant_temperature=d["shaded_radiant_temperature"],
            shaded_longwave_mean_radiant_temperature_delta=d[
                "shaded_longwave_mean_radiant_temperature_delta"
            ],
            shaded_shortwave_mean_radiant_temperature_delta=d[
                "shaded_shortwave_mean_radiant_temperature_delta"
            ],
            shaded_mean_radiant_temperature=d["shaded_mean_radiant_temperature"],
            unshaded_radiant_temperature=d["unshaded_radiant_temperature"],
            unshaded_longwave_mean_radiant_temperature_delta=d[
                "unshaded_longwave_mean_radiant_temperature_delta"
            ],
            unshaded_shortwave_mean_radiant_temperature_delta=d[
                "unshaded_shortwave_mean_radiant_temperature_delta"
            ],
            unshaded_mean_radiant_temperature=d["unshaded_mean_radiant_temperature"],
        )

    def to_json(self) -> str:
        """Create a JSON string from this object."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_string: str) -> "SimulationResult":
        """Create this object from a JSON string."""

        return cls.from_dict(json.loads(json_string))

    def to_file(self, path: Path) -> Path:
        """Write this object to a JSON file."""

        if Path(path).suffix != ".json":
            raise ValueError("path must be a JSON file.")

        with open(Path(path), "w") as fp:
            fp.write(self.to_json())

        return Path(path)

    @classmethod
    def from_file(cls, path: Path) -> "SimulationResult":
        """Create this object from a JSON file."""

        with open(Path(path), "r") as fp:
            return cls.from_json(fp.read())

    @property
    def epw(self) -> EPW:
        """Return the EPW object associated with this simulation result."""
        return EPW(self.epw_file)

    @property
    def simulation_directory(self) -> Path:
        """Return the simulation directory for this simulation result."""
        return simulation_directory(self.model)

    @property
    def model(self) -> Model:
        """Return the model object for this simulation result."""
        return create_model(
            identifier=self.identifier,
            ground_material=self.ground_material,
            shade_material=self.shade_material,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Create a Pandas DataFrame from this object.

        Returns:
            pd.DataFrame: Represent this object as a Pandas DataFrame.
        """

        obj_series = []
        for var in dir(self):
            for shadedness in ["shaded", "unshaded"]:
                if not var.startswith(shadedness):
                    continue
                _temp = getattr(self, var)
                if isinstance(_temp, HourlyContinuousCollection):
                    _temp = collection_to_series(_temp)
                    _temp.rename(
                        (shadedness.title(), _temp.name),
                        inplace=True,
                    )
                    obj_series.append(_temp)

        obj_df = pd.concat(obj_series, axis=1)

        return pd.concat(
            [
                pd.concat(
                    [epw_to_dataframe(self.epw, include_additional=True)],
                    axis=1,
                    keys=["EPW"],
                ),
                obj_df,
            ],
            axis=1,
        )

    def description(self, include_shade_material: bool = True) -> str:
        """Create the description for this object.

        Args:
            include_shade_material (bool, optional):
                Set to False to exclude the shade material from the description.
                Defaults to True.

        Returns:
            str:
                A description of this object.
        """
        if include_shade_material:
            return (
                f"{self.epw_file.name} - "
                f"{self.ground_material.identifier} (ground material) - "
                f"{self.shade_material.identifier} (shade material)"
            )

        return (
            f"{self.epw_file.name} - "
            f"{self.ground_material.identifier} (ground material)"
        )
