"""Module for the external comfort package, handling simulation of shaded/
unshaded surface temperatures in an abstract "openfield" condition."""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.model import Model
from honeybee_energy.material.opaque import (
    EnergyMaterial,
    EnergyMaterialVegetation,
    _EnergyMaterialOpaqueBase,
)
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

from ..bhom import decorator_factory, keys_to_pascalcase, keys_to_snakecase
from ..honeybee_extension.results import load_sql
from ..ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)
from ..ladybug_extension.epw import epw_to_dataframe
from ..ladybug_extension.epw import equality as epw_equality
from .ground_temperature import energyplus_strings
from .model import create_model, get_ground_reflectance, model_equality


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


@decorator_factory(disable=False)
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
                # print(
                #     f"{epw} is not the same as {EPW(existing_epw)} ({existing_epw.relative_to(sim_dir)}) and will be removed"
                # )
                existing_epw.unlink()
    saved_epw = (sim_dir / epw_file.name).as_posix()
    epw.save(saved_epw)

    # do the models match
    models_match = False
    existing_models = list(sim_dir.glob("*.hbjson"))
    if len(existing_models) >= 1:
        for existing_model in existing_models:
            if model_equality(
                model, Model.from_hbjson(existing_model), include_identifier=True
            ):
                # print(
                #     f"{model} is the same as {Model.from_hbjson(existing_model)} ({existing_model.relative_to(sim_dir)})"
                # )
                models_match = True
            else:
                # print(
                #     f"{model} is not the same as {Model.from_hbjson(existing_model)} ({existing_model.relative_to(sim_dir)}) and will be removed"
                # )
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
        # print("Simulating ...")  # TODO - replace with console logger
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
                calculation_update_method="Periodic",  # NOTE - changed from Timestep ... lets see how this impacts things
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
        # print("Reloading ...")  # TODO - replace with console logger
        pass
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


@decorator_factory(disable=False)
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


# pylint: disable=too-few-public-methods
class SimulationResultDecoder(json.JSONEncoder):
    def default(self, o):
        match o:
            case o if isinstance(o, dict):
                if "type" in o:
                    if o["type"] == "HourlyContinuous":
                        return HourlyContinuousCollection.from_dict(o)
                    if o["type"] == "EnergyMaterial":
                        return EnergyMaterial.from_dict(o)
                    if o["type"] == "EnergyMaterialVegetation":
                        return EnergyMaterialVegetation.from_dict(o)
                    raise ValueError(f"Unknown \"type\": {o['type']}")
            case _:
                return super().default(o)


class SimulationResultEncoder(json.JSONEncoder):
    def default(self, o):
        match o:
            case o if isinstance(
                o, (EPW, HourlyContinuousCollection, _EnergyMaterialOpaqueBase)
            ):
                return o.to_dict()
            case _:
                return super().default(o)


# pylint: enable=too-few-public-methods


@dataclass(init=True, repr=True, eq=True)
class SimulationResult:
    """An object containing all the results of a mean radiant temperature
        simulation. This object uses the SolarCal algorithm to calculate the
        mean radiant temperature.

    References:
        Arens, E., Hoyt, T., Zhou, X., Huang, L., Zhang, H., & Schiavon, S.
        (2015). Modeling the comfort effects of short-wave solar radiation
        indoors. In Building and Environment (Vol. 88, pp. 3-9). Elsevier BV.
        https://doi.org/10.1016/j.buildenv.2014.09.004


    Args:
        epw_file (str):
            An epw file path.
        ground_material (_EnergyMaterialOpaqueBase):
            A surface material for the ground zones topmost face.
        shade_material (_EnergyMaterialOpaqueBase):
            A surface material for the shade zones faces.

    Returns:
        SimulationResult:
            An object containing all the results of a mean radiant
            temperature simulation.
    """

    epw_file: Path = field(repr=False, compare=True)
    ground_material: _EnergyMaterialOpaqueBase = field(repr=False, compare=True)
    shade_material: _EnergyMaterialOpaqueBase = field(repr=False, compare=True)
    identifier: str = field(init=True, repr=True, compare=True, default=None)

    shaded_down_temperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    shaded_up_temperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )

    unshaded_down_temperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    unshaded_up_temperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )

    shaded_radiant_temperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    shaded_longwave_mean_radiant_temperature_delta: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    shaded_shortwave_mean_radiant_temperature_delta: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    shaded_mean_radiant_temperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )

    unshaded_radiant_temperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )
    unshaded_longwave_mean_radiant_temperature_delta: HourlyContinuousCollection = (
        field(init=True, repr=False, compare=False, default=None)
    )
    unshaded_shortwave_mean_radiant_temperature_delta: HourlyContinuousCollection = (
        field(init=True, repr=False, compare=False, default=None)
    )
    unshaded_mean_radiant_temperature: HourlyContinuousCollection = field(
        init=True, repr=False, compare=False, default=None
    )

    _t: str = field(
        init=False,
        repr=False,
        compare=True,
        default="BH.oM.LadybugTools.SimulationResult",
    )

    def __post_init__(self):
        self.epw_file = Path(self.epw_file).resolve()
        _epw = self.epw
        _model = self.model

        # run simulation and populate object with results if not already done
        if not all(
            [
                isinstance(self.shaded_down_temperature, HourlyContinuousCollection),
                isinstance(self.unshaded_down_temperature, HourlyContinuousCollection),
                isinstance(self.shaded_up_temperature, HourlyContinuousCollection),
                isinstance(self.unshaded_up_temperature, HourlyContinuousCollection),
            ]
        ):
            results = simulate_surface_temperatures(
                model=_model,
                epw_file=self.epw_file,
                remove_dir=True if self.identifier is None else False,
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

        # populate datacollections as pandas series property accessors
        for var in dir(self):
            if var.startswith("_"):
                continue
            _temp = getattr(self, var)
            if isinstance(_temp, HourlyContinuousCollection):
                setattr(self, f"{var}_series", collection_to_series(_temp))

    @property
    def epw(self) -> EPW:
        """Return the EPW object associated with this simulation result."""
        return EPW(self.epw_file)

    @property
    def model(self) -> Model:
        """Return the model object for this simulation result."""
        return create_model(
            self.ground_material,
            self.shade_material,
            str(uuid.uuid4()) if self.identifier is None else self.identifier,
        )

    @decorator_factory(disable=False)
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

        obj_series = []
        for var in dir(self):
            for shadedness in ["shaded", "unshaded"]:
                if var.startswith(shadedness):
                    _temp = getattr(self, var)
                    if isinstance(_temp, pd.Series):
                        obj_series.append(
                            _temp.rename(
                                (self.identifier, shadedness.title(), _temp.name)
                            )
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

    @decorator_factory(disable=False)
    def to_dict(self) -> dict[str, Any]:
        """Create a dictionary representation of this object.

        Returns:
            dict: A dictionary representation of this object.
        """

        obj_dict = {}
        for var in dir(self):
            if isinstance(
                getattr(self, var),
                (HourlyContinuousCollection, _EnergyMaterialOpaqueBase),
            ):
                obj_dict[var] = getattr(self, var).to_dict()
            if var in ["_t", "identifier", "epw_file"]:
                obj_dict[var] = getattr(self, var)

        return keys_to_pascalcase(obj_dict)

    @classmethod
    @decorator_factory(disable=False)
    def from_dict(cls, dictionary: dict[str, Any]):
        """Create this object from its dictionary representation.

        Args:
            dictionary (dict): A dictionary representation of this object.

        Returns:
            obj: This object.
        """

        dictionary.pop("_t", None)
        dictionary = keys_to_snakecase(dictionary)

        # iterate over elements of dictionary and convert where necessary
        for k, v in dictionary.items():
            if isinstance(v, dict):
                if "type" in v:
                    if v["type"] == "HourlyContinuous":
                        dictionary[k] = HourlyContinuousCollection.from_dict(v)
                    elif v["type"] == "EnergyMaterial":
                        dictionary[k] = EnergyMaterial.from_dict(v)
                    elif v["type"] == "EnergyMaterialVegetation":
                        dictionary[k] = EnergyMaterialVegetation.from_dict(v)
                    else:
                        raise ValueError(f"Unknown type: {v['type']}")

        return cls(**dictionary)

    @decorator_factory(disable=False)
    def to_json(self, **kwargs) -> str:
        """Create a JSON representation of this object.

        Keyword Args:
            kwargs: Additional keyword arguments to pass to json.dumps.

        Returns:
            str: A JSON representation of this object.
        """

        return json.dumps(self.to_dict(), cls=SimulationResultEncoder, **kwargs)

    @classmethod
    @decorator_factory(disable=False)
    def from_json(cls, json_string: str):
        """Create this object from its JSON representation.

        Args:
            json_string (str): A JSON representation of this object.

        Returns:
            obj: This object.
        """

        return cls.from_dict(json.loads(json_string, cls=SimulationResultDecoder))
