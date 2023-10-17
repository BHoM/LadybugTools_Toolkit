"""Module for the external comfort package, handling simulation of shaded/
unshaded surface temperatures in an abstract "openfield" condition."""
# pylint: disable=E0401
import json
from pathlib import Path
from typing import Optional

# pylint: enable=E0401

import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.model import Model
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
from pydantic import BaseModel, Field, root_validator, validator

from ..bhom import decorator_factory, CONSOLE_LOGGER
from ..honeybee_extension.results import load_sql
from ..ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)
from ..ladybug_extension.epw import epw_to_dataframe
from ..ladybug_extension.epw import equality as epw_equality
from ..ladybug_extension.groundtemperature import energyplus_strings
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


class SimulationResult(BaseModel):
    """_"""

    epw_file: Path = Field(alias="EpwFile")
    ground_material: EnergyMaterial | EnergyMaterialVegetation = Field(
        alias="GroundMaterial"
    )
    shade_material: EnergyMaterial | EnergyMaterialVegetation = Field(
        alias="ShadeMaterial"
    )
    identifier: Optional[str] = Field(alias="Identifier", default="unnamed")

    shaded_down_temperature: Optional[HourlyContinuousCollection] = Field(
        default=None, alias="ShadedDownTemperature", repr=False
    )
    shaded_up_temperature: Optional[HourlyContinuousCollection] = Field(
        default=None, alias="ShadedUpTemperature", repr=False
    )

    unshaded_down_temperature: Optional[HourlyContinuousCollection] = Field(
        default=None, alias="UnshadedDownTemperature", repr=False
    )
    unshaded_up_temperature: Optional[HourlyContinuousCollection] = Field(
        default=None, alias="UnshadedUpTemperature", repr=False
    )

    shaded_radiant_temperature: Optional[HourlyContinuousCollection] = Field(
        default=None, alias="ShadedRadiantTemperature", repr=False
    )
    shaded_longwave_mean_radiant_temperature_delta: Optional[
        HourlyContinuousCollection
    ] = Field(
        default=None, alias="ShadedLongwaveMeanRadiantTemperatureDelta", repr=False
    )
    shaded_shortwave_mean_radiant_temperature_delta: Optional[
        HourlyContinuousCollection
    ] = Field(
        default=None, alias="ShadedShortwaveMeanRadiantTemperatureDelta", repr=False
    )
    shaded_mean_radiant_temperature: Optional[HourlyContinuousCollection] = Field(
        default=None, alias="ShadedMeanRadiantTemperature", repr=False
    )

    unshaded_radiant_temperature: Optional[HourlyContinuousCollection] = Field(
        default=None, alias="UnshadedRadiantTemperature", repr=False
    )
    unshaded_longwave_mean_radiant_temperature_delta: Optional[
        HourlyContinuousCollection
    ] = Field(
        default=None, alias="UnshadedLongwaveMeanRadiantTemperatureDelta", repr=False
    )
    unshaded_shortwave_mean_radiant_temperature_delta: Optional[
        HourlyContinuousCollection
    ] = Field(
        default=None, alias="UnshadedShortwaveMeanRadiantTemperatureDelta", repr=False
    )
    unshaded_mean_radiant_temperature: Optional[HourlyContinuousCollection] = Field(
        default=None, alias="UnshadedMeanRadiantTemperature", repr=False
    )

    @root_validator
    @classmethod
    def post_init_simulation(cls, values):  # pylint: disable=E0213
        """_"""

        _epw = EPW(values["epw_file"])
        _model = create_model(
            values["ground_material"],
            values["shade_material"],
            values["identifier"],
        )

        # run simulation and populate object with results if not already done
        if not all(
            [
                values["shaded_down_temperature"],
                values["unshaded_down_temperature"],
                values["shaded_up_temperature"],
                values["unshaded_up_temperature"],
            ]
        ):
            results = simulate_surface_temperatures(
                model=_model,
                epw_file=values["epw_file"],
                remove_dir=not bool(values["identifier"]),
            )
            for k, v in results.items():
                if isinstance(values[k], HourlyContinuousCollection):
                    continue
                values[k] = v

        # calculate other variables
        values["shaded_radiant_temperature"] = radiant_temperature(
            [
                values["shaded_down_temperature"],
                values["shaded_up_temperature"],
            ],
        )
        values["unshaded_radiant_temperature"] = radiant_temperature(
            [
                values["unshaded_down_temperature"],
                values["unshaded_up_temperature"],
            ],
        )

        # calculate MRT
        params = SolarCalParameter()
        shaded_cal = OutdoorSolarCal(
            location=_epw.location,
            direct_normal_solar=_epw.direct_normal_radiation,
            diffuse_horizontal_solar=_epw.diffuse_horizontal_radiation,
            horizontal_infrared=_epw.horizontal_infrared_radiation_intensity,
            surface_temperatures=values["shaded_radiant_temperature"],
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
            surface_temperatures=values["unshaded_down_temperature"],
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
                values[
                    f"{shadedness}_{var.replace('mrt', 'mean_radiant_temperature')}"
                ] = getattr(cal, var)

        return values

    class Config:
        """_"""

        arbitrary_types_allowed = True
        allow_population_by_field_name = True
        json_encoders = {
            EnergyMaterial: lambda v: v.to_dict(),
            EnergyMaterialVegetation: lambda v: v.to_dict(),
            HourlyContinuousCollection: lambda v: v.to_dict(),
        }

    @validator("epw_file")
    @classmethod
    def validate_epw_file(cls, value: Path):  # pylint: disable=E0213
        """_"""
        if not value.suffix == ".epw":
            raise ValueError(f"File {value} is not an .epw file.")
        if not value.exists():
            raise ValueError(f"File {value} does not exist.")
        return value.resolve()

    @validator(
        "ground_material",
        "shade_material",
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
        pre=True,
    )
    @classmethod
    def convert_dict_to_collection(cls, value: dict) -> object:  # pylint: disable=E0213
        """_"""
        if not isinstance(value, dict):
            return value
        if "type" not in value:
            return value
        if value["type"] == "HourlyContinuous":
            return HourlyContinuousCollection.from_dict(value)
        if value["type"] == "EnergyMaterial":
            return EnergyMaterial.from_dict(value)
        if value["type"] == "EnergyMaterialVegetation":
            return EnergyMaterialVegetation.from_dict(value)
        return value

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
            self.identifier,
        )

    @decorator_factory(disable=False)
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

    def description(self) -> str:
        """_"""
        return (
            f"{self.epw_file.name} - {self.ground_material.identifier} "
            f"(ground material) - {self.shade_material.identifier} "
            "(shade material)"
        )
