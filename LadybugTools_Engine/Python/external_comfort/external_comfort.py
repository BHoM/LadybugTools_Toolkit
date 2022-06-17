from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from honeybee.model import Model
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import Temperature
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug_comfort.collection.solarcal import HorizontalSolarCal
from ladybug_comfort.parameter.solarcal import SolarCalParameter
from ladybug_extension.datacollection import from_series, to_series

from external_comfort.encoder import Encoder
from external_comfort.model import create_model
from external_comfort.simulate import energyplus, radiance, SIMULATION_DIRECTORY


class ExternalComfortEncoder(Encoder):
    """A JSON encoder for the ExternalComfort and ExternalComfortResult classes."""

    def default(self, obj):
        if isinstance(obj, ExternalComfort):
            return obj.to_dict()
        if isinstance(obj, ExternalComfortResult):
            return obj.to_dict()
        return super(ExternalComfortEncoder, self).default(obj)


@dataclass(frozen=True)
class ExternalComfort:
    epw: EPW = field(init=True, repr=True)
    ground_material: _EnergyMaterialOpaqueBase = field(init=True, repr=True)
    shade_material: _EnergyMaterialOpaqueBase = field(init=True, repr=True)
    identifier: str = field(init=True, repr=False, default=None)
    model: Model = field(init=False, repr=False)

    def __post_init__(self) -> ExternalComfort:
        object.__setattr__(
            self,
            "model",
            create_model(self.ground_material, self.shade_material, self.identifier),
        )
        
        # Save EPW into working directory folder for posterity
        self.epw.save(SIMULATION_DIRECTORY / self.model.identifier / Path(self.epw.file_path).name)

        # Rename object based on model identifier
        object.__setattr__(self, "identifier", self.model.identifier)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.identifier}, {self.epw}, {self.ground_material.identifier}, {self.shade_material.identifier})"

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        d = {
            "epw": self.epw,
            "ground_material": self.ground_material,
            "shade_material": self.shade_material,
            "model": self.model,
        }
        return d

    def to_json(self, file_path: str) -> Path:
        """Return this object as a json file

        Returns:
            Path: The json file path.
        """

        file_path: Path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)

        with open(file_path, "w") as fp:
            json.dump(self.to_dict(), fp, cls=ExternalComfortEncoder, indent=4)

        return file_path


@dataclass(frozen=True)
class ExternalComfortResult:
    external_comfort: ExternalComfort = field(init=True, repr=True, compare=True)

    shaded_below_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    shaded_above_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    shaded_direct_radiation: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    shaded_diffuse_radiation: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    shaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    shaded_mean_radiant_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )

    unshaded_below_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    unshaded_above_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    unshaded_direct_radiation: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    unshaded_diffuse_radiation: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    unshaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    unshaded_mean_radiant_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> ExternalComfortResult:
        """Calculate the mean radiant tempertaure, and constituent parts of this value from the External Comfort configuration."""

        print(f"- Running external comfort calculation for {self.external_comfort.identifier}")

        # Run EnergyPlus and Radiance simulations
        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            for f in [radiance, energyplus]:
                results.append(
                    executor.submit(
                        f, self.external_comfort.model, self.external_comfort.epw
                    )
                )

        # Populate simulation results
        for x in results:
            for k, v in x.result().items():
                object.__setattr__(self, k, v)

        # Populate calculated results
        object.__setattr__(
            self,
            "shaded_longwave_mean_radiant_temperature",
            self._radiant_temperature_from_collections(
                [
                    self.shaded_below_temperature,
                    self.shaded_above_temperature,
                ],
                [0.5, 0.5],
            ),
        )

        object.__setattr__(
            self,
            "unshaded_longwave_mean_radiant_temperature",
            self._radiant_temperature_from_collections(
                [
                    self.unshaded_below_temperature,
                    self.unshaded_above_temperature,
                ],
                [0.5, 0.5],
            ),
        )

        object.__setattr__(
            self,
            "shaded_mean_radiant_temperature",
            self._mean_radiant_temperature(
                self.external_comfort.epw,
                self.shaded_longwave_mean_radiant_temperature,
                self.shaded_direct_radiation,
                self.shaded_diffuse_radiation,
            ),
        )

        object.__setattr__(
            self,
            "unshaded_mean_radiant_temperature",
            self._mean_radiant_temperature(
                self.external_comfort.epw,
                self.unshaded_longwave_mean_radiant_temperature,
                self.unshaded_direct_radiation,
                self.unshaded_diffuse_radiation,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        attributes = [
            "external_comfort",
            "shaded_below_temperature",
            "shaded_above_temperature",
            "shaded_direct_radiation",
            "shaded_diffuse_radiation",
            "shaded_longwave_mean_radiant_temperature",
            "shaded_mean_radiant_temperature",
            "unshaded_below_temperature",
            "unshaded_above_temperature",
            "unshaded_direct_radiation",
            "unshaded_diffuse_radiation",
            "unshaded_longwave_mean_radiant_temperature",
            "unshaded_mean_radiant_temperature",
        ]
        return {attribute: getattr(self, attribute) for attribute in attributes}

    def to_json(self, file_path: str) -> Path:
        """Write the content of this object to a JSON file

        Returns:
            Path: The path to the newly created JSON file.
        """

        file_path: Path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)

        with open(file_path, "w") as fp:
            json.dump(self.to_dict(), fp, cls=ExternalComfortEncoder, indent=4)

        return file_path

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.external_comfort.identifier})"

    @staticmethod
    def _mean_radiant_temperature(
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

    @staticmethod
    def _radiant_temperature_from_collections(
        collections: List[HourlyContinuousCollection], view_factors: List[float]
    ) -> HourlyContinuousCollection:
        """Calculate the radiant temperature from a list of hourly continuous collections and view factors to each of those collections.

        Args:
            collections (List[HourlyContinuousCollection]): A list of hourly continuous collections.
            view_factors (List[float]): A list of view factors to each of the collections.

        Returns:
            HourlyContinuousCollection: An HourlyContinuousCollection of the effective radiant temperature.
        """

        if len(collections) != len(view_factors):
            raise ValueError(
                "The number of collections and view factors must be the same."
            )
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
        mrt_series.name = "Temperature (C)"
        return from_series(mrt_series)

    @staticmethod
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

        if len(surface_temperatures) != len(view_factors):
            raise ValueError(
                "The number of surface temperatures and view factors must be the same."
            )

        resultant_temperature = 0
        for i, temp in enumerate(surface_temperatures):
            temperature_kelvin = temp + 273.15
            resultant_temperature = (
                resultant_temperature + np.pow(temperature_kelvin, 4) * view_factors[i]
            )
        mean_radiant_temperature_kelvin = np.pow(resultant_temperature, 0.25)
        mean_radiant_temperature = mean_radiant_temperature_kelvin - 273.15
        return mean_radiant_temperature

    def to_dataframe(self) -> pd.DataFrame:
        """Create a dataframe from the simulation results.

        Returns:
            pd.DataFrame: A dataframe containing the simulation results.
        """

        attributes = [
            "shaded_below_temperature",
            "shaded_above_temperature",
            "shaded_direct_radiation",
            "shaded_diffuse_radiation",
            "shaded_longwave_mean_radiant_temperature",
            "shaded_mean_radiant_temperature",
            "unshaded_below_temperature",
            "unshaded_above_temperature",
            "unshaded_direct_radiation",
            "unshaded_diffuse_radiation",
            "unshaded_longwave_mean_radiant_temperature",
            "unshaded_mean_radiant_temperature",
        ]
        series: List[pd.Series] = []
        for attribute in attributes:
            series.append(to_series(getattr(self, attribute)))
        return pd.concat(
            series,
            axis=1,
            keys=[f"{self.__class__.__name__} - {i}" for i in attributes],
        )
