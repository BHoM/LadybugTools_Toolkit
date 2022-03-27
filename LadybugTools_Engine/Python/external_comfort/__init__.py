from __future__ import annotations

import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from honeybee.model import Model
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import Temperature
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug_comfort.collection.solarcal import HorizontalSolarCal
from ladybug_comfort.parameter.solarcal import SolarCalParameter
from ladybug_extension.datacollection.from_series import from_series
from ladybug_extension.datacollection.to_series import to_series

from external_comfort.model import create_model
from external_comfort.simulate import energyplus, radiance


@dataclass(frozen=True)
class ExternalComfort:
    epw: EPW = field(init=True, repr=True)
    ground_material: _EnergyMaterialOpaqueBase = field(init=True, repr=True)
    shade_material: _EnergyMaterialOpaqueBase = field(init=True, repr=True)
    model: Model = field(init=False, repr=False)

    def __post_init__(self) -> ExternalComfort:
        object.__setattr__(
            self, "model", create_model(self.ground_material, self.shade_material)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(epw={self.epw}, ground_material={self.ground_material.identifier}, shade_material={self.shade_material.identifier}, model={self.model.identifier})"
        )


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
            self.radiant_temperature_from_collections(
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
            self.radiant_temperature_from_collections(
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
            self.mean_radiant_temperature(
                self.external_comfort.epw,
                self.shaded_longwave_mean_radiant_temperature,
                self.shaded_direct_radiation,
                self.shaded_diffuse_radiation,
            ),
        )

        object.__setattr__(
            self,
            "unshaded_mean_radiant_temperature",
            self.mean_radiant_temperature(
                self.external_comfort.epw,
                self.unshaded_longwave_mean_radiant_temperature,
                self.unshaded_direct_radiation,
                self.unshaded_diffuse_radiation,
            ),
        )

    @staticmethod
    def mean_radiant_temperature(
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
    def radiant_temperature_from_collections(
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
    def mean_radiant_temperature_from_surfaces(
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
