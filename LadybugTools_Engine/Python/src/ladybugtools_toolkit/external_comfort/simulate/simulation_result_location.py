from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from honeybee_energy.material._base import _EnergyMaterialOpaqueBase
from ladybug.epw import EPW, HourlyContinuousCollection
import ladybug.datatype
from ladybug.header import Header
from ladybug.analysisperiod import AnalysisPeriod
from ladybug_comfort.collection.solarcal import HorizontalSolarCal
from ladybug_comfort.parameter.solarcal import SolarCalParameter
from ladybugtools_toolkit.external_comfort.model.create_model import create_model
from ladybugtools_toolkit.external_comfort.simulate.longwave_mean_radiant_temperature import (
    longwave_mean_radiant_temperature,
)
from ladybugtools_toolkit.external_comfort.simulate.solar_radiation import (
    solar_radiation,
)
from ladybugtools_toolkit.external_comfort.simulate.surface_temperature import (
    surface_temperature,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.ladybug_extension.epw.filename import filename
from ladybugtools_toolkit.ladybug_extension.epw.to_dataframe import to_dataframe


class SimulationResultLocation:
    """An object containing all the results of a mean radiant temperature simulation.

    Args:
        epw (EPW): An EPW object.
        ground_material (_EnergyMaterialOpaqueBase): A surface material for the ground zones topmost
            face.
        shade_material (_EnergyMaterialOpaqueBase): A surface material for the shade zones faces.
        identifier (str, optional): A unique identifier for the model. Defaults to None which will
            generate a unique identifier. This is useful for testing purposes!

    Returns:
        SimulationResult: An object containing all the results of a mean radiant temperature
            simulation.
    """

    def __init__(
        self,
        epw: EPW,
        ground_material: _EnergyMaterialOpaqueBase,
        shade_material: _EnergyMaterialOpaqueBase,
        sun_hours: HourlyContinuousCollection,
        sky_view_factor: float = 1,
        identifier: str = None,
    ) -> SimulationResult:
        self.epw = epw
        self.ground_material = ground_material
        self.shade_material = shade_material

        # create simulation model and obtain identifier property
        self.model = create_model(ground_material, shade_material, identifier)
        self.identifier = self.model.identifier

        # run surface temperature and radiation simulations (or load results if they already exist)
        solar_radiation_results = solar_radiation(self.model, epw)
        surface_temperature_results = surface_temperature(self.model, epw)

        # populate attributes from simulation results
        self.unshaded_total_radiation = None
        self.shaded_total_radiation = None
        self.unshaded_direct_radiation = None
        self.unshaded_diffuse_radiation = None
        self.shaded_direct_radiation = None
        self.shaded_diffuse_radiation = None
        self.shaded_below_temperature = None
        self.unshaded_below_temperature = None
        self.shaded_above_temperature = None
        self.unshaded_above_temperature = None
        for k, v in solar_radiation_results.items():
            setattr(self, k, v)
        for k, v in surface_temperature_results.items():
            setattr(self, k, v)

        # calculate LW MRT for shaded/unshaded
        sky_ratio = sky_view_factor / 2
        
        self.shaded_longwave_mean_radiant_temperature = (
            longwave_mean_radiant_temperature(
                [
                    self.shaded_below_temperature,
                    self.shaded_above_temperature,
                ],
                [1 - sky_ratio, sky_ratio],
            )
        )
        self.unshaded_longwave_mean_radiant_temperature = (
            longwave_mean_radiant_temperature(
                [
                    self.unshaded_below_temperature,
                    self.unshaded_above_temperature,
                ],
                [1 - sky_ratio, sky_ratio],
            )
        )

        # calculate LW MRT based on sun hours
        data_temp = []
        for i, fract in enumerate(sun_hours):
            data_temp.append(self.unshaded_longwave_mean_radiant_temperature.values[i] * fract
            + self.shaded_longwave_mean_radiant_temperature.values[i] * ( 1 - fract)) 
        self.location_longwave_mean_radiant_temperature = HourlyContinuousCollection(header=self.shaded_longwave_mean_radiant_temperature.header, values=data_temp)
        
        # consturct fract_body_exp
        fract_body_exp_header = Header(ladybug.datatype.TYPESDICT["Fraction"](), "fraction", AnalysisPeriod())
        fract_body_exp = HourlyContinuousCollection(header=fract_body_exp_header, values=sun_hours.values)

        # calculate MRT for shaded/unshaded
        solar_body_par = SolarCalParameter()
        shaded_fract_body_exp = 0
        unshaded_fract_body_exp = fract_body_exp
        ground_reflectivity = 0.2

        self.shaded_mean_radiant_temperature = HorizontalSolarCal(
            epw.location,
            self.shaded_direct_radiation,
            self.shaded_diffuse_radiation,
            self.shaded_longwave_mean_radiant_temperature,
            shaded_fract_body_exp,
            ground_reflectivity,
            solar_body_par,
        ).mean_radiant_temperature

        self.unshaded_mean_radiant_temperature = HorizontalSolarCal(
            epw.location,
            self.unshaded_direct_radiation,
            self.unshaded_diffuse_radiation,
            self.location_longwave_mean_radiant_temperature,
            unshaded_fract_body_exp,
            ground_reflectivity,
            solar_body_par,
        ).mean_radiant_temperature

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.identifier})"

    def to_dict(self) -> Dict[str, HourlyContinuousCollection]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        variables = [
            "epw",
            "ground_material",
            "shade_material",
            "model",
            "shaded_above_temperature",
            "shaded_below_temperature",
            "shaded_diffuse_radiation",
            "shaded_direct_radiation",
            "shaded_longwave_mean_radiant_temperature",
            "shaded_mean_radiant_temperature",
            "shaded_total_radiation",
            "unshaded_above_temperature",
            "unshaded_below_temperature",
            "unshaded_diffuse_radiation",
            "unshaded_direct_radiation",
            "unshaded_longwave_mean_radiant_temperature",
            "unshaded_mean_radiant_temperature",
            "unshaded_total_radiation",
        ]

        return {var: getattr(self, var) for var in variables}

    def to_dataframe(self, include_epw: bool = False) -> pd.DataFrame:
        """Create a Pandas DataFrame from this object.

        Args:
            include_epw (bool, optional): Set to True to include the dataframe for the EPW file
                also.

        Returns:
            pd.DataFrame: A Pandas DataFrame with this objects properties.
        """

        # get object variables
        variables = [
            "shaded_above_temperature",
            "shaded_below_temperature",
            "shaded_diffuse_radiation",
            "shaded_direct_radiation",
            "shaded_longwave_mean_radiant_temperature",
            "shaded_mean_radiant_temperature",
            "shaded_total_radiation",
            "unshaded_above_temperature",
            "unshaded_below_temperature",
            "unshaded_diffuse_radiation",
            "unshaded_direct_radiation",
            "unshaded_longwave_mean_radiant_temperature",
            "unshaded_mean_radiant_temperature",
            "unshaded_total_radiation",
        ]
        obj_series = []
        for var in variables:
            _ = to_series(getattr(self, var))
            obj_series.append(
                _.rename(
                    (
                        f"{filename(self.epw)}",
                        f"{var} - {self.ground_material.display_name} ground, {self.shade_material.display_name} shade",
                    )
                )
            )
        obj_df = pd.concat(obj_series, axis=1)

        if not include_epw:
            return obj_df

        # get epw variables
        epw_df = to_dataframe(self.epw)

        # combine and return
        return pd.concat([epw_df, obj_df], axis=1)
