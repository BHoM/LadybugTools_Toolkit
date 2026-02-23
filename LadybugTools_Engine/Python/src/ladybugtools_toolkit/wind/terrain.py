from enum import Enum, auto
from typing import Optional, Union

import numpy as np
import pandas as pd
from ladybug.datacollection import (
    HourlyContinuousCollection,
    HourlyDiscontinuousCollection,
)


class WindTerrainType(Enum):
    """A class to represent the terrain type for wind data."""

    CITY = auto()
    SUBURBAN = auto()
    COUNTRY = auto()
    WATER = auto()

    @property
    def roughness_length(self) -> float:
        d = {
            WindTerrainType.CITY.name: 1.0,
            WindTerrainType.SUBURBAN.name: 0.5,
            WindTerrainType.COUNTRY.name: 0.1,
            WindTerrainType.WATER.name: 0.03,
        }
        return d[self.name]

    @property
    def boundary_layer_height(self) -> float:
        d = {
            WindTerrainType.CITY.name: 460,
            WindTerrainType.SUBURBAN.name: 370,
            WindTerrainType.COUNTRY.name: 270,
            WindTerrainType.WATER.name: 210,
        }
        return d[self.name]

    @property
    def power_law_exponent(self) -> float:
        d = {
            WindTerrainType.CITY.name: 0.33,
            WindTerrainType.SUBURBAN.name: 0.22,
            WindTerrainType.COUNTRY.name: 0.14,
            WindTerrainType.WATER.name: 0.1,
        }
        return d[self.name]

    @classmethod
    def from_roughness_length(cls, roughness_length: float) -> "WindTerrainType":
        """Get the terrain type from a roughness length."""
        return abs(
            pd.Series({tt: tt.roughness_length for tt in cls}) - roughness_length
        ).idxmin()

    def wind_speed_at_height(
        self,
        reference_value: float,
        reference_height: float,
        target_height: float,
        log_law: bool = False,
        target_terrain_type: Optional["WindTerrainType"] = None,
    ) -> float:
        if target_terrain_type is None:
            target_terrain_type = self

        if log_law:
            if target_height <= target_terrain_type.roughness_length:
                return 0
            ref_h_ratio = reference_height / self.roughness_length
            ref_log_denominator = np.log(ref_h_ratio)
            ref_log_num = np.log(target_height / target_terrain_type.roughness_length)
            return float(reference_value * (ref_log_num / ref_log_denominator))

        ref_h_ratio = self.boundary_layer_height / reference_height
        ref_power_denominator = ref_h_ratio**self.power_law_exponent
        target_h_ratio = (
            target_height / target_terrain_type.boundary_layer_height
        ) ** target_terrain_type.power_law_exponent
        return float(target_h_ratio * (reference_value * ref_power_denominator))

    def translate_wind_speed_collection(
        self,
        wind_speed: Union[HourlyContinuousCollection, HourlyDiscontinuousCollection],
        reference_height: float,
        target_height: float,
        log_law: bool = False,
        target_terrain_type: Optional["WindTerrainType"] = None,
    ) -> HourlyContinuousCollection:
        if not isinstance(
            wind_speed, (HourlyContinuousCollection, HourlyDiscontinuousCollection)
        ):
            raise TypeError(
                "The wind speed must be an instance of HourlyContinuousCollection or HourlyDiscontinuousCollection."
            )
        if not str(wind_speed.header.data_type) == "Wind Speed":
            raise TypeError("The wind speed collection must be of type 'Wind Speed'.")
        if not wind_speed.header.unit == "m/s":
            raise TypeError("The wind speed collection must be in 'm/s'.")

        translated = [
            self.wind_speed_at_height(
                reference_value=ws,
                reference_height=reference_height,
                target_height=target_height,
                log_law=log_law,
                target_terrain_type=target_terrain_type,
            )
            for ws in wind_speed.values
        ]

        hd_collection = HourlyDiscontinuousCollection(
            header=wind_speed.header,
            values=translated,
            datetimes=wind_speed.datetimes,
        )
        hd_collection.header.metadata["terrain_type"] = self.name
        hd_collection.header.metadata["height_above_ground"] = target_height

        if isinstance(wind_speed, HourlyContinuousCollection):
            return HourlyContinuousCollection(
                header=hd_collection.header,
                values=hd_collection.values,
            )

        return hd_collection


def target_wind_speed_collection(
    reference_wind_speed: HourlyContinuousCollection,
    target_average_wind_speed: float,
    target_height: float = 10,
    reference_height: float = 10,
    reference_terrain_type: WindTerrainType = WindTerrainType.COUNTRY,
    target_terrain_type: WindTerrainType = WindTerrainType.COUNTRY,
) -> HourlyContinuousCollection:
    """Create an annual hourly collection of wind-speeds whose average equals the target value,
        translated to 10m height, using the source EPW to provide a wind-speed profile.

    Args:
        reference_wind_speed (HourlyContinuousCollection):
            A ladybug annual hourly data wind speed collection.
        target_average_wind_speed (float):
            The average wind speed for the resultant collection.
        target_height (float):
            The height at which the wind speed is translated to.
            Default is 10m.
        reference_height (float):
            The height at which the wind speed is translated from.
            Default is 10m.
        reference_terrain_type (TerrainType):
            The terrain type of the reference wind speed collection.
            Default is TerrainType.COUNTRY.
        target_terrain_type (TerrainType):
            The terrain type of the target wind speed collection.
            Default is TerrainType.COUNTRY.

    Returns:
        HourlyContinuousCollection:
            A ladybug wind speed data collection.

    """
    if target_average_wind_speed < 0:
        raise ValueError("The target average wind speed must be greater than 0.")

    # translate original wind to the new height
    translated = reference_terrain_type.translate_wind_speed_collection(
        wind_speed=reference_wind_speed,
        reference_height=reference_height,
        target_height=target_height,
        log_law=False,
        target_terrain_type=target_terrain_type,
    )
    # get translated avg speed
    adjustment_factor = target_average_wind_speed / translated.average

    return translated * adjustment_factor
