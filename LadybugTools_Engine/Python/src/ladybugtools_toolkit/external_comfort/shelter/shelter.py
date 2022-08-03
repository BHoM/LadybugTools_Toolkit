from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug.sunpath import Sun
from ladybugtools_toolkit.helpers.cardinality import cardinality
from ladybugtools_toolkit.ladybug_extension.datacollection.from_series import (
    from_series,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from shapely.geometry import Polygon


class Shelter:
    """An object representing a piece of geometry blocking exposure to wind, sky and sun.

    Args:
        porosity (float, optional): The transmissivity of the shelter. Defaults to 0.
        altitude_range (Tuple[float], optional): The altitude range covered by this shelter.
            Defaults to (0, 0).
        azimuth_range (Tuple[float], optional): The azimuth range (from 0 at North, clockwise)
            covered by this shelter. Defaults to (0, 0).

    Returns:
        Shelter: An object representing a piece of geometry blocking exposure to wind, sky and
            sun.
    """

    def __init__(
        self,
        porosity: float = 0,
        altitude_range: Tuple[float] = (0, 0),
        azimuth_range: Tuple[float] = (0, 0),
    ) -> Shelter:

        self.porosity = porosity
        self.altitude_range = altitude_range
        self.azimuth_range = azimuth_range

        if not 0 <= self.porosity <= 1:
            raise ValueError("porosity must be between 0 and 1")

        if any(
            [
                min(altitude_range) < 0,
                max(altitude_range) > 90,
                len(altitude_range) != 2,
            ]
        ):
            raise ValueError("altitude_range must be two floats between 0 and 90")

        if any(
            [
                min(azimuth_range) < 0,
                max(azimuth_range) > 360,
                len(azimuth_range) != 2,
            ]
        ):
            raise ValueError("azimuth_range must be two floats between 0 and 360.")

        # hidden properties useful to downstream operations
        self._start_altitude, self._end_altitude = sorted(self.altitude_range)
        self._start_azimuth, self._end_azimuth = self.azimuth_range
        self._start_altitude_radians, self._end_altitude_radians = [
            np.radians(i) for i in sorted(self.altitude_range)
        ]
        self._start_azimuth_radians, self._end_azimuth_radians = [
            np.radians(i) for i in self.azimuth_range
        ]
        self._width = (
            (360 - self._start_azimuth) + self._end_azimuth
            if self._start_azimuth > self._end_azimuth
            else self._end_azimuth - self._start_azimuth
        )
        self._width_radians = np.radians(self._width)
        self._height = self._end_altitude - self._start_altitude
        self._height_radians = self._height
        self._crosses_north = self._start_azimuth > self._end_azimuth

    def __repr__(self) -> str:
        if any([self.porosity == 1, self._height == 0, self._width == 0]):
            return "Unsheltered"

        return f"{1 - self.porosity:0.0%} sheltered between {cardinality(self._start_azimuth, 32)} and {cardinality(self._end_azimuth, 32)}, from {self._start_altitude}° to {self._end_altitude}°"

    def to_dict(self) -> Dict:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """
        return {
            "porosity": float(self.porosity),
            "altitude_range": [float(i) for i in self.altitude_range],
            "azimuth_range": [float(i) for i in self.azimuth_range],
        }

    def polygons(self) -> List[Polygon]:
        """Return a list of polygons representing the shade of this shelter.

        Returns:
            List[Polygon]: A list of polygons representing the shade of this shelter.
        """
        if self._width == 0:
            return []

        if self._height == 0:
            return []

        if self._crosses_north:
            return [
                Polygon(
                    [
                        (self._start_azimuth, self._start_altitude),
                        (self._start_azimuth, self._end_altitude),
                        (360, self._end_altitude),
                        (360, self._start_altitude),
                        (self._start_azimuth, self._start_altitude),
                    ]
                ),
                Polygon(
                    [
                        (0, self._start_altitude),
                        (0, self._end_altitude),
                        (self._end_azimuth, self._end_altitude),
                        (self._end_azimuth, self._start_altitude),
                        (0, self._start_altitude),
                    ]
                ),
            ]
        else:
            return [
                Polygon(
                    [
                        (self._start_azimuth, self._start_altitude),
                        (self._start_azimuth, self._end_altitude),
                        (self._end_azimuth, self._end_altitude),
                        (self._end_azimuth, self._start_altitude),
                    ]
                )
            ]

    def sun_blocked(self, suns: Union[List[Sun], Sun]) -> List[bool]:
        """Return a list of booleans indicating whether the sun is blocked by this shelter.

        Args:
            suns (Union[List[Sun], Sun]): Either a Sun object or a list of Sun objects.

        Returns:
            List[bool]: A list of booleans indicating whether the sun (or each of the suns) is
                blocked by this shelter.
        """
        if not isinstance(suns, list):
            suns = [suns]
        blocked = []
        for sun in suns:
            if not isinstance(sun, Sun):
                raise ValueError("Object input is not a Sun.")

            in_altitude_range = self._start_altitude < sun.altitude < self._end_altitude

            if self._crosses_north:
                in_azimuth_range = (sun.azimuth > self._start_azimuth) or (
                    sun.azimuth < self._end_azimuth
                )
            else:
                in_azimuth_range = self._start_azimuth < sun.azimuth < self._end_azimuth

            if in_altitude_range and in_azimuth_range:
                blocked.append(True)
            else:
                blocked.append(False)
        return blocked

    def sky_blocked(self) -> float:
        """Return the proportion of the sky occluded by the shelter (including porosity to increase
            sky visibility if the shelter is porous).

        Returns:
            The proportion of sky occluded by the spherical patch
        """

        if self._width_radians < 0:
            raise ValueError(
                "You cannot occlude the sky with a negatively sized patch."
            )

        if any(
            [
                self._start_altitude_radians < 0,
                self._start_altitude_radians > np.pi / 2,
                self._end_altitude_radians < 0,
                self._end_altitude_radians > np.pi / 2,
            ]
        ):
            raise ValueError("Start and end altitudes must be between 0 and pi/2.")

        area_occluded = abs(
            (np.sin(self._end_altitude_radians) - np.sin(self._start_altitude_radians))
            * self._width_radians
            / (2 * np.pi)
        )

        return area_occluded * (1 - self.porosity)

    def effective_wind_speed(self, epw: EPW) -> HourlyContinuousCollection:
        """Return the wind speed (at original height of 10m from EPW) when subjected to this
            shelter.

           The proportion of shelter occluding the altitude, and the porosity of the shelter
            determines how much of the wind to block. If not blocked by the shelter, but within 10°,
            a 10% increase is applied to estimate impact of edge acceleration.

        Args:
            epw (EPW): An EPW object.

        Returns:
            HourlyContinuousCollection: An HourlyContinuousCollection object with the effective wind
            speed impacted by this shelter.
        """

        if self.porosity == 1:
            return epw.wind_speed

        edge_acceleration_width = (
            10  # degrees either side of edge in which to increase wind speed
        )
        edge_acceleration_factor = (
            1.1  # amount by which to increase wind speed by for edge effects
        )

        shelter_height_factor = np.interp(self._height / 90, [0, 1], [1, 0.25])

        wind_direction = to_series(epw.wind_direction)
        wind_speed = to_series(epw.wind_speed)
        df = pd.concat([wind_speed, wind_direction], axis=1)
        modified_values = []
        for _, row in df.iterrows():
            if self._crosses_north:
                # wind blocked by shelter
                if (row[1] > self._start_azimuth) or (row[1] < self._end_azimuth):
                    modified_values.append(
                        row[0] * self.porosity * shelter_height_factor
                    )
                # wind not blocked by shelter, but it's within 10deg of shelter
                elif (row[1] > self._start_azimuth - edge_acceleration_width) or (
                    row[1] < self._end_azimuth + edge_acceleration_width
                ):
                    modified_values.append(
                        row[0]
                        * edge_acceleration_factor
                        * self.porosity
                        * shelter_height_factor
                    )
                # wind not blocked by shelter
                else:
                    modified_values.append(row[0])
            else:
                # wind blocked by shelter
                if (row[1] > self._start_azimuth) and (row[1] < self._end_azimuth):
                    modified_values.append(
                        row[0] * self.porosity * shelter_height_factor
                    )
                # wind not blocked by shelter, but it's within 10deg of shelter
                elif (row[1] > self._start_azimuth - edge_acceleration_width) and (
                    row[1] < self._end_azimuth + edge_acceleration_width
                ):
                    modified_values.append(
                        row[0]
                        * edge_acceleration_factor
                        * self.porosity
                        * shelter_height_factor
                    )
                else:
                    modified_values.append(row[0])
        return from_series(
            pd.Series(modified_values, index=df.index, name="Wind Speed (m/s)")
        )
