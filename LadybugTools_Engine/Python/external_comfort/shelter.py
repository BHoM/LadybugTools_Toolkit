from __future__ import annotations

import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from typing import List

import numpy as np
import pandas as pd
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug_extension.datacollection import from_series, to_series
from ladybug_extension.sun import sun_positions_as_array
from shapely.geometry.polygon import Point, Polygon


def _cardinality(angle_from_north: float, directions: int = 16):
    """
    Returns the cardinal orientation of a given angle, where that angle is related to north at 0 degrees.

    Args:
        angle_from_north (float): The angle to north in degrees (+Ve is interpreted as clockwise from north at 0.0 degrees).
        directions (int): The number of cardinal directions into which angles shall be binned (This value should be one of 4, 8, 16 or 32, and is centred about "north").

    Returns:
        int: The cardinal direction the angle represents.
    """

    if angle_from_north > 360 or angle_from_north < 0:
        raise ValueError(
            "The angle entered is beyond the normally expected range for an orientation in degrees."
        )

    cardinal_directions = {
        4: ["N", "E", "S", "W"],
        8: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        16: [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ],
        32: [
            "N",
            "NbE",
            "NNE",
            "NEbN",
            "NE",
            "NEbE",
            "ENE",
            "EbN",
            "E",
            "EbS",
            "ESE",
            "SEbE",
            "SE",
            "SEbS",
            "SSE",
            "SbE",
            "S",
            "SbW",
            "SSW",
            "SWbS",
            "SW",
            "SWbW",
            "WSW",
            "WbS",
            "W",
            "WbN",
            "WNW",
            "NWbW",
            "NW",
            "NWbN",
            "NNW",
            "NbW",
        ],
    }

    if directions not in cardinal_directions:
        raise ValueError(
            f"The number of directions must be one of {list(cardinal_directions.keys())}."
        )

    val = int((angle_from_north / (360 / directions)) + 0.5)

    arr = cardinal_directions[directions]

    return arr[(val % directions)]


class Shelter:
    def __init__(
        self,
        altitude_range: List[float],
        azimuth_range: List[float],
        porosity: float = 0,
    ) -> Shelter:
        """A shelter describing the solar and wind protection affored by a specific external comfort typology.

        Args:
            altitude_range (List[float]): The range of altitudes in degrees above the horizon at which the shelter is effective.
            azimuth_range (List[float]): The range of azimuths in degrees from north at which the shelter is effective.
            porosity (float): The proportion of the shelter area that is exposed to the sun.

        Returns:
            Shelter: A shelter object.
        """

        if not (0 <= porosity <= 1):
            raise ValueError(f"Porosity must be between 0 and 1")

        if len(altitude_range) != 2:
            raise ValueError("Altitude range must be a list of two floats.")

        if len(azimuth_range) != 2:
            raise ValueError("Azimuth range must be a list of two floats.")

        if (min(altitude_range) < 0) or (max(altitude_range) > 90):
            raise ValueError(f"Altitude range must fall between 0 and 90")

        if (min(azimuth_range) < 0) or (max(azimuth_range) > 360):
            raise ValueError(f"Azimuth range must fall between 0 and 360")

        self.porosity = porosity
        self.altitude_range = altitude_range
        self.azimuth_range = azimuth_range

        self.polygons = self._create_polygons()

    def __str__(self) -> str:
        return f"{__class__.__name__}"

    def __repr__(self) -> str:
        return str(self)

    @property
    def description(self) -> str:
        """A description of the shelter."""
        min_azimuth = self.azimuth_range[0]
        max_azimuth = self.azimuth_range[1]

        min_altitude = self.altitude_range[0]
        max_altitude = self.altitude_range[1]

        if min_azimuth == max_azimuth:
            return "Fully exposed"
        if min_altitude == max_altitude:
            return "Fully exposed"
        if (
            (min_azimuth == 0)
            and (max_azimuth == 360)
            and (min_altitude == 0)
            and (max_altitude == 90)
        ):
            if self.porosity == 0:
                return "Fully enclosed"
            else:
                return f"Fully enclosed by shelter with {self.porosity:0.0%} porosity"

        if self.porosity == 0:
            return f"Sheltered between {min_azimuth}° and {max_azimuth}° degrees from North, between altitudes of {min_altitude}° and {max_altitude}°"
        else:
            return f"Sheltered between {min_azimuth}° and {max_azimuth}° degrees from North, between altitudes of {min_altitude}° and {max_altitude}° with {self.porosity:0.0%} porous shelter"

    def _shelter_vertices(self) -> List[List[List[float]]]:
        """Returns the vertices of the shelter polygons."""

        if self.altitude_range[0] == self.altitude_range[1]:
            return []

        if self.azimuth_range[0] == self.azimuth_range[1]:
            return []

        if self.azimuth_range[0] > self.azimuth_range[1]:
            vertices = []
            for local_azimuth_range in [
                [self.azimuth_range[0], 360],
                [0, self.azimuth_range[1]],
            ]:
                vertices.append(
                    [
                        [local_azimuth_range[0], self.altitude_range[0]],
                        [local_azimuth_range[0], self.altitude_range[1]],
                        [local_azimuth_range[1], self.altitude_range[1]],
                        [local_azimuth_range[1], self.altitude_range[0]],
                    ]
                )
            return vertices
        else:
            return [
                [
                    [self.azimuth_range[0], self.altitude_range[0]],
                    [self.azimuth_range[0], self.altitude_range[1]],
                    [self.azimuth_range[1], self.altitude_range[1]],
                    [self.azimuth_range[1], self.altitude_range[0]],
                ]
            ]

    def _create_polygons(self) -> List[Polygon]:
        """Create a list of Polygons defining the shelter."""
        polygons = []
        for shelter_vertices in self._shelter_vertices():
            polygons.append(Polygon(shelter_vertices))
        return polygons

    def _shelter_area(self) -> float:
        """Returns the total sky area occluded by the shelter."""
        return sum([polygon.area for polygon in self._create_polygons()])

    def _sky_area(self) -> float:
        """Returns the total sky area."""
        return 360 * 90

    @property
    def _sky_view_factor(self) -> float:
        total_sky_area = self._sky_area()
        shelter_area = self._shelter_area()
        sheltered_sky_proportion = shelter_area / total_sky_area
        visible_sky_proportion = 1 - sheltered_sky_proportion
        return (sheltered_sky_proportion * self.porosity) + (visible_sky_proportion * 1)

    def _is_sun_sheltered(self, epw: EPW) -> List[bool]:
        """For each hour of the year, return True if the sun is sheltered by the shelter."""
        if self.porosity == 1:
            return [False] * 8760

        if not self.polygons:
            return [False] * 8760

        sun_locations = sun_positions_as_array(epw)
        masks = []
        for polygon in self.polygons:
            local_mask = []
            for sun_location in sun_locations:
                local_mask.append(
                    polygon.contains(Point(sun_location[1], sun_location[0]))
                )
            masks.append(local_mask)
        return np.any([masks], axis=1)[0]

    # TODO - add method to account for solar disc in radiation protection (if sun within X degrees of shelter, then apply slight reduction)

    def _unshaded_shaded_interpolant(self, epw: EPW) -> List[float]:
        """For an annual hourly timestep, return a list of floats between 0 and 1, describing which case to use for mean-radiant-temperature calculation.

        Example:
        0 == shaded
        1 == unshaded
        Assume that you have 2 datasets, one describing unshaded MRT and another describing shaded MRT. If these represent the minimum and maximum MRT values for the year,
        then the interpolant will be a list of 0s and 1s, where 0 represents the shaded case and 1 represents the unshaded case. When the sun is fully blocked, then the interpolant would equal 0,
        wheras if the sun is fully visible the interpolant equals 1.  Overnight, we only care about the sky view factor - so use that value when the sun is down as the value between which we
        interpolate (1 is equal to a fully visible night sky).
        """
        sky_view_factor = self._sky_view_factor
        if sky_view_factor == 0:
            return [0] * 8760
        elif sky_view_factor == 1:
            return [1] * 8760

        sol_alt, _ = sun_positions_as_array(epw).T
        sun_sheltered = self._is_sun_sheltered(epw)

        interpolants = []
        for alt, shelter in list(zip(*[sol_alt, sun_sheltered])):
            if alt <= 0:
                interpolants.append(sky_view_factor)
            elif shelter:
                interpolants.append(1 - (1 - self.porosity))
            else:
                interpolants.append(1)

        interpolants = (
            pd.Series(interpolants, index=to_series(epw.wind_direction).index)
            .ewm(halflife=1)
            .mean()
            .values
        )

        return interpolants

    @property
    def _is_sheltered(self) -> bool:
        """Return True if the shelter provides shelter from either wind or sun, and False if not."""
        min_altitude = min(self.altitude_range)
        max_altitude = max(self.altitude_range)
        min_azimuth = min(self.azimuth_range)
        max_azimuth = max(self.azimuth_range)
        if min_altitude == max_altitude:
            return False
        if min_azimuth == max_azimuth:
            return False
        if self.porosity < 1:
            if self._sky_view_factor != 1:
                return True
        return False

    def wind_speed_factor(self, altitude_threshold: float = 45) -> List[float]:
        """Helper method to obtain a factor to apply to wind speed, based on the proportion of vertical shelter below the altitude threshold.

        Args:
            altitude_threshold (float, optional): The altitude below which wind speed will be reduced. Defaults to 45.

        Returns:
            float: The wind speed reduction factor to be applied to the wind speed. If 1 then no wind speed reduction is applied. If 0 then wind is reduced to 0
        """

        if not 0 < altitude_threshold <= 90:
            raise ValueError("Altitude threshold must be between 0 and 90")

        if self.porosity == 1:
            return 1.0

        min_altitude = min(self.altitude_range)
        max_altitude = max(self.altitude_range)

        if min_altitude == max_altitude:
            return 1
        elif (min_altitude >= altitude_threshold) and (
            max_altitude > altitude_threshold
        ):
            return 1

        if (min_altitude < altitude_threshold) and (max_altitude <= altitude_threshold):
            area_shelter_below_threshold = max_altitude - min_altitude
        elif (min_altitude < altitude_threshold) and (
            max_altitude > altitude_threshold
        ):
            area_shelter_below_threshold = altitude_threshold - min_altitude
        else:
            area_shelter_below_threshold = 0

        shelter_proportion = area_shelter_below_threshold / altitude_threshold
        effective_shelter = (shelter_proportion * self.porosity) + (
            (1 - shelter_proportion)
        )

        return effective_shelter

    def effective_wind_speed(self, epw: EPW) -> HourlyContinuousCollection:
        """Adjust wind_speed based on shelter positioning and porosity."""

        min_azimuth = min(self.azimuth_range)
        max_azimuth = max(self.azimuth_range)

        wd = to_series(epw.wind_direction)
        ws = to_series(epw.wind_speed)

        if min_azimuth > max_azimuth:
            mask = (wd > min_azimuth) | (wd < max_azimuth)
        else:
            mask = (wd > min_azimuth) & (wd < max_azimuth)

        adjustment_factor = self.wind_speed_factor(altitude_threshold=45)
        ws_adjusted = ws.where(~mask, ws * adjustment_factor)
        return from_series(ws_adjusted)


if __name__ == "__main__":
    pass
