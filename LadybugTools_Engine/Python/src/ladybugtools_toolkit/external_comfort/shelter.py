from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug.sunpath import Sun
from shapely.geometry import Polygon

from ..bhomutil.bhom_object import BHoMObject, bhom_dict_to_dict
from ..ladybug_extension.datacollection import from_series, to_series
from ..ladybug_extension.epw import sun_position_list


@dataclass(init=True, repr=True, eq=True)
class Shelter(BHoMObject):
    """An object representing a piece of geometry blocking exposure to wind,
        sky and sun.

    Args:
        wind_porosity (float, optional):
            The transmissivity of the shelter to wind. Defaults to 0.
        radiation_porosity (float, optional):
            The transmissivity of the shelter to radiation (from surfaces,
            sky and sun). Defaults to 0.
        altitude_range (Tuple[float], optional):
            The altitude range covered by this shelter.
            Defaults to (0, 0).
        azimuth_range (Tuple[float], optional):
            The azimuth range (from 0 at North, clockwise)
            covered by this shelter. Defaults to (0, 0).

    Returns:
        Shelter:
            An object representing a piece of geometry blocking exposure to
            wind, sky and sun.
    """

    wind_porosity: float = field(init=True, repr=False, compare=True, default=0)
    radiation_porosity: float = field(init=True, repr=False, compare=True, default=0)
    altitude_range: List[float] = field(
        init=True, repr=False, compare=True, default_factory=lambda: [0, 0]
    )
    azimuth_range: List[float] = field(
        init=True, repr=False, compare=True, default_factory=lambda: [0, 0]
    )

    _t: str = field(
        init=False, repr=False, compare=True, default="BH.oM.LadybugTools.Shelter"
    )

    def __post_init__(self):

        if (not 0 <= self.wind_porosity <= 1) or (
            not 0 <= self.radiation_porosity <= 1
        ):
            raise ValueError("porosity must be between 0 and 1")

        if any(
            [
                min(self.altitude_range) < 0,
                max(self.altitude_range) > 90,
                len(self.altitude_range) != 2,
            ]
        ):
            raise ValueError("altitude_range must be two floats between 0 and 90")

        if any(
            [
                min(self.azimuth_range) < 0,
                max(self.azimuth_range) > 360,
                len(self.azimuth_range) != 2,
            ]
        ):
            raise ValueError("azimuth_range must be two floats between 0 and 360.")

        # wrap methods within this class
        super().__post_init__()

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> Shelter:
        """Create this object from a dictionary."""

        sanitised_dict = bhom_dict_to_dict(dictionary)
        sanitised_dict.pop("_t", None)

        return cls(
            wind_porosity=sanitised_dict["wind_porosity"],
            radiation_porosity=sanitised_dict["radiation_porosity"],
            altitude_range=sanitised_dict["altitude_range"],
            azimuth_range=sanitised_dict["azimuth_range"],
        )

    @classmethod
    def from_json(cls, json_string: str) -> Shelter:
        """Create this object from a JSON string."""

        dictionary = json.loads(json_string)

        return cls.from_dict(dictionary)

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

    def sun_blocked(self, suns: List[Sun]) -> List[bool]:
        """Return a list of booleans indicating whether the sun is blocked by
            this shelter.

        Args:
            suns (List[Sun]]):
                A list of Sun objects.

        Returns:
            List[bool]:
                A list of booleans indicating whether the sun (or each of
                the suns) is blocked by this shelter.
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

        return area_occluded * (1 - self.radiation_porosity)

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

        if self.wind_porosity == 1:
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
                        row[0] * self.wind_porosity * shelter_height_factor
                    )
                # wind not blocked by shelter, but it's within 10deg of shelter
                elif (row[1] > self._start_azimuth - edge_acceleration_width) or (
                    row[1] < self._end_azimuth + edge_acceleration_width
                ):
                    modified_values.append(
                        row[0]
                        * edge_acceleration_factor
                        * self.wind_porosity
                        * shelter_height_factor
                    )
                # wind not blocked by shelter
                else:
                    modified_values.append(row[0])
            else:
                # wind blocked by shelter
                if (row[1] > self._start_azimuth) and (row[1] < self._end_azimuth):
                    modified_values.append(
                        row[0] * self.wind_porosity * shelter_height_factor
                    )
                # wind not blocked by shelter, but it's within 10deg of shelter
                elif (row[1] > self._start_azimuth - edge_acceleration_width) and (
                    row[1] < self._end_azimuth + edge_acceleration_width
                ):
                    modified_values.append(
                        row[0]
                        * edge_acceleration_factor
                        * self.wind_porosity
                        * shelter_height_factor
                    )
                else:
                    modified_values.append(row[0])
        return from_series(
            pd.Series(modified_values, index=df.index, name="Wind Speed (m/s)")
        )

    @property
    def _start_altitude(self) -> float:
        return min(self.altitude_range)

    @property
    def _start_altitude_radians(self) -> float:
        return np.radians(self._start_altitude)

    @property
    def _end_altitude(self) -> float:
        return max(self.altitude_range)

    @property
    def _end_altitude_radians(self) -> float:
        return np.radians(self._end_altitude)

    @property
    def _start_azimuth(self) -> float:
        return self.azimuth_range[0]

    @property
    def _start_azimuth_radians(self) -> float:
        return np.radians(self._start_azimuth)

    @property
    def _end_azimuth(self) -> float:
        return self.azimuth_range[-1]

    @property
    def _end_azimuth_radians(self) -> float:
        return np.radians(self._end_azimuth)

    @property
    def _width(self) -> float:
        return (
            (360 - self._start_azimuth) + self._end_azimuth
            if self._start_azimuth > self._end_azimuth
            else self._end_azimuth - self._start_azimuth
        )

    @property
    def _width_radians(self) -> float:
        return np.radians(self._width)

    @property
    def _height(self) -> float:
        return self._end_altitude - self._start_altitude

    @property
    def _height_radians(self) -> float:
        return np.radians(self._height)

    @property
    def _crosses_north(self) -> bool:
        return self._start_azimuth > self._end_azimuth

    def overlaps(self, other_shelter: Shelter) -> bool:
        """Returns True if this and the other_shelter overlap each other.

        Args:
            other_shelter (Shelter):
                The other shelter to assess for overlap.

        Returns:
            bool:
                True if this and the other_shelter overlap in any way.
        """
        for poly1 in self.polygons():
            for poly2 in other_shelter.polygons():
                if any(
                    [
                        poly1.crosses(poly2),
                        poly1.contains(poly2),
                        poly1.within(poly2),
                        poly1.covers(poly2),
                        poly1.covered_by(poly2),
                        poly1.overlaps(poly2),
                    ]
                ):
                    return True
        return False

    @staticmethod
    def any_shelters_overlap(shelters: List[Shelter]) -> bool:
        """Check whether any shelter in a list overlaps with any other shelter in the list.

        Args:
            shelters (List[Shelter]):
                A list of shelter objects.

        Returns:
            bool:
                True if any shelter in the list overlaps with any other shelter in the list.
        """

        for shelter1 in shelters:
            for shelter2 in shelters:
                if shelter1 == shelter2:
                    continue
                try:
                    if shelter1.overlaps.__wrapped__(shelter2):
                        return True
                except AttributeError:
                    if shelter1.overlaps(shelter2):
                        return True
        return False

    @staticmethod
    def sun_exposure(shelters: List[Shelter], epw: EPW) -> List[float]:
        """Return NaN if sun below horizon, and a value between 0-1 for sun-hidden to sun-exposed.

        Args:
            shelters (List[Shelter]):
                Shelters that could block the sun.
            epw (EPW):
                An EPW object.

        Returns:
            List[float]:
                A list of sun visibility values for each hour of the year.
        """

        suns = sun_position_list(epw)
        sun_is_up = np.array([sun.altitude > 0 for sun in suns])

        nans = np.empty(len(epw.dry_bulb_temperature))
        nans[:] = np.NaN

        if len(shelters) == 0:
            return np.where(sun_is_up, 1, nans)

        blocked = []
        for shelter in shelters:
            temp = np.where(shelter.sun_blocked(suns), shelter.radiation_porosity, nans)
            temp = np.where(np.logical_and(np.isnan(temp), sun_is_up), 1, temp)
            blocked.append(temp)

        return pd.DataFrame(blocked).T.min(axis=1).values.tolist()

    @staticmethod
    def sky_exposure(shelters: List[Shelter]) -> float:
        """Determine the proportion of the sky visible beneath a set of shelters. Includes porosity of
            shelters in the resultant value (e.g. fully enclosed by a single 50% porous shelter would
            mean 50% sky exposure).

        Args:
            shelters (List[Shelter]):
                Shelters that could block the sun.

        Returns:
            float:
                The proportion of sky visible beneath shelters.
        """

        if Shelter.any_shelters_overlap(shelters):
            raise ValueError(
                "Shelters overlap, so sky-exposure calculation cannot be completed."
            )

        exposure = 1
        for shelter in shelters:
            exposure -= shelter.sky_blocked()
        return exposure
