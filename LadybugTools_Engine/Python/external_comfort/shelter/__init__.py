from __future__ import annotations

import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug_extension.datacollection.from_series import from_series
from ladybug_extension.datacollection.to_series import to_series
from ladybug.sunpath import Sun
from shapely.geometry import Polygon


class Shelter:
    def __init__(
        self,
        porosity: float = 0.0,
        altitude_range: Tuple[float] = (0, 90),
        azimuth_range: Tuple[float] = (0, 360),
    ):
        """A shelter describing the solar and wind protection affored by a specific external comfort typology.

        Args:
            altitude_range (Tuple[float]): The range of altitudes in degrees above the horizon at which the shelter is effective.
            azimuth_range (Tuple[float]): The range of azimuths in degrees from north at which the shelter is effective.
            porosity (float): The proportion of the shelter area that is exposed to the sun.

        Returns:
            Shelter: A shelter object.
        """

        if not (0 <= porosity <= 1):
            raise ValueError(f"Porosity must be between 0 and 1")

        if len(altitude_range) != 2:
            raise ValueError("Altitude range must be a list of two floats.")

        if (min(altitude_range) < 0) or (max(altitude_range) > 90):
            raise ValueError(f"Altitude range must fall between 0 and 90")

        if len(azimuth_range) != 2:
            raise ValueError("Azimuth range must be a list of two floats.")

        if (min(azimuth_range) < 0) or (max(azimuth_range) > 360):
            raise ValueError(f"Azimuth range must fall between 0 and 360")

        self.porosity = porosity
        self.altitude_range = altitude_range
        self.azimuth_range = azimuth_range

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(porosity={self.porosity}, altitude_range={self.altitude_range}, azimuth_range={self.azimuth_range})"

    def __eq__(self, shelter: Shelter) -> bool:
        if (self.altitude_range == shelter.altitude_range) and (
            self.azimuth_range == shelter.azimuth_range
        ):
            return True
        return False

    def polygons(self) -> List[Polygon]:
        """Return a list of polygons representing the shade of this shelter.

        Returns:
            List[Polygon]: A list of polygons representing the shade of this shelter.
        """        
        if self.width == 0:
            return []

        if self.height == 0:
            return []
        
        if self.crosses_north:
            return [
                Polygon(
                    [
                        (self.start_azimuth, self.start_altitude),
                        (self.start_azimuth, self.end_altitude),
                        (360, self.end_altitude),
                        (360, self.start_altitude),
                        (self.start_azimuth, self.start_altitude),
                    ]
                ),
                Polygon(
                    [
                        (0, self.start_altitude),
                        (0, self.end_altitude),
                        (self.end_azimuth, self.end_altitude),
                        (self.end_azimuth, self.start_altitude),
                        (0, self.start_altitude),
                    ]
                ),
            ]
        else:
            return [
                Polygon(
                    [
                        (self.start_azimuth, self.start_altitude),
                        (self.start_azimuth, self.end_altitude),
                        (self.end_azimuth, self.end_altitude),
                        (self.end_azimuth, self.start_altitude),
                    ]
                )
            ]

    def sun_blocked(self, suns: Union[List[Sun], Sun]) -> List[bool]:
        """Return a list of booleans indicating whether the sun is blocked by this shelter.

        Args:
            suns (Union[List[Sun], Sun]): Either a Sun object or a list of Sun objects.

        Returns:
            List[bool]: A list of booleans indicating whether the sun (or each of the suns) is blocked by this shelter.
        """        
        if not isinstance(suns, list):
            suns = [suns]
        blocked = []
        for s in suns:
            if not isinstance(s, Sun):
                raise ValueError("Object input is not a Sun.")

            in_altitude_range = self.start_altitude < s.altitude < self.end_altitude

            if self.crosses_north:
                in_azimuth_range = (s.azimuth > self.start_azimuth) or (
                    s.azimuth < self.end_azimuth
                )
            else:
                in_azimuth_range = self.start_azimuth < s.azimuth < self.end_azimuth

            if in_altitude_range and in_azimuth_range:
                blocked.append(True)
            else:
                blocked.append(False)
        return blocked
    
    def effective_wind_speed(self, epw: EPW) -> HourlyContinuousCollection:
        """Return the wind speed (at original height of 10m from EPW) when subjected to this shelter. 
           The proportion of shelter occluding the altitude, and the porosity of the shelter determines how much of the wind to block

        Args:
            epw (EPW): An EPW object.

        Returns:
            HourlyContinuousCollection: An HourlyContinuousCollection object with the effective wind speed impacted by this shelter.
        """        
        
        if self.porosity == 1:
            return epw.wind_speed
        
        shelter_height_factor = np.interp(self.height / 90, [0, 1], [1, 0.25])

        wd = to_series(epw.wind_direction)
        ws = to_series(epw.wind_speed)
        df = pd.concat([ws, wd], axis=1)
        modified_values = []
        for _, row in df.iterrows():
            if self.crosses_north:
                if (row[1] > self.start_azimuth) or (row[1] < self.end_azimuth):
                    modified_values.append(row[0] * self.porosity * shelter_height_factor)
                else:
                    modified_values.append(row[0])
            else:
                if self.start_azimuth < row[1] < self.end_azimuth:
                    modified_values.append(row[0] * self.porosity * shelter_height_factor)
                else:
                    modified_values.append(row[0])
        return from_series(pd.Series(modified_values, index=df.index, name="Wind Speed (m/s)"))

    def overlaps(self, other: Shelter) -> bool:
        """Return True if this shelter overlaps with another"""
        return overlaps([self, other])

    @property
    def description(self) -> str:
        if (self.height == 0) or (self.width == 0) or (self.porosity == 1):
            return None
        return f"sheltered between ({self.start_azimuth}, {self.start_altitude}) and ({self.end_azimuth}, {self.end_altitude}) with porosity of {self.porosity:0.0%}"
    
    @property
    def start_azimuth(self) -> float:
        """The azimuth at which the shelter starts"""
        return self.azimuth_range[0]

    @property
    def end_azimuth(self) -> float:
        """The azimuth at which the shelter ends"""
        return self.azimuth_range[1]

    @property
    def start_azimuth_radians(self) -> float:
        """The azimuth of the start of the shelter in radians"""
        return np.radians(self.start_azimuth)

    @property
    def end_azimuth_radians(self) -> float:
        """The azimuth of the end of the shelter in radians"""
        return np.radians(self.end_azimuth)

    @property
    def width(self) -> float:
        """The width of the shelter (degrees)"""
        if self.start_azimuth > self.end_azimuth:
            return (360 - self.start_azimuth) + self.end_azimuth
        else:
            return self.end_azimuth - self.start_azimuth

    @property
    def width_radians(self) -> float:
        """The width of the shelter (radians)"""
        return np.radians(self.width)

    @property
    def start_altitude(self) -> float:
        """The altitude at which the shelter starts"""
        return min(self.altitude_range)

    @property
    def end_altitude(self) -> float:
        """The altitude at which the shelter ends"""
        return max(self.altitude_range)

    @property
    def start_altitude_radians(self) -> float:
        """The altitude of the start of the shelter in radians"""
        return np.radians(self.start_altitude)

    @property
    def end_altitude_radians(self) -> float:
        """The altitude of the end of the shelter in radians"""
        return np.radians(self.end_altitude)

    @property
    def height(self) -> float:
        """The height of the shelter (degrees)"""
        return self.end_altitude - self.start_altitude

    @property
    def height_radians(self) -> float:
        """The height of the shelter (radians)"""
        return np.radians(self.height)

    @property
    def crosses_north(self) -> bool:
        """Whether the shelter crosses the north azimuth"""
        if self.start_azimuth > self.end_azimuth:
            return True
        return False

    @property
    def sky_occluded(self) -> float:
        """Return the proportion of the sky blocked by this shelter"""
        occluded_proportion = (
            (np.sin(self.end_altitude_radians) - np.sin(self.start_altitude_radians))
            * self.width_radians
            / (2 * np.pi)
        )
        return occluded_proportion

    @property
    def sky_visible(self) -> float:
        """Return the proportion of the sky visible around this shelter"""
        return 1 - self.sky_occluded

def _overlaps(shelter1: Shelter, shelter2: Shelter) -> bool:
    """Return True if two shelters overlap with each other.

    Args:
        shelter1 (Shelter): The first shelter to compare.
        shelter2 (Shelter): The second shelter to compare.

    Returns:
        bool: True if the two shelters overlap with each other.
    """    

    for poly1 in shelter1.polygons():
        for poly2 in shelter2.polygons():
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

def overlaps(shelters: List[Shelter]) -> bool:
    """Checjk whether any shelter in a list overlaps with any other shelter in the list.

    Args:
        shelters (List[Shelter]): A list of shelter objects.

    Returns:
        bool: True if any shelter in the list overlaps with any other shelter in the list.
    """    
    for shelter1 in shelters:
        for shelter2 in shelters:
            if shelter1 != shelter2:
                if _overlaps(shelter1, shelter2):
                    return True
    return False
