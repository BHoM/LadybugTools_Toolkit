from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union
import numpy as np
import pandas as pd
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug_extension.datacollection.from_series import from_series
from ladybug_extension.datacollection.to_series import to_series
from ladybug.sunpath import Sun
from shapely.geometry import Polygon

@dataclass(frozen=True)
class Shelter:
    porosity: float = field(init=True, compare=False, default=1)
    altitude_range: List[float] = field(init=True, compare=True, default_factory=list)
    azimuth_range: List[float] = field(init=True, compare=True, default_factory=list)
    
    def __post_init__(self) -> Shelter:

        if not (0 <= self.porosity <= 1):
            raise ValueError(f"Porosity must be between 0 and 1")

        if len(self.altitude_range) == 0:
            object.__setattr__(self, 'altitude_range', [0, 0])
        elif (len(self.altitude_range) != 2) or (max(self.altitude_range) > 90) or (min(self.altitude_range) < 0):
            raise ValueError(f"{__class__.__name__} altitude range must be a list of two floats between 0 and 90.")

        if len(self.azimuth_range) == 0:
            object.__setattr__(self, 'azimuth_range', [0, 0])
        elif (len(self.azimuth_range) != 2) or (max(self.azimuth_range) > 360) or (min(self.azimuth_range) < 0):
            raise ValueError(f"{__class__.__name__} azimuth range must be a list of two floats between 0 and 90.")
    
    def _polygons(self) -> List[Polygon]:
        """Return a list of polygons representing the shade of this shelter.

        Returns:
            List[Polygon]: A list of polygons representing the shade of this shelter.
        """        
        if self._width == 0:
            return []

        if self._height == 0:
            return []
        
        if self.crosses_north:
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
            List[bool]: A list of booleans indicating whether the sun (or each of the suns) is blocked by this shelter.
        """        
        if not isinstance(suns, list):
            suns = [suns]
        blocked = []
        for s in suns:
            if not isinstance(s, Sun):
                raise ValueError("Object input is not a Sun.")

            in_altitude_range = self._start_altitude < s.altitude < self._end_altitude

            if self.crosses_north:
                in_azimuth_range = (s.azimuth > self._start_azimuth) or (
                    s.azimuth < self._end_azimuth
                )
            else:
                in_azimuth_range = self._start_azimuth < s.azimuth < self._end_azimuth

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
        
        shelter_height_factor = np.interp(self._height / 90, [0, 1], [1, 0.25])

        wd = to_series(epw.wind_direction)
        ws = to_series(epw.wind_speed)
        df = pd.concat([ws, wd], axis=1)
        modified_values = []
        for _, row in df.iterrows():
            if self.crosses_north:
                if (row[1] > self._start_azimuth) or (row[1] < self._end_azimuth):
                    modified_values.append(row[0] * self.porosity * shelter_height_factor)
                else:
                    modified_values.append(row[0])
            else:
                if self._start_azimuth < row[1] < self._end_azimuth:
                    modified_values.append(row[0] * self.porosity * shelter_height_factor)
                else:
                    modified_values.append(row[0])
        return from_series(pd.Series(modified_values, index=df.index, name="Wind Speed (m/s)"))

    def overlaps(self, other: Shelter) -> bool:
        """Return True if this shelter overlaps with another"""
        return self._overlaps([self, other])

    @property
    def description(self) -> str:
        if (self._height == 0) or (self._width == 0) or (self.porosity == 1):
            return None
        return f"sheltered between ({self._start_azimuth}, {self._start_altitude}) and ({self._end_azimuth}, {self._end_altitude}) with porosity of {self.porosity:0.0%}"
    
    @property
    def _start_azimuth(self) -> float:
        """The azimuth at which the shelter starts"""
        return self.azimuth_range[0]

    @property
    def _end_azimuth(self) -> float:
        """The azimuth at which the shelter ends"""
        return self.azimuth_range[1]

    @property
    def _start_azimuth_radians(self) -> float:
        """The azimuth of the start of the shelter in radians"""
        return np.radians(self._start_azimuth)

    @property
    def _end_azimuth_radians(self) -> float:
        """The azimuth of the end of the shelter in radians"""
        return np.radians(self._end_azimuth)

    @property
    def _width(self) -> float:
        """The width of the shelter (degrees)"""
        if self._start_azimuth > self._end_azimuth:
            return (360 - self._start_azimuth) + self._end_azimuth
        else:
            return self._end_azimuth - self._start_azimuth

    @property
    def _width_radians(self) -> float:
        """The width of the shelter (radians)"""
        return np.radians(self._width)

    @property
    def _start_altitude(self) -> float:
        """The altitude at which the shelter starts"""
        return min(self.altitude_range)

    @property
    def _end_altitude(self) -> float:
        """The altitude at which the shelter ends"""
        return max(self.altitude_range)

    @property
    def _start_altitude_radians(self) -> float:
        """The altitude of the start of the shelter in radians"""
        return np.radians(self._start_altitude)

    @property
    def _end_altitude_radians(self) -> float:
        """The altitude of the end of the shelter in radians"""
        return np.radians(self._end_altitude)

    @property
    def _height(self) -> float:
        """The height of the shelter (degrees)"""
        return self._end_altitude - self._start_altitude

    @property
    def _height_radians(self) -> float:
        """The height of the shelter (radians)"""
        return np.radians(self._height)

    @property
    def crosses_north(self) -> bool:
        """Whether the shelter crosses the north azimuth"""
        if self._start_azimuth > self._end_azimuth:
            return True
        return False

    @property
    def sky_occluded(self) -> float:
        """Return the proportion of the sky blocked by this shelter"""
        occluded_proportion = (
            (np.sin(self._end_altitude_radians) - np.sin(self._start_altitude_radians))
            * self._width_radians
            / (2 * np.pi)
        )
        return occluded_proportion

    @property
    def sky_visible(self) -> float:
        """Return the proportion of the sky visible around this shelter"""
        return 1 - self.sky_occluded

    @staticmethod
    def _overlaps(shelters: List[Shelter]) -> bool:
        """Check whether any shelter in a list overlaps with any other shelter in the list.

        Args:
            shelters (List[Shelter]): A list of shelter objects.

        Returns:
            bool: True if any shelter in the list overlaps with any other shelter in the list.
        """

        def __overlaps(shelter1: Shelter, shelter2: Shelter) -> bool:
            """Return True if two shelters overlap with each other.

            Args:
                shelter1 (Shelter): The first shelter to compare.
                shelter2 (Shelter): The second shelter to compare.

            Returns:
                bool: True if the two shelters overlap with each other.
            """    

            for poly1 in shelter1._polygons():
                for poly2 in shelter2._polygons():
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

        for shelter1 in shelters:
            for shelter2 in shelters:
                if shelter1 != shelter2:
                    if __overlaps(shelter1, shelter2):
                        return True
        return False
