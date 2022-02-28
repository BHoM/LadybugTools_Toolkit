from __future__ import annotations
from typing import List
import numpy as np
from sphericalpolygon import Sphericalpolygon
from ladybug_extension.sun import sun_positions_as_array
from ladybug_extension.datacollection import from_series, to_series
from ladybug_extension.sun import sun_positions
from ladybug.epw import EPW, HourlyContinuousCollection

from regions import PixCoord, PolygonSkyRegion, PolygonPixelRegion
from shapely.geometry.polygon import Point, Polygon

class Shelter2:
    def __init__(self, altitude_range: List[float], azimuth_range: List[float], porosity: float = 0) -> Shelter:
        
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
    
    def _shelter_vertices(self) -> List[List[List[float]]]:
        # TODO - check ndarray dimensions and change output dtype!

        if self.altitude_range[0] == self.altitude_range[1]:
            return []
        
        if self.azimuth_range[0] == self.azimuth_range[1]:
            return []
        
        if self.azimuth_range[0] > self.azimuth_range[1]:
            vertices = []
            for local_azimuth_range in [[self.azimuth_range[0], 360], [0, self.azimuth_range[1]]]:
                vertices.append([
                    [local_azimuth_range[0], self.altitude_range[0]],
                    [local_azimuth_range[0], self.altitude_range[1]],
                    [local_azimuth_range[1], self.altitude_range[1]],
                    [local_azimuth_range[1], self.altitude_range[0]],
                ])
            return vertices
        else:
            return [[
                    [self.azimuth_range[0], self.altitude_range[0]],
                    [self.azimuth_range[0], self.altitude_range[1]],
                    [self.azimuth_range[1], self.altitude_range[1]],
                    [self.azimuth_range[1], self.altitude_range[0]],
                ]]

    def _create_polygons(self) -> List[Polygon]:
        """Create a list of Polygons defining the shelter."""
        polygons = []
        for shelter_vertices in self._shelter_vertices():
            polygons.append(Polygon(shelter_vertices))
        return polygons
    
    def _shelter_area(self) -> float:
        return sum([polygon.area for polygon in self._create_polygons()])
    
    def _sky_area(self) -> float:
        return 360 * 90
    
    @property
    def _sky_view_factor(self) -> float:
        total_sky_area = self._sky_area()
        shelter_area = self._shelter_area()
        sheltered_sky_proportion = shelter_area / total_sky_area
        visible_sky_proportion = 1 - sheltered_sky_proportion
        return (sheltered_sky_proportion * self.porosity) + (visible_sky_proportion * 1)
    
    def _solar_radiation_reduction_factor(self, epw: EPW, solar_disk_radius: float = 5) -> List[float]:
        """For each timestep, return 0 if sun blocked by polygon, 1 if sun not blocked and a value in between if the sun is partially blocked"""
        # TODO  - fix this method so that it reyturns 1 for overnight (sun below horizon), 1 for sun behind shade, and value between 1 and 0 for sun between dshade and solar_disk_radius distance from the shade
        sun_locations = sun_positions_as_array(epw)
        shade_to_sun_distance = []
        for x, y in sun_locations:
            if y < 0:
                shade_to_sun_distance.append(0)
                continue
            pt = Point(y, x)
            if any([poly.contains(pt) for poly in self.polygons]):
                shade_to_sun_distance.append(0)
                continue
            
            distance_to_nearest_polygon = min([poly.exterior.distance(pt) for poly in self.polygons])
            shade_to_sun_distance.append(distance_to_nearest_polygon)
        
        # interpolate to give a factoral value for sun distance
        distances = np.clip(shade_to_sun_distance, 0,  solar_disk_radius)
        distances = 1 - np.interp(distances, [0, solar_disk_radius], [0, 1])
        return distances
    
    def _is_sun_blocked(self, epw: EPW) -> List[bool]:
        sun_locations = sun_positions_as_array(epw)
        masks = []
        for polygon in self.polygons:
            local_mask = []
            for sun_location in sun_locations:
                local_mask.append(polygon.contains(Point(sun_location[1], sun_location[0])))
            masks.append(local_mask)
        return np.any([masks], axis=1)[0]
    
    def __sun_distance_from_polygon(self, epw: EPW) -> List[float]:
        sun_locations = sun_positions_as_array(epw)
        distances = []
        for polygon in self.polygons:
            local_distances = []
            for sun_location in sun_locations:
                
                local_distances.append(polygon.distance(Point(sun_location[1], sun_location[0])))
            distances.append(min(local_distances))
        return np.min(distances, axis=1)
    


class Shelter:
    def __init__(self, vertices: np.ndarray([4, 2], dtype=float), porosity: float = 0) -> Sphericalpolygon:
        """A shelter object capable of providing shade/shelter from wind/sun."""

        self.vertices = np.array(vertices)
        self.porosity = porosity

        if not self.vertices.shape == (4, 2):
            raise ValueError("Shelter vertices must be a list of 4 altitude/azimuth coordinates.")

        if not (0 <= porosity <= 1):
            raise ValueError(f"Porosity must be between 0 and 1")
        
        if (self.min_altitude < 0) or (self.max_altitude > 90):
            raise ValueError(f"Altitude range must fall between 0 and 90")
        
        if (self.min_azimuth < 0) or (self.max_azimuth > 360):
            raise ValueError(f"Azimuth range must fall between 0 and 360")

    @classmethod
    def from_altitude_azimuth_range(cls, altitude_range: List[float], azimuth_range: List[float], porosity: float = 0) -> Shelter:
        """Create a shelter object from an altitude and azimuth range.

        This method expects the shelter altitude range to be between [0, 90], and the azimuth range to be between [0, 360]. The azimuth range may be input as [270, 90] which represents a shelter to the north.
        
        Args:
            altitude_range (List[float]): A list of two floats representing the minimum and maximum altitude of the shelter.
            azimuth_range (List[float]): A list of two floats representing the minimum and maximum azimuth of the shelter.
            porosity (float, optional): The porosity of the shelter that is visible. Defaults to 0, meaning the shelter blocks all wind/radiation.
        
        Returns:
            Shelter: A shelter object.
        """

        if len(altitude_range) != 2:
            raise ValueError("Altitude range must be a list of two floats.")
        if len(azimuth_range) != 2:
            raise ValueError("Azimuth range must be a list of two floats.")
        
        

        vertices = np.array([
            [altitude_range[0], azimuth_range[0]],
            [altitude_range[0], azimuth_range[1]],
            [altitude_range[1], azimuth_range[1]],
            [altitude_range[1], azimuth_range[0]]
        ])
        
        return cls(vertices, porosity)

    def __str__(self) -> str:
        return f"{__class__.__name__}\n{self.vertices}"
    
    def __repr__(self) -> str:
        return str(self)
    
    @property
    def shelter_altitudes(self) -> List[float]:
        """Return a list of the sheltered patch vertex altitudes."""
        return self.vertices[:, 0]
    
    @property
    def shelter_azimuths(self) -> List[float]:
        """Return a list of the sheltered patch vertex azimuths."""
        return self.vertices[:, 1]

    @property
    def min_altitude(self) -> float:
        return self.shelter_altitudes.min()
    
    @property
    def max_altitude(self) -> float:
        return self.shelter_altitudes.max()
    
    @property
    def min_azimuth(self) -> float:
        return self.shelter_azimuths.min()
    
    @property
    def max_azimuth(self) -> float:
        return self.shelter_azimuths.max()
    
    @property
    def sheltered_altitude_range(self) -> List[float]:
        return [self.min_altitude, self.max_altitude]

    @property
    def sheltered_azimuth_range(self) -> List[float]:
        return [self.min_azimuth, self.max_azimuth]

    @property
    def is_sheltered(self) -> bool:
        """Return True if the shelter provides shelter from either wind or sun, and False if not."""
        if self.porosity < 1:
            if self.sky_view_factor < 0.999:
                return True
        return False
    
    @property
    def _sky_polygon(self) -> Sphericalpolygon:
        y0, y1 = [0, 90]
        x0, x1 = [-179.99999, 179.99999]
        vertices = np.array([
            [y0, x0],
            [y0, x1],
            [y1, x1],
            [y1, x0]
        ])
        return Sphericalpolygon.from_array(vertices)
    
    @property
    def _sky_area(self) -> float:
        """The area of the whole sky."""
        return self._sky_polygon.area()

    @property
    def _shelter_polygon(self) -> Sphericalpolygon:
        """Return a Sphericalpolygon object representing the shelter."""
        altitudes = np.clip(self.shelter_altitudes, 0.0001, 89.9999)
        azimuths = np.clip(self.shelter_azimuths, 0.0001, 359.9999)
        azimuths = [np.interp(i, [0.0001, 359.9999], [-179.9999, 179.9999]) for i in azimuths]
        if min(altitudes) == max(altitudes):
            altitudes[1] = altitudes[0] + 0.0001
            altitudes[2] = altitudes[0] + 0.0001
        if min(azimuths) == max(azimuths):
            azimuths[1] = azimuths[0] + 0.0001
            azimuths[2] = azimuths[0] + 0.0001
        vertices = np.stack((altitudes, azimuths), axis=1)
        return Sphericalpolygon.from_array(vertices)

    @property
    def _shelter_area(self) -> float:
        """The area of the shelter as a solid angle."""
        return self._shelter_polygon.area()
    
    @property
    def proportion_sky_visible(self) -> float:
        return (self.sky_solid_angle - self._shelter_area) / self.sky_solid_angle

    @property
    def sky_view_factor(self) -> float:
        """The fraction of the sky that is visible - including the effects of shelter porosity."""
        if self.porosity == 1:
            return 1
        
        proportion_sky_visible = (self.sky_solid_angle - self._shelter_area) / self.sky_solid_angle
        proportion_sky_sheltered = 1 - proportion_sky_visible

        print(self.sky_solid_angle, self._shelter_area)

        return (proportion_sky_sheltered * self.porosity) + (proportion_sky_visible * 1)


        # visible_sky_proportion = 1 - sheltered_sky_proportion
        print(sheltered_sky_proportion)
        # return (visible_sky_proportion * 1) + (sheltered_sky_proportion * self.porosity)
    
    def sun_visible(self, epw: EPW) -> List[bool]:
        """Return a list of True/False where True means the sun is visible and False means it is hidden by the shelter.

        Returns:
            List[bool]: True if sun visible, False if not
        """
        sun_locations = sun_positions_as_array(epw)
        return self._shelter_polygon.contains_points(sun_locations)
    
    def fuzzy_sun_mask(self) -> List[float]:
        """Return a list of floats where 1 means the sun is fully visible and anything less than 1 means there is some shading applied to the solar disk (the region around the sun).

        Returns:
            List[float]: Visibility of the sun (and surrounding region) as a proportion.
        """
        raise NotImplementedError()
    
    def wind_speed_reduction_factor(self, altitude_threshold: float = 45) -> List[float]:
        """Helper method to obtain a wind speed reduction factor, based on the proportion of vertical shelter below the altitude threshold.

        Args:
            altitude_threshold (float, optional): The altitude below which wind speed will be reduced. Defaults to 45.
        
        Returns:
            float: The wind speed reduction factor to be applied to the wind speed.
        """

        maximum_reduction_factor = 0.85

        high = min(altitude_threshold, self.max_altitude)
        low = max(0, self.min_altitude)

        if self.min_altitude > altitude_threshold:
            return 1 * self.porosity

        return np.interp((high - self.min_altitude) / altitude_threshold, [0, 1], [1, maximum_reduction_factor]) * self.porosity

    def effective_wind_speed(self, epw: EPW) -> HourlyContinuousCollection:
        """Adjust wind_speed based on shelter positioning and porosity."""
        if not self.is_sheltered:
            return epw.wind_speed
        else:
            wd = to_series(epw.wind_direction)
            ws = to_series(epw.wind_speed)
            sheltered_mask = wd.between(self.min_azimuth, self.max_azimuth).values
            adjustment_factor = self.wind_speed_reduction_factor(altitude_threshold=45)
            ws_adjusted = ws.where(~sheltered_mask, ws * adjustment_factor)
        return from_series(ws_adjusted)
