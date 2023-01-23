from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, cartesian_to_spherical, spherical_to_cartesian
from ladybug.epw import EPW, AnalysisPeriod, HourlyContinuousCollection, Location
from ladybug.hourlyplot import HourlyPlot
from ladybug.sunpath import Sun
from ladybug_geometry.geometry2d import Vector2D
from ladybug_geometry.geometry3d import Vector3D
from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from ladybugtools_toolkit.helpers import wind_direction_average
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon
from spherical_geometry.polygon import SphericalPolygon
from spherical_geometry.vector import (
    lonlat_to_vector,
    normalize_vector,
    vector_to_radec,
)

from ..bhomutil.bhom_object import BHoMObject, bhom_dict_to_dict
from ..ladybug_extension.datacollection import from_series, to_series
from ..ladybug_extension.epw import Sunpath, sun_position_list


def altaz_to_xyz(altitude: float, azimuth: float) -> Tuple[float]:
    """Convert an altitude and azimuth into xyz coordinates, with x=right, y=forwards, z=up."""

    if (altitude > 90) or (altitude < 0):
        raise ValueError(f"altitude must be in range 0-90")

    # remap azimuth to 0-360 range
    azimuth = azimuth % 360

    # convert inputs to radians
    azimuth = np.deg2rad(azimuth)
    altitude = np.deg2rad(altitude)

    x_axis = Vector3D(1, 0, 0)
    north_vector = Vector3D(0, 1, 0)

    x, y, z = north_vector.rotate(x_axis, altitude).rotate_xy(-azimuth).to_array()
    return x, y, z


def xyz_to_altaz(x: float, y: float, z: float) -> Tuple[float]:
    """Convert an xyz vector into altitude, azimuth coordinates, with altitude in degrees elevation above horizon and azimuth in degrees clockwise from North."""
    vector = Vector3D(x, y, z).normalize()

    try:
        azimuth = np.rad2deg(
            Vector2D(0, 1).angle_clockwise(Vector2D(vector.x, vector.y))
        )
    except ZeroDivisionError:
        return 90.0, 0.0

    altitude = 90 - np.rad2deg(
        np.arccos(vector.z / np.sqrt(vector.x**2 + vector.y**2 + vector.z**2))
    )

    return altitude, float(azimuth) % 360


def point3d_to_altaz(point: Point3D) -> Tuple[float]:
    """Convert a ladybug Point3D object into a altitude, azimuth coordinate."""
    x, y, z = point
    return xyz_to_altaz(x, y, z)


def altaz_to_skycoord(altitude: float, azimuth: float) -> SkyCoord:
    """Convert an altitude, azimuth coordinate to a SkyCoord object"""
    return SkyCoord(
        alt=altitude * u.degree,
        az=azimuth * u.degree,
        frame="altaz",
        representation_type="spherical",
    )


@dataclass(init=True, repr=True, eq=True)
class ShelterRaDec(BHoMObject):
    vertices: List[List[float]] = field(
        init=True, repr=False
    )  # a list of ra-dec coordinates representing the vertices of the sheltering object
    wind_porosity: float = field(init=True, repr=True, compare=True, default=0)
    radiation_porosity: float = field(init=True, repr=True, compare=True, default=0)
    inside: List[float] = field(init=True, repr=False, default=None)

    _t: str = field(
        init=False, repr=False, compare=True, default="BH.oM.LadybugTools.Shelter"
    )

    @staticmethod
    def xyz_to_radec(
        xyz: List[List[float]], north_angle: float = 0
    ) -> List[List[float]]:
        xyz: np.ndarray = np.array(xyz)
        # check that size is (n, 3)
        if len(xyz.shape) != 2:
            raise ValueError(
                "xyz must be a list of x, y, z coordinates and 2-dimensional ([[x, y, z], ...])."
            )
        if xyz.shape[-1] != 3:
            raise ValueError(f"xyz shape must be (n, 3), currently it is {xyz.shape}.")
        ra, dec = vector_to_radec(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])
        # ra += 90 + north_angle
        return ra, dec


@dataclass(init=True, repr=True, eq=True)
class Shelter(BHoMObject):

    vertices: List[List[float]] = field(
        init=True, repr=False
    )  # a list of xyz coordinates representing the vertices of the sheltering object
    wind_porosity: float = field(init=True, repr=True, compare=True, default=0)
    radiation_porosity: float = field(init=True, repr=True, compare=True, default=0)

    _inside: List[float] = field(init=True, repr=False, default=None)

    _t: str = field(
        init=False, repr=False, compare=True, default="BH.oM.LadybugTools.Shelter"
    )

    # TODO - convert all coordinates internally to ra-dec form for easier processing

    def __post_init__(self):

        self.vertices = np.array(self.vertices)
        # check for 100% porosity
        if all([self.wind_porosity == 1, self.radiation_porosity == 1]):
            raise ValueError(
                "This shelter would have no effect as it allows all wind and radiation through"
            )

        # handle "null" values from classmethods and replace with defaults
        if self.wind_porosity is None:
            self.wind_porosity = 0
        if self.radiation_porosity is None:
            self.radiation_porosity = 0

        # check porosity range limits
        if (not 0 <= self.wind_porosity <= 1) or (
            not 0 <= self.radiation_porosity <= 1
        ):
            raise ValueError("porosity must be between 0 and 1")

        # check for potentially bad geometry
        sp = self.spherical_polygon
        # if len(list(sp.to_lonlat())) != 1:
        #     raise ValueError("more than 1 spherical_polygon created by this object!")

        self._inside = self.spherical_polygon.inside

        # check for self-intersection
        try:
            vector_to_radec(*np.array(list(self.spherical_polygon.inside)).flatten())
        except TypeError:
            raise ValueError("Shelter polygon created by inputs is self-intersecting!")

        # check for sub-ground shelter
        if sum(self.vertices[:, -1] < 0) > 0:
            raise ValueError("vertices cannot have z-coordinates < 0")

        # wrap methods within this class
        super().__post_init__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vertices={len(self.vertices)}, wind_porosity={self.wind_porosity}, radiation_porosity={self.radiation_porosity})"

    @property
    def sky_area(self) -> float:
        """The area of the sky, in steradians."""
        return 2 * np.pi

    @property
    def sky_coords(self) -> SkyCoord:
        return SkyCoord(
            x=self.vertices.T[0],
            y=self.vertices.T[1],
            z=self.vertices.T[2],
            representation_type="cartesian",
        )

    @property
    def spherical_polygon(self) -> SphericalPolygon:
        return SphericalPolygon(
            [i.cartesian.xyz.value for i in self.sky_coords], inside=self._inside
        )

    @staticmethod
    def sun_to_sky_coord(sun: Sun) -> SkyCoord:
        return SkyCoord(ra=sun.azimuth * u.degree, dec=sun.altitude * u.degree)

    def sun_exposure(self, suns: Union[Sun, List[Sun]]) -> List[float]:

        if isinstance(suns, Sun):
            suns = [suns]

        spherical_polygon = self.spherical_polygon

        exposure = []
        for sun in suns:
            if sun.altitude <= 0:
                exposure.append(0)
                continue
            sun_coord = self.sun_to_sky_coord(sun)
            if spherical_polygon.contains_point(sun_coord.cartesian.xyz.value.T):
                exposure.append(self.radiation_porosity)
            else:
                exposure.append(1)

        return exposure

    def sun_exposure_location(
        self, location: Location, analysis_period: AnalysisPeriod = AnalysisPeriod()
    ) -> List[float]:
        sunpath = Sunpath.from_location(location)
        suns = [sunpath.calculate_sun_from_hoy(i) for i in analysis_period.hoys]
        return self.sun_exposure(suns)

    def sun_exposure_epw(
        self, epw: EPW, analysis_period: AnalysisPeriod = AnalysisPeriod()
    ) -> List[float]:
        return self.sun_exposure_location(epw.location, analysis_period)

    def wind_exposure(self, azimuth: float) -> float:
        """Get the factor to be applied to wind from the given azimuth."""

        if self.wind_porosity == 1:
            return 1

        # create segment where wind wind comes from
        wind_range = np.array([azimuth - 5, azimuth + 5]) % 360
        az_inside = wind_direction_average(wind_range)
        pt_inside = (
            spherical_to_cartesian(
                r=1,
                lon=az_inside * u.degree,
                lat=45 * u.degree,
            ),
        )
        vertices = np.array(
            [
                spherical_to_cartesian(
                    r=1,
                    lon=wind_range[0] * u.degree,
                    lat=0 * u.degree,
                ),
                spherical_to_cartesian(
                    r=1,
                    lon=wind_range[1] * u.degree,
                    lat=0 * u.degree,
                ),
                spherical_to_cartesian(
                    r=1,
                    lon=wind_range[1] * u.degree,
                    lat=90 * u.degree,
                ),
                spherical_to_cartesian(
                    r=1,
                    lon=wind_range[0] * u.degree,
                    lat=90 * u.degree,
                ),
            ]
        )
        wind_segment = SphericalPolygon(vertices, pt_inside)

        # get proportion of intersection
        intersection_amount = 1 - wind_segment.overlap(self.spherical_polygon)

        # TODO - fix acceleration part around edges here
        # Also fix malformed polygon errot that sometimes happens when
        # sh = Shelter.from_ranges(azimuth_range=[-5, 10], altitude_range=[0, 90])
        # f = sh.plot()
        # sh.wind_exposure(2)

        # if proportion < 0.1, then increase exposure by 10% to approximate edge effects
        if intersection_amount < 0.1:
            intersection_amount = 1 + (intersection_amount / 0.1)

        return intersection_amount

    @classmethod
    def from_ranges(
        cls,
        azimuth_range: List[float],
        altitude_range: List[float],
        wind_porosity: float = None,
        radiation_porosity: float = None,
    ) -> Shelter:

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
                len(azimuth_range) != 2,
            ]
        ):
            raise ValueError("azimuth_range must be two floats in degrees about north")

        # convert to array
        altitude_range = np.array(altitude_range)
        azimuth_range = np.array(azimuth_range) % 360

        # get angle difference
        if azimuth_range[0] < azimuth_range[1]:
            angle_between = np.diff(azimuth_range)[0]
        else:
            angle_between = 360 - azimuth_range.max() + azimuth_range.min()

        # get point "inside" the range as an xyz triple
        az_inside = wind_direction_average(azimuth_range)
        alt_inside = altitude_range.mean()
        pt_inside = (
            spherical_to_cartesian(
                r=1,
                lon=az_inside * u.degree,
                lat=alt_inside * u.degree,
            ),
        )

        # convert ranges into XYZ vertices
        vertices = np.array(
            [
                spherical_to_cartesian(
                    r=1,
                    lon=azimuth_range[0] * u.degree,
                    lat=altitude_range[0] * u.degree,
                ),
                spherical_to_cartesian(
                    r=1,
                    lon=azimuth_range[1] * u.degree,
                    lat=altitude_range[0] * u.degree,
                ),
                spherical_to_cartesian(
                    r=1,
                    lon=azimuth_range[1] * u.degree,
                    lat=altitude_range[1] * u.degree,
                ),
                spherical_to_cartesian(
                    r=1,
                    lon=azimuth_range[0] * u.degree,
                    lat=altitude_range[1] * u.degree,
                ),
            ]
        )

        obj = cls(
            vertices=vertices,
            wind_porosity=wind_porosity,
            radiation_porosity=radiation_porosity,
            _inside=pt_inside,
        )

        return obj

    def area(self) -> float:
        area = self.spherical_polygon.area()

        # alternative method using Greens Theorem originally from https://github.com/anutkk/sphericalgeometry
        if np.isnan(area):
            print("wasnan")
            lons, lats = np.array(list(self.spherical_polygon.to_lonlat())[0])

            # colatitudes relative to (0,0)
            a = np.sin(lats / 2) ** 2 + np.cos(lats) * np.sin(lons / 2) ** 2
            colat = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            # azimuths relative to (0,0)
            az = np.arctan2(np.cos(lats) * np.sin(lons), np.sin(lats)) % (2 * np.pi)

            # Calculate diffs
            # daz = diff(az) % (2*pi)
            daz = np.diff(az)
            daz = (daz + np.pi) % (2 * np.pi) - np.pi

            deltas = np.diff(colat) / 2
            colat = colat[0:-1] + deltas

            # Perform integral
            integrands = (1 - np.cos(colat)) * daz

            # Integrate
            area = abs(sum(integrands)) / (4 * np.pi)

            area = abs(min(area, 1 - area))

            # convert ratio of sphere total area to steradian
            area = area * 2 * np.pi

            return area

        return area

    def plot(self, suns: Union[Sun, List[Sun]] = None) -> plt.Figure:
        # get polygon vertices
        _x, _y = np.stack(list(self.spherical_polygon.to_radec()), axis=0)[0]
        inside = vector_to_radec(
            *np.array(list(self.spherical_polygon.inside)).flatten()
        )
        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        [ax.tick_params(axis=i, colors="lightgrey") for i in ["x", "y"]]
        ax.grid(alpha=0.5, ls=":", color="lightgrey")
        ax.spines["polar"].set_visible(False)
        ax.invert_yaxis()
        ax.set_ylim(90, -1)
        _x = np.deg2rad(_x)
        ax.scatter(_x, _y, c="red", s=5)
        patches = [mpatch.Polygon(np.array([_x, _y]).T, closed=True)]
        p = PatchCollection(patches, alpha=0.4)
        ax.add_collection(p)
        ax.scatter(np.deg2rad(inside[0]), inside[1])

        if suns is not None:
            if isinstance(suns, Sun):
                suns = [suns]
            exposure = self.sun_exposure(suns)
            sun_coords = SkyCoord([self.sun_to_sky_coord(sun) for sun in suns])
            ax.scatter(
                np.deg2rad(sun_coords.ra.value),
                sun_coords.dec.value,
                c=["y" if exp else "b" for exp in exposure],
            )

        return fig

    def sky_sheltered(self) -> float:
        """The proportion of the sky blocked by this shelter (not incorporating any porosity)."""
        return self.area() / (2 * np.pi)

    def sky_exposure(self, include_porosity: bool = True) -> float:
        """Calculate the proportion of sky a person is exposed to, optionally including the porosity of the shelter."""
        sheltered = self.sky_sheltered()
        if include_porosity:
            return 1 - ((sheltered * (1 - self.radiation_porosity)) / self.sky_area)
        return 1 - sheltered

    def overlaps(self, other: Shelter) -> bool:
        if self.spherical_polygon.overlap(other.spherical_polygon) > 0:
            return True
        return False

    # @classmethod
    # def from_dict(cls, dictionary: Dict[str, Any]) -> Shelter:
    #     """Create this object from a dictionary."""

    #     sanitised_dict = bhom_dict_to_dict(dictionary)
    #     sanitised_dict.pop("_t", None)

    #     return cls(
    #         wind_porosity=sanitised_dict["wind_porosity"],
    #         radiation_porosity=sanitised_dict["radiation_porosity"],
    #         altitude_range=sanitised_dict["altitude_range"],
    #         azimuth_range=sanitised_dict["azimuth_range"],
    #     )

    # @classmethod
    # def from_json(cls, json_string: str) -> Shelter:
    #     """Create this object from a JSON string."""

    #     dictionary = json.loads(json_string)

    #     return cls.from_dict(dictionary)

    # def effective_wind_speed(self, epw: EPW) -> HourlyContinuousCollection:
    #     """Return the wind speed (at original height of 10m from EPW) when subjected to this
    #         shelter.

    #        The proportion of shelter occluding the altitude, and the porosity of the shelter
    #         determines how much of the wind to block. If not blocked by the shelter, but within 10°,
    #         a 10% increase is applied to estimate impact of edge acceleration.

    #     Args:
    #         epw (EPW): An EPW object.

    #     Returns:
    #         HourlyContinuousCollection: An HourlyContinuousCollection object with the effective wind
    #         speed impacted by this shelter.
    #     """

    #     if self.wind_porosity == 1:
    #         return epw.wind_speed

    #     edge_acceleration_width = (
    #         10  # degrees either side of edge in which to increase wind speed
    #     )
    #     edge_acceleration_factor = (
    #         1.1  # amount by which to increase wind speed by for edge effects
    #     )

    #     shelter_height_factor = np.interp(self._height / 90, [0, 1], [1, 0.25])

    #     wind_direction = to_series(epw.wind_direction)
    #     wind_speed = to_series(epw.wind_speed)
    #     df = pd.concat([wind_speed, wind_direction], axis=1)
    #     modified_values = []
    #     for _, row in df.iterrows():
    #         if self._crosses_north:
    #             # wind blocked by shelter
    #             if (row[1] > self._start_azimuth) or (row[1] < self._end_azimuth):
    #                 modified_values.append(
    #                     row[0] * self.wind_porosity * shelter_height_factor
    #                 )
    #             # wind not blocked by shelter, but it's within 10deg of shelter
    #             elif (row[1] > self._start_azimuth - edge_acceleration_width) or (
    #                 row[1] < self._end_azimuth + edge_acceleration_width
    #             ):
    #                 modified_values.append(
    #                     row[0]
    #                     * edge_acceleration_factor
    #                     * self.wind_porosity
    #                     * shelter_height_factor
    #                 )
    #             # wind not blocked by shelter
    #             else:
    #                 modified_values.append(row[0])
    #         else:
    #             # wind blocked by shelter
    #             if (row[1] > self._start_azimuth) and (row[1] < self._end_azimuth):
    #                 modified_values.append(
    #                     row[0] * self.wind_porosity * shelter_height_factor
    #                 )
    #             # wind not blocked by shelter, but it's within 10deg of shelter
    #             elif (row[1] > self._start_azimuth - edge_acceleration_width) and (
    #                 row[1] < self._end_azimuth + edge_acceleration_width
    #             ):
    #                 modified_values.append(
    #                     row[0]
    #                     * edge_acceleration_factor
    #                     * self.wind_porosity
    #                     * shelter_height_factor
    #                 )
    #             else:
    #                 modified_values.append(row[0])
    #     return from_series(
    #         pd.Series(modified_values, index=df.index, name="Wind Speed (m/s)")
    #     )

    # @property
    # def _start_altitude(self) -> float:
    #     return min(self.altitude_range)

    # @property
    # def _start_altitude_radians(self) -> float:
    #     return np.radians(self._start_altitude)

    # @property
    # def _end_altitude(self) -> float:
    #     return max(self.altitude_range)

    # @property
    # def _end_altitude_radians(self) -> float:
    #     return np.radians(self._end_altitude)

    # @property
    # def _start_azimuth(self) -> float:
    #     return self.azimuth_range[0]

    # @property
    # def _start_azimuth_radians(self) -> float:
    #     return np.radians(self._start_azimuth)

    # @property
    # def _end_azimuth(self) -> float:
    #     return self.azimuth_range[-1]

    # @property
    # def _end_azimuth_radians(self) -> float:
    #     return np.radians(self._end_azimuth)

    # @property
    # def _width(self) -> float:
    #     return (
    #         (360 - self._start_azimuth) + self._end_azimuth
    #         if self._start_azimuth > self._end_azimuth
    #         else self._end_azimuth - self._start_azimuth
    #     )

    # @property
    # def _width_radians(self) -> float:
    #     return np.radians(self._width)

    # @property
    # def _height(self) -> float:
    #     return self._end_altitude - self._start_altitude

    # @property
    # def _height_radians(self) -> float:
    #     return np.radians(self._height)

    # @property
    # def _crosses_north(self) -> bool:
    #     return self._start_azimuth > self._end_azimuth

    # def overlaps(self, other_shelter: Shelter) -> bool:
    #     """Returns True if this and the other_shelter overlap each other.

    #     Args:
    #         other_shelter (Shelter):
    #             The other shelter to assess for overlap.

    #     Returns:
    #         bool:
    #             True if this and the other_shelter overlap in any way.
    #     """
    #     for poly1 in self.polygons():
    #         for poly2 in other_shelter.polygons():
    #             if any(
    #                 [
    #                     poly1.crosses(poly2),
    #                     poly1.contains(poly2),
    #                     poly1.within(poly2),
    #                     poly1.covers(poly2),
    #                     poly1.covered_by(poly2),
    #                     poly1.overlaps(poly2),
    #                 ]
    #             ):
    #                 return True
    #     return False

    # @staticmethod
    # def any_shelters_overlap(shelters: List[Shelter]) -> bool:
    #     """Check whether any shelter in a list overlaps with any other shelter in the list.

    #     Args:
    #         shelters (List[Shelter]):
    #             A list of shelter objects.

    #     Returns:
    #         bool:
    #             True if any shelter in the list overlaps with any other shelter in the list.
    #     """

    #     for shelter1 in shelters:
    #         for shelter2 in shelters:
    #             if shelter1 == shelter2:
    #                 continue
    #             try:
    #                 if shelter1.overlaps.__wrapped__(shelter2):
    #                     return True
    #             except AttributeError:
    #                 if shelter1.overlaps(shelter2):
    #                     return True
    #     return False

    # @staticmethod
    # def sun_exposure(shelters: List[Shelter], epw: EPW) -> List[float]:
    #     """Return NaN if sun below horizon, and a value between 0-1 for sun-hidden to sun-exposed.

    #     Args:
    #         shelters (List[Shelter]):
    #             Shelters that could block the sun.
    #         epw (EPW):
    #             An EPW object.

    #     Returns:
    #         List[float]:
    #             A list of sun visibility values for each hour of the year.
    #     """

    #     suns = sun_position_list(epw)
    #     sun_is_up = np.array([sun.altitude > 0 for sun in suns])

    #     nans = np.empty(len(epw.dry_bulb_temperature))
    #     nans[:] = np.NaN

    #     if len(shelters) == 0:
    #         return np.where(sun_is_up, 1, nans)

    #     blocked = []
    #     for shelter in shelters:
    #         temp = np.where(shelter.sun_blocked(suns), shelter.radiation_porosity, nans)
    #         temp = np.where(np.logical_and(np.isnan(temp), sun_is_up), 1, temp)
    #         blocked.append(temp)

    #     return pd.DataFrame(blocked).T.min(axis=1).values.tolist()

    # @staticmethod
    # def sky_exposure(shelters: List[Shelter]) -> float:
    #     """Determine the proportion of the sky visible beneath a set of shelters. Includes porosity of
    #         shelters in the resultant value (e.g. fully enclosed by a single 50% porous shelter would
    #         mean 50% sky exposure).

    #     Args:
    #         shelters (List[Shelter]):
    #             Shelters that could block the sun.

    #     Returns:
    #         float:
    #             The proportion of sky visible beneath shelters.
    #     """

    #     if Shelter.any_shelters_overlap(shelters):
    #         raise ValueError(
    #             "Shelters overlap, so sky-exposure calculation cannot be completed."
    #         )

    #     exposure = 1
    #     for shelter in shelters:
    #         exposure -= shelter.sky_blocked()
    #     return exposure
