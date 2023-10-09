import json
from dataclasses import dataclass, field
from enum import Enum
from types import FunctionType
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from honeybee.face import Face
from ladybug.epw import EPW
from ladybug.sunpath import Sun
from ladybug.viewsphere import ViewSphere
from ladybug_geometry.geometry3d import (
    Face3D,
    LineSegment3D,
    Plane,
    Point3D,
    Ray3D,
    Vector3D,
)
from mpl_toolkits import mplot3d
from tqdm import tqdm

from ..bhom import decorator_factory, keys_to_pascalcase, keys_to_snakecase
from ..helpers import wind_speed_at_height
from ..ladybug_extension.epw import sun_position_list, unique_wind_speed_direction


# pylint: disable=too-few-public-methods
class ShelterDecoder(json.JSONDecoder):
    def default(self, o):
        match o:
            case o if isinstance(o, dict):
                if "type" in o:
                    if o["type"] == "Point3D":
                        return Point3D.from_dict(o)
                    raise ValueError(f"Unknown \"type\": {o['type']}")
            case _:
                return super().default(o)


class ShelterEncoder(json.JSONEncoder):
    def default(self, o):
        match o:
            case o if isinstance(o, (Point3D)):
                return o.to_dict()
            case _:
                return super().default(o)


# pylint: enable=too-few-public-methods


@dataclass(init=True, repr=True, eq=True)
class Shelter:

    """A Shelter object, used to determine exposure to sun, sky and wind.

    Args:
        vertices (List[Point3D]):
            A list of coplanar ladybug Point3D objects representing the
            vertices of the shelter object.
        wind_porosity (tuple[float], optional):
            The transmissivity of the shelter to wind. Defaults to 0.
        radiation_porosity (tuple[float], optional):
            The transmissivity of the shelter to radiation (from surfaces, sky
            and sun). Defaults to 0.

    Returns:
        Shelter: A Shelter object.
    """

    vertices: tuple[Point3D] = field(
        init=True, repr=False
    )  # a list of xyz coordinates representing the vertices of the sheltering object
    wind_porosity: float | tuple[float] = field(
        init=True, repr=True, compare=True, default=None
    )
    radiation_porosity: float | tuple[float] = field(
        init=True, repr=True, compare=True, default=None
    )

    _t: str = field(
        init=False, repr=False, compare=True, default="BH.oM.LadybugTools.Shelter"
    )

    def __post_init__(self) -> None:
        if self.vertices is None:
            raise ValueError("vertices cannot be None")

        if len(self.vertices) < 3:
            raise ValueError("vertices must contain at least 3 points")

        _plane = Plane.from_three_points(*self.vertices[:3])
        for vertex in self.vertices[3:]:
            if not np.isclose(a=_plane.distance_to_point(point=vertex), b=0):
                raise ValueError(
                    "All vertices must be coplanar. Please check your input."
                )

        # validate wind porosity
        if self.wind_porosity is None:
            self.wind_porosity = np.zeros(8760)
        elif isinstance(self.wind_porosity, (float, int)):
            self.wind_porosity = np.full(8760, self.wind_porosity)
        if len(self.wind_porosity) != 8760:
            raise ValueError(
                "wind_porosity must be a float or a list/tuple of 8760 values"
            )

        self.wind_porosity = np.atleast_1d(self.wind_porosity)

        if sum(self.wind_porosity > 1) + sum(self.wind_porosity < 0) != 0:
            raise ValueError("wind_porosity values must be between 0 and 1")

        # validate radiation porosity
        if self.radiation_porosity is None:
            self.radiation_porosity = np.zeros(8760)
        elif isinstance(self.radiation_porosity, (float, int)):
            self.radiation_porosity = np.full(8760, self.radiation_porosity)
        if len(self.radiation_porosity) != 8760:
            raise ValueError(
                "radiation_porosity must be a float or a list/tuple of 8760 values"
            )

        self.radiation_porosity = np.atleast_1d(self.radiation_porosity)

        if sum(self.radiation_porosity > 1) + sum(self.radiation_porosity < 0) != 0:
            raise ValueError("radiation_porosity values must be between 0 and 1")

        # check that the shelter has some effect
        if all(
            [self.wind_porosity.sum() == 8760, self.radiation_porosity.sum() == 8760]
        ):
            raise ValueError(
                "This shelter would have no effect as it does not impact wind or radiation exposure."
            )

    def __repr__(self) -> str:
        return (
            f"Shelter(vertices={len(self.vertices)}, "
            f"wind_porosity={self.wind_porosity.mean()}(avg), "
            f"radiation_porosity={self.radiation_porosity.mean()}(avg))"
        )

    @classmethod
    @decorator_factory(disable=False)
    def from_overhead_linear(
        cls,
        width: float = 3,
        height_above_ground: float = 3.5,
        length: float = 2000,
        wind_porosity: float = 0,
        radiation_porosity: float = 0,
        angle: float = 0,
    ) -> "Shelter":
        """Create a linear shelter object oriented north-south, which is
            effectively infinite in its length.

        Args:
            width (float, optional):
                The width of the shelter in m. Defaults to 3m.
            height_above_ground (float, optional):
                The height of the shelter in m. Defaults to 3.5m.
            wind_porosity (float, optional):
                The transmissivity of the shelter to wind. Defaults to 0.
            radiation_porosity (float, optional):
                The transmissivity of the shelter to radiation from sun.
                Defaults to 0.
            angle (float, optional):
                The angle of the shelter in degrees clockwise from north at 0.
                A value of 0 is north-south. A value of 45 is
                northeast-southwest. A value of 90 is east-west.
            length (float, optional):
                The length of the shelter in m. Defaults to 2000m (centered
                about origin, so 1000m north and 1000m south).

        Returns:
            Shelter: A Shelter object.
        """
        origin = Point3D()
        angle = np.deg2rad(-angle)
        return cls(
            vertices=(
                Point3D(width / 2, -length / 2, height_above_ground).rotate_xy(
                    angle, origin
                ),
                Point3D(width / 2, length / 2, height_above_ground).rotate_xy(
                    angle, origin
                ),
                Point3D(-width / 2, length / 2, height_above_ground).rotate_xy(
                    angle, origin
                ),
                Point3D(-width / 2, -length / 2, height_above_ground).rotate_xy(
                    angle, origin
                ),
            ),
            wind_porosity=wind_porosity,
            radiation_porosity=radiation_porosity,
        )

    @classmethod
    @decorator_factory(disable=False)
    def from_overhead_circle(
        cls,
        radius: float = 1.5,
        height_above_ground: float = 3.5,
        wind_porosity: float = 0,
        radiation_porosity: float = 0,
    ) -> "Shelter":
        """Create a circular overhead shelter object.

        Args:
            radius (float, optional):
                The radius of the shelter in m. Defaults to 1.5m.
            height_above_ground (float, optional):
                The height of the shelter in m. Defaults to 3.5m.
            wind_porosity (float, optional):
                The transmissivity of the shelter to wind. Defaults to 0.
            radiation_porosity (float, optional):
                The transmissivity of the shelter to radiation from sun.
                Defaults to 0.

        Returns:
            Shelter: A Shelter object.
        """
        return cls(
            vertices=Face3D.from_regular_polygon(
                side_count=36,
                radius=radius,
                base_plane=Plane(o=Point3D(z=height_above_ground)),
            ).vertices,
            wind_porosity=wind_porosity,
            radiation_porosity=radiation_porosity,
        )

    @classmethod
    @decorator_factory(disable=False)
    def from_adjacent_wall(
        cls,
        distance_from_wall: float = 2,
        wall_height: float = 2,
        wall_length: float = 3,
        wind_porosity: float = 0,
        radiation_porosity: float = 0,
        angle: float = 0,
    ) -> "Shelter":
        """Create a shelter object representative of an adjacent wall.

        Args:
            distance_from_wall (float, optional):
                The distance from the wall in m. Defaults to 2m.
            wall_height (float, optional):
                The height of the wall in m. Defaults to 2m.
            wall_length (float, optional):
                The length of the wall in m. Defaults to 3m.
            wind_porosity (float, optional):
                The transmissivity of the shelter to wind. Defaults to 0.
            radiation_porosity (float, optional):
                The transmissivity of the shelter to radiation from sun.
                Defaults to 0.
            angle (float, optional):
                The angle of the shelter in degrees clockwise from north at 0.
                A value of 0 is directly north. A value of 45 is
                northeast and a value of 90 is to the east.

        Returns:
            Shelter: A Shelter object.
        """
        origin = Point3D()
        angle = np.deg2rad(-angle)
        base_line = LineSegment3D(
            p=origin.move(Vector3D(x=-wall_length / 2, y=distance_from_wall)),
            v=Vector3D(x=wall_length),
        )

        return cls(
            vertices=Face3D.from_extrusion(
                extrusion_vector=Vector3D(0, 0, wall_height), line_segment=base_line
            )
            .rotate_xy(origin=origin, angle=angle)
            .vertices,
            wind_porosity=wind_porosity,
            radiation_porosity=radiation_porosity,
        )

    @classmethod
    @decorator_factory(disable=False)
    def from_lb_face3d(
        cls, face: Face3D, wind_porosity: float = 0, radiation_porosity: float = 0
    ) -> "Shelter":
        """Create a shelter object from a Ladybug Face3D object."""
        return cls(
            vertices=face.vertices,
            wind_porosity=wind_porosity,
            radiation_porosity=radiation_porosity,
        )

    @classmethod
    @decorator_factory(disable=False)
    def from_hb_face(cls, face: Face) -> "Shelter":
        """Create a shelter object from a Honeybee Face object."""
        vertices = face.vertices
        wind_porosity = 0
        radiation_porosity = (
            0
            if face.properties.radiance.modifier.is_opaque
            else face.properties.radiance.modifier.average_transmittance
        )
        return cls(
            Vertices=vertices,
            WindPorosity=wind_porosity,
            RadiationPorosity=radiation_porosity,
        )

    @property
    def origin(self) -> Point3D:
        """Create the origin point, representing an analytical person-point."""
        return Point3D(0, 0, 1.2)

    @property
    def face(self) -> Face3D:
        """Create the face of the shelter object."""
        return Face3D(self.vertices)

    @decorator_factory(disable=False)
    def to_dict(self) -> dict[str, Any]:
        """Create a dictionary representation of this object.

        Returns:
            dict: A dictionary representation of this object.
        """

        obj_dict = {
            "vertices": [i.to_dict() for i in self.vertices],
            "wind_porosity": self.wind_porosity.tolist(),
            "radiation_porosity": self.radiation_porosity.tolist(),
            "_t": self._t,
        }

        return keys_to_pascalcase(obj_dict)

    @classmethod
    @decorator_factory(disable=False)
    def from_dict(cls, dictionary: dict[str, Any]):
        """Create this object from its dictionary representation.

        Args:
            dictionary (dict): A dictionary representation of this object.

        Returns:
            obj: This object.
        """

        dictionary.pop("_t", None)
        dictionary = keys_to_snakecase(dictionary)

        dictionary["vertices"] = [Point3D.from_dict(i) for i in dictionary["vertices"]]

        return cls(**dictionary)

    @decorator_factory(disable=False)
    def to_json(self, **kwargs) -> str:
        """Create a JSON representation of this object.

        Keyword Args:
            kwargs: Additional keyword arguments to pass to json.dumps.

        Returns:
            str: A JSON representation of this object.
        """

        return json.dumps(self.to_dict(), cls=ShelterEncoder, **kwargs)

    @classmethod
    @decorator_factory(disable=False)
    def from_json(cls, json_string: str) -> "Shelter":
        """Create this object from its JSON representation.

        Args:
            json_string (str): A JSON representation of this object.

        Returns:
            obj: This object.
        """

        return cls.from_dict(json.loads(json_string, cls=ShelterDecoder))

    @decorator_factory(disable=False)
    def rotate(self, angle: float, center: Point3D = Point3D()) -> "Shelter":
        """Rotate the shelter about 0, 0, 0 by the given angle in degrees clockwise from north at 0.

        Args:
            angle (float):
                An angle in degrees.
            center (Point3D, optional):
                A Point3D representing the center of rotation. Defaults to 0, 0, 0.

        Returns:
            Shelter:
                The rotated shelter object.
        """
        angle = np.deg2rad(-angle)
        return Shelter(
            self.face.rotate_xy(angle, center).vertices,
            self.wind_porosity,
            self.radiation_porosity,
        )

    @decorator_factory(disable=False)
    def move(self, vector: Vector3D) -> "Shelter":
        """Move the shelter by the given vector.

        Args:
            vector (Vector3D):
                A Vector3D representing the movement.

        Returns:
            Shelter:
                The moved shelter object.
        """
        return Shelter(
            self.face.move(vector).vertices, self.wind_porosity, self.radiation_porosity
        )

    @decorator_factory(disable=False)
    def sky_exposure(self, include_radiation_porosity: bool = True) -> float:
        """Determine the proportion of sky the analytical point is exposed to.
            Also account for radiation_porosity in that exposure.

        Args:
            include_radiation_porosity (bool, optional):
                If True, then increase exposure according to shelter porosity. Defaults to True.

        Returns:
            float:
                A value between 0 and 1 denoting the proportion of sky exposure.
        """
        view_sphere = ViewSphere()
        rays = [
            Ray3D(self.origin, vector) for vector in view_sphere.reinhart_dome_vectors
        ]
        n_intersections = sum(bool(self.face.intersect_line_ray(ray)) for ray in rays)
        if include_radiation_porosity:
            return 1 - ((n_intersections / len(rays)) * (1 - self.radiation_porosity))
        return 1 - (n_intersections / len(rays))

    @decorator_factory(disable=False)
    def annual_sun_exposure(
        self, epw: EPW, include_radiation_porosity: bool = True
    ) -> list[float]:
        """Calculate annual hourly sun exposure. Overnight hours where sun is below horizon default to 0 sun visibility.

        Args:
            epw (EPW):
                A Ladybug EPW object.
            include_radiation_porosity (bool, optional):
                If True, then increase exposure according to shelter porosity.
                Defaults to True.

        Returns:
            List[float]:
                A list of annual hourly values denoting sun exposure values.
        """
        if all(self.radiation_porosity == 1):
            return epw.wind_speed.values

        _sun_exposure = []
        for radiation_porosity, sun in list(
            zip(*[self.radiation_porosity, sun_position_list(epw)])
        ):
            if sun.altitude < 0:
                _sun_exposure.append(0)
                continue

            if radiation_porosity == 1:
                _sun_exposure.append(1)
                continue

            ray = Ray3D(self.origin, (sun.position_3d() - self.origin).normalize())

            if self.face.intersect_line_ray(ray) is None:
                _sun_exposure.append(1)
            else:
                _sun_exposure.append(
                    radiation_porosity if include_radiation_porosity else 0
                )
        return _sun_exposure

    @decorator_factory(disable=False)
    def wind_obstruction_multiplier(
        self,
        wind_direction: float,
        obstruction_band_width: float = 10,
        n_samples_xy: int = 4,
        n_samples_rotational: int = 5,
    ) -> float:
        """Determine the multiplier for wind speed from a given direction based on
            shelter obstruction and edge acceleration effects.

        * When fully obstructed, then the porosity of the shelter is used to reduce wind speed.
        * Where partially obstructed, edge effects are expected based on the level of porosity of the shelter, and the proportion of obstructed/unobstructed area.
        * Where not obstructed, then wind is kept per input.

        Args:
            wind_direction (float):
                A single wind direction between 0-360.
            obstruction_band_width (float):
                The azimuthal range over which obstruction is checked. Defaults to 10 degrees.
            n_samples_xy (int):
                The number of samples to take in the xy plane.
            n_samples_rotational (int):
                The number of samples to take in the rotational plane.

        Returns:
            float:
                A multiplier which describes how much obstruction there is from the shelter to the given wind direction.
        """

        if all(self.wind_porosity == 1):
            return 1

        wind_direction_rad = np.deg2rad(wind_direction)
        wind_direction_vector = Vector3D(
            np.sin(wind_direction_rad), np.cos(wind_direction_rad)
        )  # TODO - check that this is going the right direction

        wind_direction_plane = Plane(n=wind_direction_vector, o=self.origin)

        if not any(wind_direction_plane.is_point_above(i) for i in self.vertices):
            # the shelter is in the opposite direction to the wind, no multiplier needed
            return 1

        # check if shelter is too high to impact wind
        if all(i.z > self.origin.z + 5 for i in self.vertices):
            # the shelter is too high to impact wind, no multiplier needed
            return 1

        if all(i.z < 0 for i in self.vertices):
            # the shelter is too low to impact wind, no multiplier needed
            return 1

        if len(set(i.z for i in self.vertices)) == 1:
            # the shelter is flat, no multiplier needed
            # TODO - if shelter angled, approximate funneling towards/away from ground effect
            return 1

        wind_ray = Ray3D(p=self.origin, v=wind_direction_vector)

        wind_rays = []
        for az_angle in np.deg2rad(
            np.linspace(
                -obstruction_band_width / 2, obstruction_band_width / 2, n_samples_xy
            )
        ):
            wr = wind_ray.rotate_xy(origin=self.origin, angle=az_angle)
            for rotation in np.deg2rad(np.linspace(0, 180, n_samples_rotational)):
                wind_rays.append(
                    wr.rotate(
                        origin=self.origin, angle=rotation, axis=wind_direction_vector
                    )
                )

        # remove duplicate-ish rays
        d = {}
        for wr in wind_rays:
            vv = tuple(round(i, 3) for i in wr.v.to_array())
            if vv in d:
                continue
            d[vv] = wr

        # get the number of intersections with the shelter object
        intersections = [bool(self.face.intersect_line_ray(i)) for i in d.values()]

        return sum(intersections) / len(intersections)

    def annual_wind_speed(
        self,
        epw: EPW,
        edge_acceleration_factor: float = 1.1,
    ) -> list[float]:
        """Calculate annual hourly effective wind speed. Wind from the given
        EPW file will be translated to 1.2m above ground.

        Args:
            epw (EPW):
                A Ladybug EPW object.
            edge_acceleration_factor (float, optional):
                The proportional increase in wind speed due to edge acceleration around a shelter edge. Defaults to 1.1.

        Returns:
            List[float]:
                A list of annual hourly values denoting sun exposure values.
        """

        _, unique_wind_directions = unique_wind_speed_direction(epw).T

        if all(self.wind_porosity == 1):
            return wind_speed_at_height(epw.wind_speed.values, 10, 1.2)

        # calculate only for unique wind speeds and directions to save some time
        d = {}  # a multiplier for each wind direction to adjust wind speed by
        for wd in unique_wind_directions:
            d[wd] = 1 - self.wind_obstruction_multiplier(wd)

        # calculate effective wind speed
        _wind_speed = []
        for wind_porosity, ws, wd in list(
            zip(
                *[
                    self.wind_porosity,
                    wind_speed_at_height(epw.wind_speed, 10, 1.2),
                    epw.wind_direction,
                ]
            )
        ):
            if d[wd] == 0:
                # obstructed, wind modified by shelter porosity
                _wind_speed.append(ws * wind_porosity)
            elif d[wd] == 1:
                # unobstructed, wind kept per input
                _wind_speed.append(ws)
            else:
                # partially obstructed, wind modified by shelter porosity and edge acceleration
                _wind_speed.append(
                    (ws * wind_porosity * d[wd])
                    + (ws * edge_acceleration_factor * (1 - d[wd]))
                )

        return _wind_speed

    # def _effective_wind_speed(
    #     self,
    #     wind_direction: float,
    #     wind_speed: float,
    #     edge_acceleration_factor: float = 1.1,
    #     obstruction_band_width: float = 5,
    # ) -> float:
    #     """Determine the effective wind speed from a given direction based on
    #         shelter obstruction and edge acceleration effects. This method does not account for

    #     Wind is considered "obstructured" in some way if its direction +/-5 degrees of the shelter.
    #     When fully obstructed, then the porosity of the shelter is used to reduce wind speed.
    #     Where partially obstructed, the edge effects are expected based on the level of porosity
    #     of the shelter.
    #     Where not obstructed, then wind is kept per input.

    #     Args:
    #         wind_direction (float):
    #             A single wind direction between 0-360.
    #         wind_speed (float):
    #             A wind speed in m/s.
    #         edge_acceleration_factor (float, optional):
    #             The proportional increase in wind speed due to edge acceleration around a shelter edge. Defaults to 1.1.
    #         obstruction_band_width (float, optional):
    #             The azimuthal range over which obstruction is checked. Defaults to 5.

    #     Returns:
    #         float:
    #             A resultant wind speed subject to obstruction from the shelter.
    #     """

    #     if wind_speed == 0:
    #         return 0

    #     if self.WindPorosity == 1:
    #         return wind_speed

    #     # create components for sample vectors
    #     left_rad = np.deg2rad((wind_direction - obstruction_band_width) % 360)
    #     left_x = np.sin(left_rad)
    #     left_y = np.cos(left_rad)
    #     center_rad = np.deg2rad(wind_direction)
    #     center_x = np.sin(center_rad)
    #     center_y = np.cos(center_rad)
    #     right_rad = np.deg2rad((wind_direction + obstruction_band_width) % 360)
    #     right_x = np.sin(right_rad)
    #     right_y = np.cos(right_rad)

    #     # create sample vectors
    #     sample_vectors = []
    #     for _z in np.linspace(0, 1, 5):
    #         sample_vectors.extend(
    #             [
    #                 Vector3D(left_x, left_y, _z),
    #                 Vector3D(center_x, center_y, _z),
    #                 Vector3D(right_x, right_y, _z),
    #             ]
    #         )

    #     # check intersections
    #     rays = [Ray3D(self.origin, vector) for vector in sample_vectors]
    #     n_intersections = sum(bool(self.face.intersect_line_ray(ray)) for ray in rays)

    #     # return effective wind speed
    #     if n_intersections == len(rays):
    #         # fully obstructed
    #         return wind_speed * self.WindPorosity
    #     if n_intersections == 0:
    #         # fully unobstructed
    #         return wind_speed
    #     # get the propostion of obstruction adn adjust acceleration based on that proportion
    #     proportion_obstructed = n_intersections / len(rays)
    #     return (
    #         (wind_speed * self.WindPorosity)
    #         * proportion_obstructed  # obstructed wind component
    #     ) + (
    #         wind_speed
    #         * edge_acceleration_factor
    #         * (1 - proportion_obstructed)  # edge wind component
    #     )


#     def annual_effective_wind_speed(
#         self,
#         epw: EPW,
#         edge_acceleration_factor: float = 1.1,
#         obstruction_band_width: float = 5,
#         height_above_ground: float = 10,
#         reference_height: float = 10,
#         terrain_roughness_length: float = 0.03,
#     ) -> List[float]:
#         """Determine the effective wind speed from a given direction based on shelter obstruction and edge acceleration effects.

#         Wind is considered "obstructured" in some way if its direction +/-5 degrees of the shelter.
#         When fully obstructed, then the porosity of the shelter is used to reduce wind speed.
#         Where partially obstructed, the edge effects are expected based on the level of porosity
#         of the shelter.
#         Where not obstructed, then wind is kept per input.

#         Args:
#             epw (EPW):
#                 A Ladybug EPW object.
#             edge_acceleration_factor (float, optional):
#                 The proportional increase in wind speed due to edge acceleration around a shelter edge. Defaults to 1.1.
#             obstruction_band_width (float, optional):
#                 The azimuthal range over which obstruction is checked. Defaults to 5.
#             height_above_ground (float, optional):
#                 The height above ground in meters at which the wind speed is measured. Defaults to 10 which is typical for an EPW.
#             reference_height (float, optional):
#                 The height above ground in meters at which the reference wind speed is measured. Defaults to 10 which is typical for an EPW.
#             terrain_roughness_length (float, optional):
#                 The terrain roughness length in meters. Defaults to 0.03 which is typical open flat terrain with a few isolated obstacles.

#         Returns:
#             List[float]:
#                 A resultant list of EPW aligned wind speeds subject to obstruction from the shelter.
#         """
#         _ws_wd = unique_wind_speed_direction(epw)

#         # adjust for wind speed at height
#         ws_wd = []
#         for ws, wd in _ws_wd:
#             ws_wd.append(
#                 (
#                     wind_speed_at_height(
#                         reference_value=ws,
#                         reference_height=reference_height,
#                         target_height=height_above_ground,
#                         terrain_roughness_length=terrain_roughness_length,
#                         log_function=True,
#                     ),
#                     wd,
#                 )
#             )

#         ws_effective = {}
#         for ws, wd in ws_wd:
#             ws_effective[(ws, wd)] = self.effective_wind_speed(
#                 wd, ws, edge_acceleration_factor, obstruction_band_width
#             )
#         # cast back to original epw
#         effective_wind_speeds = []
#         for ws, wd in list(zip(*[epw.wind_speed.values, epw.wind_direction.values])):
#             effective_wind_speeds.append(ws_effective[(ws, wd)])
#         return effective_wind_speeds

#     def set_porosity(self, porosity: float) -> Shelter:
#         """Return this shelter with an adjusted porosity value applied to both wind and radiation components."""
#         return Shelter(self.Vertices, porosity, porosity)

#     def visualise(self) -> plt.Figure:
#         """Visualise this shelter to check validity and that it exists where you think it should!"""

#         fig = plt.figure(figsize=(5, 5))
#         ax = mplot3d.Axes3D(fig, auto_add_to_figure=False)
#         fig.add_axes(ax)
#         ax.scatter(*self.origin.to_array())
#         # add shelter as a polygon
#         vtx = np.array([i.to_array() for i in self.face.vertices])
#         tri = mplot3d.art3d.Poly3DCollection([vtx])
#         tri.set_color("grey")
#         tri.set_alpha(0.5)
#         tri.set_edgecolor("k")
#         ax.add_collection3d(tri)
#         # format axes
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")
#         ax.set_zlabel("z")
#         # set lims
#         ax.set_xlim(min(i[0] for i in vtx), max(i[0] for i in vtx))
#         ax.set_ylim(min(i[1] for i in vtx), max(i[1] for i in vtx))
#         # pylint: disable=no-member
#         ax.set_zlim(min(i[2] for i in vtx), max(i[2] for i in vtx))
#         # pylint: enable=no-member
#         return fig


# def sky_exposure(
#     shelters: List[Shelter], include_radiation_porosity: bool = True
# ) -> float:
#     """Determine the proportion of sky the analytical point is exposed to under a combination of shelters. Also account for radiation_porosity in that exposure.

#     Args:
#         shelters (List[Shelter]):
#             A list of shelter objects.
#         include_radiation_porosity (bool, optional):
#             If True, then increase exposure according to shelter porosity. Defaults to True.

#     Returns:
#         float:
#             A value between 0 and 1 denoting the proportion of sky exposure.
#     """
#     if len(shelters) == 0:
#         return 1

#     view_sphere = ViewSphere()
#     rays = [
#         Ray3D(shelters[0].origin, vector)
#         for vector in view_sphere.reinhart_dome_vectors
#     ]

#     # get intersections for each patch, for each shelter
#     intersections = []
#     for shelter in shelters:
#         intersections.append(
#             [
#                 1
#                 if shelter.face.intersect_line_ray(ray) is None
#                 else (shelter.RadiationPorosity if include_radiation_porosity else 0)
#                 for ray in rays
#             ]
#         )
#     return np.array(intersections).prod(axis=0).sum() / len(rays)


# def sun_exposure(
#     shelters: List[Shelter], sun: Sun, include_radiation_porosity: bool = True
# ) -> float:
#     """Determine the proportion of sun the analytical point is exposed to under a combination of shelters. Also account for radiation_porosity in that exposure.

#     Args:
#         shelters (List[Shelter]):
#             A list of shelter objects.
#         sun (Sun):
#             A LB Sun object.
#         include_radiation_porosity (bool, optional):
#             If True, then increase exposure according to shelter porosity. Defaults to True.

#     Returns:
#         float:
#             A value between 0 and 1 denoting the overall proportion of sky exposure.
#     """
#     if len(shelters) == 0:
#         return 1 if sun.altitude > 0 else 0

#     sun_exposures = []
#     for shelter in shelters:
#         sun_exposures.append(shelter.sun_exposure(sun, include_radiation_porosity))
#     return np.prod(sun_exposures)


# def annual_sun_exposure(
#     shelters: List[Shelter], epw: EPW, include_radiation_porosity: bool = True
# ) -> List[float]:
#     """Calculate annual hourly sun exposure under a set of shelters. Where sun is below horizon default to 0 sun visibility.

#     Args:
#         shelters (List[Shelter]):
#             A list of shelter objects.
#         epw (EPW):
#             A Ladybug EPW object.
#         include_radiation_porosity (bool, optional):
#             If True, then increase exposure according to shelter porosity. Defaults to True.

#     Returns:
#         List[float]:
#             A list of annual hourly values denoting sun exposure values.
#     """

#     suns = sun_position_list(epw)
#     if len(shelters) == 0:
#         return [1 if sun.altitude > 0 else 0 for sun in suns]

#     sun_exposures = []
#     for shelter in shelters:
#         sun_exposures.append(
#             [shelter.sun_exposure(sun, include_radiation_porosity) for sun in suns]
#         )
#     return np.prod(sun_exposures, axis=0)


# def effective_wind_speed(
#     shelters: List[Shelter],
#     wind_speed: float,
#     wind_direction: float,
#     edge_acceleration_factor: float = 1.1,
#     obstruction_band_width: float = 5,
# ) -> float:
#     """Determine the effective wind speed from a given direction based on multiple obstructing shelters and edge acceleration effects.

#     Wind is considered "obstructured" in some way if its direction +/-5 degrees of the shelter.
#     When fully obstructed, then the porosity of the shelter is used to reduce wind speed.
#     Where partially obstructed, the edge effects are expected based on the level of porosity
#     of the shelter.
#     Where not obstructed, then wind is kept per input.

#     Args:
#         shelters (List[Shelter]):
#             A list of shelter objects.
#         wind_direction (float):
#             A single wind direction between 0-360.
#         wind_speed (float):
#             A wind speed in m/s.
#         edge_acceleration_factor (float, optional):
#             The proportional increase in wind speed due to edge acceleration around a shelter edge. Defaults to 1.1.
#         obstruction_band_width (float, optional):
#             The azimuthal range over which obstruction is checked. Defaults to 5.

#     Returns:
#         float:
#             A resultant wind speed subject to obstruction from the shelter.
#     """

#     if len(shelters) == 0:
#         return wind_speed

#     effective_wind_speeds = []
#     for shelter in shelters:
#         effective_wind_speeds.append(
#             shelter.effective_wind_speed(
#                 wind_direction,
#                 wind_speed,
#                 edge_acceleration_factor,
#                 obstruction_band_width,
#             )
#         )
#     return np.min(effective_wind_speeds)


# def annual_effective_wind_speed(
#     shelters: List[Shelter],
#     epw: EPW,
#     edge_acceleration_factor: float = 1.1,
#     obstruction_band_width: float = 5,
# ) -> List[float]:
#     """Determine the effective wind speed from a given direction based on multiple obstructing shelters and edge acceleration effects.

#     Wind is considered "obstructured" in some way if its direction +/-5 degrees of the shelter.
#     When fully obstructed, then the porosity of the shelter is used to reduce wind speed.
#     Where partially obstructed, the edge effects are expected based on the level of porosity
#     of the shelter.
#     Where not obstructed, then wind is kept per input.

#     Args:
#         shelters (List[Shelter]):
#             A list of shelter objects.
#         epw (EPW):
#             A Ladybug EPW object.
#         edge_acceleration_factor (float, optional):
#             The proportional increase in wind speed due to edge acceleration around a shelter edge. Defaults to 1.1.
#         obstruction_band_width (float, optional):
#             The azimuthal range over which obstruction is checked. Defaults to 5.

#     Returns:
#         List[float]:
#             A resultant list of EPW aligned wind speeds subject to obstruction from the shelter.
#     """
#     if len(shelters) == 0:
#         return epw.wind_speed

#     ws_wd = unique_wind_speed_direction(epw)
#     all_effective_wind_speeds = []
#     for shelter in shelters:
#         ws_effective = {}
#         for ws, wd in ws_wd:
#             ws_effective[(ws, wd)] = shelter.effective_wind_speed(
#                 wd, ws, edge_acceleration_factor, obstruction_band_width
#             )
#         # cast back to original epw
#         effective_wind_speeds = []
#         for ws, wd in list(zip(*[epw.wind_speed.values, epw.wind_direction.values])):
#             effective_wind_speeds.append(ws_effective[(ws, wd)])
#         all_effective_wind_speeds.append(effective_wind_speeds)
#     return np.min(all_effective_wind_speeds, axis=0)


# class Shelters(Enum):
#     """A list of pre-defined Shelter forms."""

#     NORTH_SOUTH_LINEAR = Shelter(
#         Vertices=_LINEAR_SHELTER_VERTICES_NORTH_SOUTH,
#     )
#     EAST_WEST_LINEAR = Shelter(
#         Vertices=_LINEAR_SHELTER_VERTICES_NORTH_SOUTH,
#     ).rotate(90)
#     NORTHEAST_SOUTHWEST_LINEAR = Shelter(
#         Vertices=_LINEAR_SHELTER_VERTICES_NORTH_SOUTH,
#     ).rotate(45)
#     NORTHWEST_SOUTHEAST_LINEAR = Shelter(
#         Vertices=_LINEAR_SHELTER_VERTICES_NORTH_SOUTH,
#     ).rotate(135)

#     OVERHEAD_SMALL = Shelter(
#         Vertices=_OVERHEAD_SHELTER_VERTICES_SMALL,
#     )
#     OVERHEAD_LARGE = Shelter(
#         Vertices=_OVERHEAD_SHELTER_VERTICES_LARGE,
#     )
#     CANOPY_N_E_S_W = Shelter(Vertices=_CANOPY_NORTH)
#     CANOPY_NE_SE_SW_NW = Shelter(Vertices=_CANOPY_NORTH).rotate(45)

#     NORTH = Shelter(
#         Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
#     )
#     NORTHEAST = Shelter(
#         Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
#     ).rotate(45)
#     EAST = Shelter(
#         Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
#     ).rotate(90)
#     SOUTHEAST = Shelter(
#         Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
#     ).rotate(135)
#     SOUTH = Shelter(
#         Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
#     ).rotate(180)
#     SOUTHWEST = Shelter(
#         Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
#     ).rotate(225)
#     WEST = Shelter(
#         Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
#     ).rotate(270)
#     NORTHWEST = Shelter(
#         Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
#     ).rotate(315)
