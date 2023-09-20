from __future__ import annotations

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
from ladybug_geometry.geometry3d import Face3D, Point3D, Ray3D, Vector3D
from mpl_toolkits import mplot3d

from ..bhomutil.bhom_object import BHoMObject
from ..bhomutil.encoder import BHoMEncoder
from ..helpers import wind_speed_at_height
from ..ladybug_extension.epw import sun_position_list, unique_wind_speed_direction

_LINEAR_SHELTER_VERTICES_NORTH_SOUTH = [
    Point3D(2, -1000, 3.5),
    Point3D(2, 1000, 3.5),
    Point3D(-2, 1000, 3.5),
    Point3D(-2, -1000, 3.5),
]
_OVERHEAD_SHELTER_VERTICES_SMALL = [
    Point3D(2, 0, 3.5),
    Point3D(1.879385, 0.68404, 3.5),
    Point3D(1.532089, 1.285575, 3.5),
    Point3D(1, 1.732051, 3.5),
    Point3D(0.347296, 1.969616, 3.5),
    Point3D(-0.347296, 1.969616, 3.5),
    Point3D(-1.0, 1.732051, 3.5),
    Point3D(-1.532089, 1.285575, 3.5),
    Point3D(-1.879385, 0.68404, 3.5),
    Point3D(-2, 0, 3.5),
    Point3D(-1.879385, -0.68404, 3.5),
    Point3D(-1.532089, -1.285575, 3.5),
    Point3D(-1, -1.732051, 3.5),
    Point3D(-0.347296, -1.969616, 3.5),
    Point3D(0.347296, -1.969616, 3.5),
    Point3D(1.0, -1.732051, 3.5),
    Point3D(1.532089, -1.285575, 3.5),
    Point3D(1.879385, -0.68404, 3.5),
]
_OVERHEAD_SHELTER_VERTICES_LARGE = [
    Point3D(250, 0, 3.5),
    Point3D(234.923155, 85.505036, 3.5),
    Point3D(191.511111, 160.696902, 3.5),
    Point3D(125, 216.506351, 3.5),
    Point3D(43.412044, 246.201938, 3.5),
    Point3D(-43.412044, 246.201938, 3.5),
    Point3D(-125.0, 216.506351, 3.5),
    Point3D(-191.511111, 160.696902, 3.5),
    Point3D(-234.923155, 85.505036, 3.5),
    Point3D(-250, 0, 3.5),
    Point3D(-234.923155, -85.505036, 3.5),
    Point3D(-191.511111, -160.696902, 3.5),
    Point3D(-125, -216.506351, 3.5),
    Point3D(-43.412044, -246.201938, 3.5),
    Point3D(43.412044, -246.201938, 3.5),
    Point3D(125.0, -216.506351, 3.5),
    Point3D(191.511111, -160.696902, 3.5),
    Point3D(234.923155, -85.505036, 3.5),
]
_DIRECTIONAL_SHELTER_VERTICES_NORTH = [
    Point3D(1, 1, 0),
    Point3D(-1, 1, 0),
    Point3D(-1, 1, 5),
    Point3D(1, 1, 5),
]
_CANOPY_NORTH = [
    Point3D(1, 1, 5),
    Point3D(-1, 1, 5),
    Point3D(-1, -1, 5),
    Point3D(1, -1, 5),
]


@dataclass(init=True, repr=True, eq=True)
class Shelter(BHoMObject):

    """A Shelter object, used to determine exposure to sun, sky and wind.

    Args:
        vertices (List[Point3D]):
            A list of coplanar ladybug Point3D objects representing the
            vertices of the shelter object.
        wind_porosity (float, optional):
            The transmissivity of the shelter to wind. Defaults to 0.
        radiation_porosity (float, optional):
            The transmissivity of the shelter to radiation (from surfaces, sky
            and sun). Defaults to 0.

    Returns:
        Shelter: A Shelter object.
    """

    Vertices: List[Point3D] = field(
        init=True, repr=False
    )  # a list of xyz coordinates representing the vertices of the sheltering object
    WindPorosity: float = field(init=True, repr=True, compare=True, default=0)
    RadiationPorosity: float = field(init=True, repr=True, compare=True, default=0)

    _t: str = field(
        init=False, repr=False, compare=True, default="BH.oM.LadybugTools.Shelter"
    )

    def __post_init__(self) -> None:
        # handle nulls
        if self.Vertices is None:
            raise ValueError("vertices cannot be None")
        if self.WindPorosity is None:
            self.WindPorosity = 0
        if self.RadiationPorosity is None:
            self.RadiationPorosity = 0

        if (not 0 <= self.WindPorosity <= 1) or (not 0 <= self.RadiationPorosity <= 1):
            raise ValueError("porosity values must be between 0 and 1")

        if all([self.WindPorosity == 1, self.RadiationPorosity == 1]):
            raise ValueError(
                "This shelter would have no effect as it does not impact wind or radiation exposure."
            )

        # create face for shelter
        self.face = Face3D(self.Vertices)
        self.face.check_planar(0.001)

        # create origin pt - representing an analytical person-point
        self.origin = Point3D(0, 0, 1.2)

        # wrap methods within this class
        super().__post_init__()

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> Shelter:
        """Create this object from a dictionary."""

        if all(isinstance(i, Point3D) for i in dictionary["Vertices"]):
            # from a standard dictionary, no special treatment required

            return cls(
                WindPorosity=dictionary["WindPorosity"],
                RadiationPorosity=dictionary["RadiationPorosity"],
                Vertices=dictionary["Vertices"],
            )

        vertices = [Point3D(i["X"], i["Y"], i["Z"]) for i in dictionary["Vertices"]]

        return cls(
            WindPorosity=dictionary["WindPorosity"],
            RadiationPorosity=dictionary["RadiationPorosity"],
            Vertices=vertices,
        )

    @classmethod
    def from_json(cls, json_string: str) -> Shelter:
        """Create this object from a JSON string."""

        dictionary = json.loads(json_string)

        return cls.from_dict(dictionary)

    @classmethod
    def from_lb_face3d(
        cls, face: Face3D, wind_porosity: float = 0, radiation_porosity: float = 0
    ) -> Shelter:
        """Create a shelter object from a Ladybug Face3D object."""
        return cls(
            vertices=face.vertices,
            wind_porosity=wind_porosity,
            radiation_porosity=radiation_porosity,
        )

    @classmethod
    def from_hb_face(cls, face: Face) -> Shelter:
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

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as it's dictionary equivalent."""
        dictionary = {}
        for k, v in self.__dict__.items():
            if isinstance(getattr(self, k), FunctionType):
                continue
            dictionary[k] = v
        dictionary["_t"] = self._t
        return dictionary

    @property
    def vertices(self) -> List[Point3D]:
        """A handy accessor using proper Python naming conventions."""
        return self.Vertices

    @property
    def wind_porosity(self) -> float:
        """A handy accessor using proper Python naming conventions."""
        return self.WindPorosity

    @property
    def radiation_porosity(self) -> float:
        """A handy accessor using proper Python naming conventions."""
        return self.RadiationPorosity

    def to_json(self) -> str:
        """Return this object as it's JSON string equivalent."""
        return json.dumps(self.to_dict(), cls=BHoMEncoder)

    def rotate(self, angle: float, center: Point3D = Point3D()) -> Shelter:
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
        rotated_vertices = self.face.rotate_xy(angle, center).vertices
        return Shelter(rotated_vertices, self.WindPorosity, self.RadiationPorosity)

    def move(self, vector: Vector3D) -> Shelter:
        """Move the shelter by the given vector.

        Args:
            vector (Vector3D):
                A Vector3D representing the movement.

        Returns:
            Shelter:
                The moved shelter object.
        """
        moved_vertices = self.face.move(vector).vertices
        return Shelter(moved_vertices, self.WindPorosity, self.RadiationPorosity)

    def sky_exposure(self, include_radiation_porosity: bool = True) -> float:
        """Determine the proportion of sky the analytical point is exposed to. Also account for radiation_porosity in that exposure.

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
            return 1 - ((n_intersections / len(rays)) * (1 - self.RadiationPorosity))
        return 1 - (n_intersections / len(rays))

    def sun_exposure(self, sun: Sun, include_radiation_porosity: bool = True) -> float:
        """Determine the proportion of sun the analytical point is exposed to. Also account for radiation_porosity in that exposure.

        Args:
            sun (Sun):
                A LB Sun object.
            include_radiation_porosity (bool, optional):
                If True, then increase exposure according to shelter porosity. Defaults to True.

        Returns:
            float:
                A value between 0 and 1 denoting the proportion of sun exposure.
        """
        if sun.altitude < 0:
            return 0
        ray = Ray3D(self.origin, (sun.position_3d() - self.origin).normalize())
        if self.face.intersect_line_ray(ray) is None:
            return 1
        return self.RadiationPorosity if include_radiation_porosity else 0

    def annual_sun_exposure(
        self, epw: EPW, include_radiation_porosity: bool = True
    ) -> List[float]:
        """Calculate annual hourly sun exposure. Overnight hours where sun is below horizon default to 0 sun visibility.

        Args:
            epw (EPW):
                A Ladybug EPW object.
            include_radiation_porosity (bool, optional):
                If True, then increase exposure according to shelter porosity. Defaults to True.

        Returns:
            List[float]:
                A list of annual hourly values denoting sun exposure values.
        """
        suns = sun_position_list(epw)
        return [self.sun_exposure(sun, include_radiation_porosity) for sun in suns]

    def effective_wind_speed(
        self,
        wind_direction: float,
        wind_speed: float,
        edge_acceleration_factor: float = 1.1,
        obstruction_band_width: float = 5,
    ) -> float:
        """Determine the effective wind speed from a given direction based on shelter obstruction and edge acceleration effects.

        Wind is considered "obstructured" in some way if its direction +/-5 degrees of the shelter.
        When fully obstructed, then the porosity of the shelter is used to reduce wind speed.
        Where partially obstructed, the edge effects are expected based on the level of porosity
        of the shelter.
        Where not obstructed, then wind is kept per input.

        Args:
            wind_direction (float):
                A single wind direction between 0-360.
            wind_speed (float):
                A wind speed in m/s.
            edge_acceleration_factor (float, optional):
                The proportional increase in wind speed due to edge acceleration around a shelter edge. Defaults to 1.1.
            obstruction_band_width (float, optional):
                The azimuthal range over which obstruction is checked. Defaults to 5.

        Returns:
            float:
                A resultant wind speed subject to obstruction from the shelter.
        """

        if wind_speed == 0:
            return 0

        if self.WindPorosity == 1:
            return wind_speed

        # create components for sample vectors
        left_rad = np.deg2rad((wind_direction - obstruction_band_width) % 360)
        left_x = np.sin(left_rad)
        left_y = np.cos(left_rad)
        center_rad = np.deg2rad(wind_direction)
        center_x = np.sin(center_rad)
        center_y = np.cos(center_rad)
        right_rad = np.deg2rad((wind_direction + obstruction_band_width) % 360)
        right_x = np.sin(right_rad)
        right_y = np.cos(right_rad)

        # create sample vectors
        sample_vectors = []
        for _z in np.linspace(0, 1, 5):
            sample_vectors.extend(
                [
                    Vector3D(left_x, left_y, _z),
                    Vector3D(center_x, center_y, _z),
                    Vector3D(right_x, right_y, _z),
                ]
            )

        # check intersections
        rays = [Ray3D(self.origin, vector) for vector in sample_vectors]
        n_intersections = sum(bool(self.face.intersect_line_ray(ray)) for ray in rays)

        # return effective wind speed
        if n_intersections == len(rays):
            # fully obstructed
            return wind_speed * self.WindPorosity
        if n_intersections == 0:
            # fully unobstructed
            return wind_speed
        # get the propostion of obstruction adn adjust acceleration based on that proportion
        proportion_obstructed = n_intersections / len(rays)
        return (
            (wind_speed * self.WindPorosity)
            * proportion_obstructed  # obstructed wind component
        ) + (
            wind_speed
            * edge_acceleration_factor
            * (1 - proportion_obstructed)  # edge wind component
        )

    def annual_effective_wind_speed(
        self,
        epw: EPW,
        edge_acceleration_factor: float = 1.1,
        obstruction_band_width: float = 5,
        height_above_ground: float = 10,
        reference_height: float = 10,
        terrain_roughness_length: float = 0.03,
    ) -> List[float]:
        """Determine the effective wind speed from a given direction based on shelter obstruction and edge acceleration effects.

        Wind is considered "obstructured" in some way if its direction +/-5 degrees of the shelter.
        When fully obstructed, then the porosity of the shelter is used to reduce wind speed.
        Where partially obstructed, the edge effects are expected based on the level of porosity
        of the shelter.
        Where not obstructed, then wind is kept per input.

        Args:
            epw (EPW):
                A Ladybug EPW object.
            edge_acceleration_factor (float, optional):
                The proportional increase in wind speed due to edge acceleration around a shelter edge. Defaults to 1.1.
            obstruction_band_width (float, optional):
                The azimuthal range over which obstruction is checked. Defaults to 5.
            height_above_ground (float, optional):
                The height above ground in meters at which the wind speed is measured. Defaults to 10 which is typical for an EPW.
            reference_height (float, optional):
                The height above ground in meters at which the reference wind speed is measured. Defaults to 10 which is typical for an EPW.
            terrain_roughness_length (float, optional):
                The terrain roughness length in meters. Defaults to 0.03 which is typical open flat terrain with a few isolated obstacles.

        Returns:
            List[float]:
                A resultant list of EPW aligned wind speeds subject to obstruction from the shelter.
        """
        _ws_wd = unique_wind_speed_direction(epw)

        # adjust for wind speed at height
        ws_wd = []
        for ws, wd in _ws_wd:
            ws_wd.append(
                (
                    wind_speed_at_height(
                        reference_value=ws,
                        reference_height=reference_height,
                        target_height=height_above_ground,
                        terrain_roughness_length=terrain_roughness_length,
                        log_function=True,
                    ),
                    wd,
                )
            )

        ws_effective = {}
        for ws, wd in ws_wd:
            ws_effective[(ws, wd)] = self.effective_wind_speed(
                wd, ws, edge_acceleration_factor, obstruction_band_width
            )
        # cast back to original epw
        effective_wind_speeds = []
        for ws, wd in list(zip(*[epw.wind_speed.values, epw.wind_direction.values])):
            effective_wind_speeds.append(ws_effective[(ws, wd)])
        return effective_wind_speeds

    def set_porosity(self, porosity: float) -> Shelter:
        """Return this shelter with an adjusted porosity value applied to both wind and radiation components."""
        return Shelter(self.Vertices, porosity, porosity)

    def visualise(self) -> plt.Figure:
        """Visualise this shelter to check validity and that it exists where you think it should!"""

        fig = plt.figure(figsize=(5, 5))
        ax = mplot3d.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.scatter(*self.origin.to_array())
        # add shelter as a polygon
        vtx = np.array([i.to_array() for i in self.face.vertices])
        tri = mplot3d.art3d.Poly3DCollection([vtx])
        tri.set_color("grey")
        tri.set_alpha(0.5)
        tri.set_edgecolor("k")
        ax.add_collection3d(tri)
        # format axes
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # set lims
        ax.set_xlim(min(i[0] for i in vtx), max(i[0] for i in vtx))
        ax.set_ylim(min(i[1] for i in vtx), max(i[1] for i in vtx))
        # pylint: disable=no-member
        ax.set_zlim(min(i[2] for i in vtx), max(i[2] for i in vtx))
        # pylint: enable=no-member
        return fig


def sky_exposure(
    shelters: List[Shelter], include_radiation_porosity: bool = True
) -> float:
    """Determine the proportion of sky the analytical point is exposed to under a combination of shelters. Also account for radiation_porosity in that exposure.

    Args:
        shelters (List[Shelter]):
            A list of shelter objects.
        include_radiation_porosity (bool, optional):
            If True, then increase exposure according to shelter porosity. Defaults to True.

    Returns:
        float:
            A value between 0 and 1 denoting the proportion of sky exposure.
    """
    if len(shelters) == 0:
        return 1

    view_sphere = ViewSphere()
    rays = [
        Ray3D(shelters[0].origin, vector)
        for vector in view_sphere.reinhart_dome_vectors
    ]

    # get intersections for each patch, for each shelter
    intersections = []
    for shelter in shelters:
        intersections.append(
            [
                1
                if shelter.face.intersect_line_ray(ray) is None
                else (shelter.RadiationPorosity if include_radiation_porosity else 0)
                for ray in rays
            ]
        )
    return np.array(intersections).prod(axis=0).sum() / len(rays)


def sun_exposure(
    shelters: List[Shelter], sun: Sun, include_radiation_porosity: bool = True
) -> float:
    """Determine the proportion of sun the analytical point is exposed to under a combination of shelters. Also account for radiation_porosity in that exposure.

    Args:
        shelters (List[Shelter]):
            A list of shelter objects.
        sun (Sun):
            A LB Sun object.
        include_radiation_porosity (bool, optional):
            If True, then increase exposure according to shelter porosity. Defaults to True.

    Returns:
        float:
            A value between 0 and 1 denoting the overall proportion of sky exposure.
    """
    if len(shelters) == 0:
        return 1 if sun.altitude > 0 else 0

    sun_exposures = []
    for shelter in shelters:
        sun_exposures.append(shelter.sun_exposure(sun, include_radiation_porosity))
    return np.prod(sun_exposures)


def annual_sun_exposure(
    shelters: List[Shelter], epw: EPW, include_radiation_porosity: bool = True
) -> List[float]:
    """Calculate annual hourly sun exposure under a set of shelters. Where sun is below horizon default to 0 sun visibility.

    Args:
        shelters (List[Shelter]):
            A list of shelter objects.
        epw (EPW):
            A Ladybug EPW object.
        include_radiation_porosity (bool, optional):
            If True, then increase exposure according to shelter porosity. Defaults to True.

    Returns:
        List[float]:
            A list of annual hourly values denoting sun exposure values.
    """

    suns = sun_position_list(epw)
    if len(shelters) == 0:
        return [1 if sun.altitude > 0 else 0 for sun in suns]

    sun_exposures = []
    for shelter in shelters:
        sun_exposures.append(
            [shelter.sun_exposure(sun, include_radiation_porosity) for sun in suns]
        )
    return np.prod(sun_exposures, axis=0)


def effective_wind_speed(
    shelters: List[Shelter],
    wind_speed: float,
    wind_direction: float,
    edge_acceleration_factor: float = 1.1,
    obstruction_band_width: float = 5,
) -> float:
    """Determine the effective wind speed from a given direction based on multiple obstructing shelters and edge acceleration effects.

    Wind is considered "obstructured" in some way if its direction +/-5 degrees of the shelter.
    When fully obstructed, then the porosity of the shelter is used to reduce wind speed.
    Where partially obstructed, the edge effects are expected based on the level of porosity
    of the shelter.
    Where not obstructed, then wind is kept per input.

    Args:
        shelters (List[Shelter]):
            A list of shelter objects.
        wind_direction (float):
            A single wind direction between 0-360.
        wind_speed (float):
            A wind speed in m/s.
        edge_acceleration_factor (float, optional):
            The proportional increase in wind speed due to edge acceleration around a shelter edge. Defaults to 1.1.
        obstruction_band_width (float, optional):
            The azimuthal range over which obstruction is checked. Defaults to 5.

    Returns:
        float:
            A resultant wind speed subject to obstruction from the shelter.
    """

    if len(shelters) == 0:
        return wind_speed

    effective_wind_speeds = []
    for shelter in shelters:
        effective_wind_speeds.append(
            shelter.effective_wind_speed(
                wind_direction,
                wind_speed,
                edge_acceleration_factor,
                obstruction_band_width,
            )
        )
    return np.min(effective_wind_speeds)


def annual_effective_wind_speed(
    shelters: List[Shelter],
    epw: EPW,
    edge_acceleration_factor: float = 1.1,
    obstruction_band_width: float = 5,
) -> List[float]:
    """Determine the effective wind speed from a given direction based on multiple obstructing shelters and edge acceleration effects.

    Wind is considered "obstructured" in some way if its direction +/-5 degrees of the shelter.
    When fully obstructed, then the porosity of the shelter is used to reduce wind speed.
    Where partially obstructed, the edge effects are expected based on the level of porosity
    of the shelter.
    Where not obstructed, then wind is kept per input.

    Args:
        shelters (List[Shelter]):
            A list of shelter objects.
        epw (EPW):
            A Ladybug EPW object.
        edge_acceleration_factor (float, optional):
            The proportional increase in wind speed due to edge acceleration around a shelter edge. Defaults to 1.1.
        obstruction_band_width (float, optional):
            The azimuthal range over which obstruction is checked. Defaults to 5.

    Returns:
        List[float]:
            A resultant list of EPW aligned wind speeds subject to obstruction from the shelter.
    """
    if len(shelters) == 0:
        return epw.wind_speed

    ws_wd = unique_wind_speed_direction(epw)
    all_effective_wind_speeds = []
    for shelter in shelters:
        ws_effective = {}
        for ws, wd in ws_wd:
            ws_effective[(ws, wd)] = shelter.effective_wind_speed(
                wd, ws, edge_acceleration_factor, obstruction_band_width
            )
        # cast back to original epw
        effective_wind_speeds = []
        for ws, wd in list(zip(*[epw.wind_speed.values, epw.wind_direction.values])):
            effective_wind_speeds.append(ws_effective[(ws, wd)])
        all_effective_wind_speeds.append(effective_wind_speeds)
    return np.min(all_effective_wind_speeds, axis=0)


class Shelters(Enum):
    """A list of pre-defined Shelter forms."""

    NORTH_SOUTH_LINEAR = Shelter(
        Vertices=_LINEAR_SHELTER_VERTICES_NORTH_SOUTH,
    )
    EAST_WEST_LINEAR = Shelter(
        Vertices=_LINEAR_SHELTER_VERTICES_NORTH_SOUTH,
    ).rotate(90)
    NORTHEAST_SOUTHWEST_LINEAR = Shelter(
        Vertices=_LINEAR_SHELTER_VERTICES_NORTH_SOUTH,
    ).rotate(45)
    NORTHWEST_SOUTHEAST_LINEAR = Shelter(
        Vertices=_LINEAR_SHELTER_VERTICES_NORTH_SOUTH,
    ).rotate(135)

    OVERHEAD_SMALL = Shelter(
        Vertices=_OVERHEAD_SHELTER_VERTICES_SMALL,
    )
    OVERHEAD_LARGE = Shelter(
        Vertices=_OVERHEAD_SHELTER_VERTICES_LARGE,
    )
    CANOPY_N_E_S_W = Shelter(Vertices=_CANOPY_NORTH)
    CANOPY_NE_SE_SW_NW = Shelter(Vertices=_CANOPY_NORTH).rotate(45)

    NORTH = Shelter(
        Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
    )
    NORTHEAST = Shelter(
        Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
    ).rotate(45)
    EAST = Shelter(
        Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
    ).rotate(90)
    SOUTHEAST = Shelter(
        Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
    ).rotate(135)
    SOUTH = Shelter(
        Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
    ).rotate(180)
    SOUTHWEST = Shelter(
        Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
    ).rotate(225)
    WEST = Shelter(
        Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
    ).rotate(270)
    NORTHWEST = Shelter(
        Vertices=_DIRECTIONAL_SHELTER_VERTICES_NORTH,
    ).rotate(315)
