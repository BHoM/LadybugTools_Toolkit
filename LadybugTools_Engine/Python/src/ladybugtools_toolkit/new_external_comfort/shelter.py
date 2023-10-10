import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import honeybee.dictutil as hb_dict_util
import honeybee_energy.dictutil as energy_dict_util
import honeybee_radiance.dictutil as radiance_dict_util
import matplotlib.pyplot as plt
import numpy as np
from honeybee.model import Face, Model, Shade
from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval
from ladybug.epw import EPW
from ladybug.viewsphere import ViewSphere
from ladybug_geometry.geometry3d import (
    Face3D,
    LineSegment3D,
    Plane,
    Point3D,
    Ray3D,
    Vector3D,
)
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d

from ..bhom import decorator_factory, keys_to_pascalcase, keys_to_snakecase
from ..ladybug_extension.epw import sun_position_list

SENSOR_HEIGHT = 1.2


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

        if height_above_ground <= SENSOR_HEIGHT:
            raise ValueError(
                f"height_above_ground must be greater than {SENSOR_HEIGHT}"
            )

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

        if radius <= 0:
            raise ValueError("radius must be greater than 0")

        if height_above_ground <= SENSOR_HEIGHT:
            raise ValueError(
                f"height_above_ground must be greater than {SENSOR_HEIGHT}"
            )

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
        distance_from_wall: float = 1.5,
        wall_height: float = 2,
        wall_length: float = 3,
        wind_porosity: float = 0,
        radiation_porosity: float = 0,
        angle: float = 0,
    ) -> "Shelter":
        """Create a shelter object representative of an adjacent wall.

        Args:
            distance_from_wall (float, optional):
                The distance from the wall in m. Defaults to 1.5m.
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

        if wall_height <= 0:
            raise ValueError("wall_height must be greater than 0")

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
            vertices=vertices,
            wind_porosity=wind_porosity,
            radiation_porosity=radiation_porosity,
        )

    @classmethod
    @decorator_factory(disable=False)
    def from_hb_shade(cls, shade: Shade) -> "Shelter":
        """Create a shelter object from a Honeybee Shade object.

        Args:
            shade (Shade):
                A Honeybee Shade object.

        Returns:
            Shelter: A Shelter object.
        """
        vertices = shade.vertices
        # get shade transmittance schedule, if it is present
        try:
            radiation_porosity = (
                shade.properties.energy.transmittance_schedule.data_collection.values
            )
            wind_porosity = (
                shade.properties.energy.transmittance_schedule.data_collection.values
            )
        except AttributeError:
            radiation_porosity = 0
            wind_porosity = 0
        return cls(
            vertices=vertices,
            wind_porosity=wind_porosity,
            radiation_porosity=radiation_porosity,
        )

    @classmethod
    @decorator_factory(disable=False)
    def from_hbjson(cls, hbjson_file: Path) -> list["Shelter"]:
        """Generate Shelter objects from a Honeybee JSON file.

        Args:
            hbjson_file (Path):
                A Path to a Honeybee JSON file.

        Returns:
            list[Shelter]: A list of Shelter objects.
        """
        hbjson_file = Path(hbjson_file)

        with open(hbjson_file) as json_file:
            data = json.load(json_file)
        try:
            hb_objs = hb_dict_util.dict_to_object(data, False)
            if hb_objs is None:
                hb_objs = energy_dict_util.dict_to_object(data, False)
                if hb_objs is None:
                    hb_objs = radiance_dict_util.dict_to_object(data, False)
        except (KeyError, ValueError):
            hb_objs = []
            for hb_dict in data.values():
                hb_obj = hb_dict_util.dict_to_object(hb_dict, False)
                if hb_obj is None:
                    hb_obj = energy_dict_util.dict_to_object(hb_dict, False)
                    if hb_obj is None:
                        hb_obj = radiance_dict_util.dict_to_object(hb_dict, False)
                hb_objs.append(hb_obj)

        if isinstance(hb_objs, Model):
            hb_objs = hb_objs.shades + hb_objs.faces

        shelters = []
        for obj in hb_objs:
            if isinstance(obj, Shade):
                shelters.append(cls.from_hb_shade(obj))
            if isinstance(obj, Face):
                shelters.append(cls.from_hb_face(obj))
        return shelters

    @decorator_factory(disable=False)
    def to_hbjson(self, hbjson_path: Path) -> Path:
        """Convert this object to a Honeybee JSON file.

        Args:
            hbjson_path (Path):
                A Path to a Honeybee JSON file.

        Returns:
            Path: A Path to the Honeybee JSON file.
        """
        # TODO - maintain the origin porosity as a properties.energy.transmissivity_schedule
        with open(hbjson_path, "w") as fp:
            json.dump(self.face3d.to_dict(), fp)

        return hbjson_path

    @property
    def origin(self) -> Point3D:
        """Create the origin point, representing an analytical person-point."""
        return Point3D(z=SENSOR_HEIGHT)

    @property
    def face3d(self) -> Face3D:
        """Create the face of the shelter object."""
        return Face3D(self.vertices)

    @property
    def hb_shade(self) -> Shade:
        """Create a Honeybee Shade object from this shelter object."""
        shd = Shade.from_vertices(identifier="shade", vertices=self.vertices)
        # pylint disable=no-member
        shd.properties.energy.transmittance_schedule = ScheduleFixedInterval(
            "porosity", (self.radiation_porosity + self.wind_porosity) / 2
        )
        # pylint enable=no-member
        return shd

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
            self.face3d.rotate_xy(angle, center).vertices,
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
            self.face3d.move(vector).vertices,
            self.wind_porosity,
            self.radiation_porosity,
        )

    @decorator_factory(disable=False)
    def set_porosity(self, porosity: float) -> "Shelter":
        """Return this shelter with an adjusted porosity value applied to both wind and radiation components."""
        return Shelter(
            vertices=self.vertices, radiation_porosity=porosity, wind_porosity=porosity
        )

    @decorator_factory(disable=False)
    def annual_sky_exposure(self, include_radiation_porosity: bool = True) -> float:
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
        n_intersections = sum(bool(self.face3d.intersect_line_ray(ray)) for ray in rays)
        if include_radiation_porosity:
            return 1 - ((n_intersections / len(rays)) * (1 - self.radiation_porosity))
        return 1 - (n_intersections / len(rays))

    @decorator_factory(disable=False)
    def annual_sun_exposure(
        self, epw: EPW, include_radiation_porosity: bool = True
    ) -> list[float]:
        """Calculate annual hourly sun exposure. Overnight hours where sun is below horizon default to np.nan sun visibility.

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
                _sun_exposure.append(np.nan)
                continue

            if radiation_porosity == 1:
                _sun_exposure.append(1)
                continue

            ray = Ray3D(self.origin, (sun.position_3d() - self.origin).normalize())

            if self.face3d.intersect_line_ray(ray) is None:
                _sun_exposure.append(1)
            else:
                _sun_exposure.append(
                    radiation_porosity if include_radiation_porosity else 0
                )
        return _sun_exposure

    def wind_exposure(
        self,
        wind_direction: float,
        obstruction_band_width: float = 10,
        n_samples_xy: int = 4,
        n_samples_rotational: int = 5,
        porosity: float = 0,
        edge_acceleration_factor: float = 1.2,
    ) -> list[float]:
        """Determine the multiplier for wind speed from a given direction based on
            shelter obstruction and edge acceleration effects.

        * When fully obstructed, the result would be 1 * porosity.
        * Where partially obstructed, edge effects are expected based on the
            level of porosity of the shelter, and the proportion of obstructed
            /unobstructed area.
        * Where not obstructed, the results would be 1.

        Args:
            wind_direction (float):
                A single wind direction between 0-360.
            obstruction_band_width (float):
                The azimuthal range over which obstruction is checked.
                Defaults to 10 degrees.
            n_samples_xy (int):
                The number of samples to take in the xy plane.
            n_samples_rotational (int):
                The number of samples to take in the rotational plane.
            porosity (bool, optional):
                A porosity value to apply to this calculation. This does not
                override the shelter porosity, but is required to enable the
                calculation of the proportional edge acceleration. Defaults
                to 0.
            edge_acceleration_factor (float, optional):
                The proportional increase in wind speed due to edge acceleration
                around a shelter edge. Defaults to 1.2.

        Returns:
            list[float]:
                A list of multipliers for each hour of the year which describes
                how much to multiply wind by for the given direction.
        """

        wind_direction_rad = np.deg2rad(wind_direction)
        wind_direction_vector = Vector3D(
            np.sin(wind_direction_rad), np.cos(wind_direction_rad)
        )  # TODO - check that this is going the right direction
        wind_direction_plane = Plane(n=wind_direction_vector, o=self.origin)

        # check that the shelter is not in the opposite direction to the wind
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
        intersections = [bool(self.face3d.intersect_line_ray(i)) for i in d.values()]

        if sum(intersections) == 0:
            return 1

        # calculate the resultant multiplier
        return (
            (porosity * sum(intersections))
            + (edge_acceleration_factor * sum([not i for i in intersections]))
        ) / len(intersections)

    @decorator_factory(disable=False)
    def annual_wind_speed(
        self,
        epw: EPW,
        include_wind_porosity: bool = True,
    ) -> list[float]:
        """Calculate annual hourly effective wind speed. Wind from the given
        EPW file will be translated to 1.2m above ground.

        Args:
            epw (EPW):
                A Ladybug EPW object.
            include_wind_porosity (bool, optional):
                If True, then increase exposure according to shelter porosity.

        Returns:
            List[float]:
                A list of annual hourly values denoting sun exposure values.
        """

        if all(self.wind_porosity == 1):
            return epw.wind_speed.values

        _wind_speed = []
        for ws, wd, porosity in list(
            zip(
                *[
                    epw.wind_speed,
                    epw.wind_direction,
                    self.wind_porosity,
                ]
            )
        ):
            if ws == 0:
                _wind_speed.append(0)
                continue
            _wind_speed.append(
                ws
                * self.wind_exposure(
                    wind_direction=wd,
                    porosity=porosity if include_wind_porosity else 0,
                )
            )

        return _wind_speed

    @decorator_factory(disable=False)
    def visualise(
        self,
        ax: plt.Axes = None,
        tri_kwargs: dict[str, Any] = None,
        lim_kwargs: dict[str, tuple[float]] = None,
    ) -> Figure:
        """Visualise this shelter to check validity and that it exists where you think it should!

        Args:
            ax (plt.Axes, optional):
                A matplotlib axes object. Defaults to None.
            tri_kwargs:
                Additional keyword arguments to pass to the Poly3DCollection (shelter) render object.
            lim_kwargs:
                Additional keyword arguments to pass to the x/y/z lims of the axes.

        """

        if ax is None:
            fig = plt.figure()
            ax = mplot3d.Axes3D(fig)
            fig.add_axes(ax)
        if not isinstance(ax, mplot3d.Axes3D):
            raise ValueError("ax must be a 3D matplotlib axes object")

        if tri_kwargs is None:
            tri_kwargs = {}
        if lim_kwargs is None:
            lim_kwargs = {}

        # TODO - make this use the Ladybug-Matplotlib renderer when it is ready

        ax.scatter(*self.origin.to_array(), c="red")

        # add shelter as a polygon
        vtx = np.array([i.to_array() for i in self.face3d.vertices])
        tri = mplot3d.art3d.Poly3DCollection([vtx])
        tri.set_color(
            tri_kwargs.get(
                "color",
                tri_kwargs.get(
                    "c", tri_kwargs.get("fc", tri_kwargs.get("facecolor", "grey"))
                ),
            )
        )
        tri.set_alpha(
            tri_kwargs.get(
                "alpha",
                tri_kwargs.get("a", 0.5),
            )
        )
        tri.set_edgecolor(
            tri_kwargs.get(
                "edgecolor",
                tri_kwargs.get("ec", "k"),
            )
        )
        ax.add_collection3d(tri)

        # format axes
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # set lims
        ax.set_xlim(
            lim_kwargs.get("xlim", (min(i[0] for i in vtx), max(i[0] for i in vtx)))
        )
        ax.set_ylim(
            lim_kwargs.get("ylim", (min(i[1] for i in vtx), max(i[1] for i in vtx)))
        )

        # pylint: disable=no-member
        ax.set_zlim(
            lim_kwargs.get("zlim", (min(i[2] for i in vtx), max(i[2] for i in vtx)))
        )
        # pylint: enable=no-member

        ax.set_aspect("equal")
        # ax.autoscale()

        return fig


def annual_sky_exposure(
    shelters: list[Shelter], include_radiation_porosity: bool = True
) -> list[float]:
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

    results = []
    for ray in rays:
        result = np.ones_like(shelters[0].radiation_porosity)
        for shelter in shelters:
            if bool(shelter.face3d.intersect_line_ray(ray)):
                result *= (
                    shelter.radiation_porosity if include_radiation_porosity else 1
                )
        results.append(result)
    results = np.clip(results, 0, None)
    return results.sum(axis=0) / (len(rays) * len(shelters))


def annual_sun_exposure(
    shelters: list[Shelter], epw: EPW, include_radiation_porosity: bool = True
) -> list[float]:
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

    result = np.ones_like(shelters[0].radiation_porosity)
    for shelter in shelters:
        result = result * shelter.annual_sun_exposure(
            epw=epw, include_radiation_porosity=include_radiation_porosity
        )
    return result


def annual_wind_speed(
    shelters: list[Shelter],
    epw: EPW,
) -> list[float]:
    """Determine the effective hourly annual wind speed based on a combination
        of multiple shelters in a given EPW file location.

    Note:
        This method does not modify wind speed from 10m (assumed for height of
        wind in EPW files). This is so that the resultant values are valid for
        use in the UTCI calculation process.

    Args:
        shelters (List[Shelter]):
            A list of shelter objects.
        epw (EPW):
            A Ladybug EPW object.

    Returns:
        List[float]:
            A resultant list of EPW aligned wind speeds subject to obstruction from the shelter.
    """
    if not bool(shelters):
        return epw.wind_speed.values

    results = []
    for shelter in shelters:
        results.append(shelter.annual_wind_speed(epw=epw))
    return np.array(results).min(axis=0)


def write_shelters_to_hbjson(shelters: list[Shelter], hbjson_path: Path) -> Path:
    """Create a Honeybee JSON file from a list of Shelter objects.

    Args:
        shelters (List[Shelter]):
            A list of Shelter objects.
        hbjson_path (Path):
            A Path to a Honeybee JSON file.

    Returns:
        Path: A Path to the Honeybee JSON file.
    """

    name = hbjson_path.stem
    directory = hbjson_path.parent
    Model(identifier="Shelters", orphaned_faces=[i.face3d for i in shelters]).to_hbjson(
        name=name, parent=directory
    )
    return hbjson_path


class Shelters(Enum):
    """A list of pre-defined Shelter forms."""

    NORTH_SOUTH_LINEAR = Shelter.from_overhead_linear(
        width=3,
        height_above_ground=3.5,
        length=2000,
        wind_porosity=0,
        radiation_porosity=0,
        angle=0,
    )
    EAST_WEST_LINEAR = Shelter.from_overhead_linear(
        width=3,
        height_above_ground=3.5,
        length=2000,
        wind_porosity=0,
        radiation_porosity=0,
        angle=90,
    )
    NORTHEAST_SOUTHWEST_LINEAR = Shelter.from_overhead_linear(
        width=3,
        height_above_ground=3.5,
        length=2000,
        wind_porosity=0,
        radiation_porosity=0,
        angle=45,
    )
    NORTHWEST_SOUTHEAST_LINEAR = Shelter.from_overhead_linear(
        width=3,
        height_above_ground=3.5,
        length=2000,
        wind_porosity=0,
        radiation_porosity=0,
        angle=135,
    )

    OVERHEAD_SMALL = Shelter.from_overhead_circle(
        radius=1.5, height_above_ground=3.5, wind_porosity=0, radiation_porosity=0
    )
    OVERHEAD_LARGE = Shelter.from_overhead_circle(
        radius=5, height_above_ground=3.5, wind_porosity=0, radiation_porosity=0
    )

    NORTH = Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        wind_porosity=0,
        radiation_porosity=0,
        angle=0,
    )
    NORTHEAST = Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        wind_porosity=0,
        radiation_porosity=0,
        angle=45,
    )
    EAST = Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        wind_porosity=0,
        radiation_porosity=0,
        angle=90,
    )
    SOUTHEAST = Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        wind_porosity=0,
        radiation_porosity=0,
        angle=135,
    )
    SOUTH = Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        wind_porosity=0,
        radiation_porosity=0,
        angle=180,
    )
    SOUTHWEST = Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        wind_porosity=0,
        radiation_porosity=0,
        angle=225,
    )
    WEST = Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        wind_porosity=0,
        radiation_porosity=0,
        angle=270,
    )
    NORTHWEST = Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        wind_porosity=0,
        radiation_porosity=0,
        angle=315,
    )
