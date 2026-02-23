"""Methods for working with angles, vectors and geometry."""

import warnings
from collections import defaultdict
from math import acos, cos, sin
from typing import Sequence, Union
from warnings import warn  # pylint: disable=E0401

import numpy as np
from ladybug.location import Location
from ladybug_geometry.geometry2d import Mesh2D, Point2D
from ladybug_geometry.geometry3d import (
    Face3D,
    LineSegment3D,
    Mesh3D,
    Plane,
    Point3D,
    Vector3D,
)
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree

from python_toolkit.helpers import cardinality


def mesh3d_isplanar(mesh: Mesh3D) -> bool:
    """Check if a mesh is planar.

    Args:
        mesh (Mesh3D): A ladybug-geometry Mesh3D.

    Returns:
        bool: True if the mesh is planar, False otherwise.

    """
    return len(set(mesh.vertex_normals)) == 1


def mesh3d_get_plane(mesh: Mesh3D) -> Plane:
    """Estimate the plane of a mesh.

    Args:
        mesh (Mesh3D): A ladybug-geometry Mesh3D.

    Returns:
        Plane: The estimated plane of the mesh.

    """
    if not mesh3d_isplanar(mesh=mesh):
        warn(
            "The mesh given is not planar. This method will return a planar mesh "
            "based on a selection of 3-points from the first 3-faces of this mesh."
        )

    plane = Plane.from_three_points(
        *[mesh.vertices[j] for j in [i[0] for i in mesh.faces[:3]]]
    )

    if plane.n.z < 0:
        warn(
            "The plane normal is pointing downwards. This method will return a plane with a normal pointing upwards."
        )
        return plane.flip()

    return plane


def pt_distances(base_point: Point2D, points: list[Point2D]) -> list[float]:
    """Return the distance from each pt to each other point in the input list"""
    # for each emitter, get the distance to all "receiving" points
    distances = cdist([base_point], points)

    return distances


def great_circle_distance(location1: Location, location2: Location) -> float:
    """Calculate the great circle distance between two points on the earth
    (specified in decimal degrees), in metres.

    Args:
        location1 (Location):
            Location object of the first location
        location2 (Location):
            Location object of the second location

    Returns:
        distance (float):
            The distance between the two locations in m

    """
    r = 6373.0  # approximate radius of earth in km
    lat1 = np.radians(location1.latitude)
    lon1 = np.radians(location1.longitude)
    lat2 = np.radians(location2.latitude)
    lon2 = np.radians(location2.longitude)
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = r * c
    return distance * 1000


def haversine(location1: Location, location2: Location) -> float:
    """Proxy for accessing the great circle distance between two locations.
    """
    return great_circle_distance(location1, location2)


def azimuth_from_cardinal(cardinal_direction: str) -> float:
    """For a given cardinal direction, return the corresponding angle in degrees.

    Args:
        cardinal_direction (str):
            The cardinal direction.

    Returns:
        float:
            The angle associated with the cardinal direction.

    """
    cardinal_directions = [
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
    ]
    if cardinal_direction not in cardinal_directions:
        raise ValueError(f"{cardinal_direction} is not a known cardinal_direction.")
    angles = np.arange(0, 360, 11.25)

    lookup = dict(zip(cardinal_directions, angles))

    return lookup[cardinal_direction]


def vector_to_azimuth(vector: list[Union[float, int]], degrees: bool = True) -> float:
    """For a vector, determine the clockwise angle to north at [0, 1].

    Args:
        vector (list[Union[float, int]]):
            A 2D vector object.
        degrees (bool, optional):
            Return the angle in degrees.
            Defaults to True.

    Returns:
        float:
            The angle between vector and north clockwise from 0-359.9.

    """
    if len(vector) != 2:
        raise ValueError("The vector must be 2D.")

    north = [0, 1]
    angle1 = np.arctan2(*north[::-1])  # type: ignore
    angle2 = np.arctan2(*vector[::-1])  # type: ignore
    rad = (angle1 - angle2) % (2 * np.pi)
    if degrees:
        return np.rad2deg(rad)
    return rad


def vector_to_azimuth_altitude(
    vector: Vector3D, degrees: bool = True
) -> tuple[float, float]:
    """Convert a 3D vector to azimuth and altitude angles.

    Args:
        vector (Vector3D):
            A 3D vector object.
        degrees (bool, optional):
            Return the angles in degrees if True, otherwise in radians.
            Defaults to True.

    Returns:
        tuple[float, float]:
            - The azimuth angle, in degrees clockwise from north (at 0 degrees).
            - The altitude angle, in degrees from the horizontal plane (at 0 degrees).

    """
    x, y, z = vector.to_array()
    azimuth = np.rad2deg(np.arctan2(x, y))
    horizontal_distance = np.sqrt(x**2 + y**2)
    altitude = np.rad2deg(np.arctan2(z, horizontal_distance))

    if azimuth < 0:
        azimuth += 360

    if degrees:
        return float(azimuth), float(altitude)

    return float(np.deg2rad(azimuth)), float(np.deg2rad(altitude))


def azimuth_altitude_to_vector(azimuth: float, altitude: float) -> Vector3D:
    """Convert azimuth and altitude angles to a 3D vector.

    Args:
        azimuth (float):
            The azimuth angle in degrees, clockwise from north (at 0 degrees).
        altitude (float):
            The altitude angle in degrees, from the horizontal plane (at 0 degrees).

    Returns:
        Vector3D:
            A 3D vector object representing the direction.

    """
    if altitude < -90 or altitude > 90:
        raise ValueError("altitude must be between -90 and 90 degrees.")
    if azimuth < -360 or azimuth > 360:
        raise ValueError("azimuth must be between -360 and 360 degrees.")

    # Convert angles from degrees to radians
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(altitude)

    # Calculate the x, y, and z components
    x = np.cos(altitude_rad) * np.sin(azimuth_rad)
    y = np.cos(altitude_rad) * np.cos(azimuth_rad)
    z = np.sin(altitude_rad)

    return Vector3D(x=x, y=y, z=z)


def azimuth_to_vector(azimuth: Union[int, float]) -> tuple[float, float]:
    """Return the X, Y vector from of an angle from north at 0-degrees.

    Args:
        azimuth (float):
            The angle from north in degrees clockwise from [0, 360], though
            any number can be input here for angles greater than a full circle.

    Returns:
        list[float]:
            A vector of length 2.

    """
    azimuth = np.radians(azimuth)

    return (np.sin(azimuth), np.cos(azimuth))

#TODO: move to python_toolkit, under a geometry folder
def circular_weighted_mean(
    angles: Sequence[Union[int, float]],
    weights: Union[Sequence[Union[int, float]], None] = None,
):
    """Get the average angle from a set of weighted angles.

    Args:
        angles (list[float]):
            A collection of equally weighted wind directions, in degrees from North (0).
        weights (list[float]):
            A collection of weights, which must sum to 1.
            Defaults to None which will equally weight all angles.

    Returns:
        float:
            An average wind direction.

    """
    # convert angles to 0-360
    angles = np.where(angles == 360, 0, angles).tolist()

    # handle case where weights are not provided
    if weights is None:
        weights = (np.ones_like(angles) / len(angles)).tolist()

    if len(angles) != len(weights):  # type: ignore
        raise ValueError("weights must be the same size as angles.")

    if any(i < 0 for i in angles) or any(i > 360 for i in angles):
        raise ValueError("Input angles exist outside of expected range (0-360).")

    # checks for opposing or equally spaced angles, with equal weighting
    if len(set(weights)) == 1:  # type: ignore
        _sorted = np.sort(angles)
        if len(set(angles)) == 2:
            a, b = np.meshgrid(_sorted, _sorted)
            if np.any(a - b == 180):
                warnings.warn(
                    "Input angles are opposing, meaning determining the mean is impossible. An attempt will be made to determine the mean, but this will be perpendicular to the opposing angles and not accurate."
                )
        if any(np.diff(_sorted) == 360 / len(angles)):
            warnings.warn(
                "Input angles are equally spaced, meaning determining the mean is impossible. An attempt will be made to determine the mean, but this will not be accurate."
            )
    weight_sum = sum(weights)  # type: ignore
    weights = [weight / weight_sum for weight in weights]  # type: ignore

    x = y = 0.0
    for angle, weight in zip(angles, weights):
        x += np.cos(np.radians(angle)) * weight
        y += np.sin(np.radians(angle)) * weight

    mean = np.degrees(np.arctan2(y, x))

    if mean < 0:
        mean = 360 + mean

    if mean in (360.0, -0.0):
        mean = 0.0

    return np.round(mean, 5)


def point_group(points: list[list[float]], threshold: float) -> list[list[float]]:
    """Cluster 2D or 3D points based on proximity.

    Args:
        points (list[list[float]]):
            A list of 2D or 3D points.
        threshold (float):
            The maximum distance between points to be considered neighbors.

    Returns:
        clusters: list[list[float]]
            The points in each cluster.

    """
    # ensure points are in the correct format (list[list[number]])
    if not all(isinstance(point, (list, tuple)) for point in points):
        raise ValueError("All points must be a list or tuple.")
    # ensure each point is numeric
    if not all(isinstance(coord, (int, float)) for point in points for coord in point):
        raise ValueError("All points must be numeric.")

    if threshold < 0:
        raise ValueError("The threshold must be greater than 0.")

    # Ensure points are in the correct format
    point_dim = len(points[0])
    if not all(len(point) == point_dim for point in points):
        raise ValueError("All points must have the same dimensionality (2D or 3D).")

    if point_dim not in (2, 3):
        raise ValueError("Only 2D or 3D points are supported.")

    if len(points) == 1:
        return [points]  # type: ignore

    if threshold == 0:
        # return original points
        return [points]

    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))

        def find(self, i):
            if self.parent[i] != i:
                self.parent[i] = self.find(self.parent[i])
            return self.parent[i]

        def union(self, i, j):
            root_i = self.find(i)
            root_j = self.find(j)
            if root_i != root_j:
                self.parent[root_i] = root_j

    tree = KDTree(points)

    # Initialize Union-Find
    uf = UnionFind(len(points))

    # Find neighboring points within radius and union them
    for i, point in enumerate(points):
        neighbor_indices = tree.query_radius(X=[point], r=threshold)[0]
        for neighbor_index in neighbor_indices:
            uf.union(i, neighbor_index)

    # Collect fused points and assign labels
    label_groups = defaultdict(list)

    for i in range(len(points)):
        root = uf.find(i)
        label_groups[root].append(i)

    clusters = []
    for _, points_indices in label_groups.items():
        clusters.append([points[i] for i in points_indices])

    return clusters


def _create_azimuth_mesh(directions: int = 36, tilt_angle: float = 0) -> Mesh3D:
    """Create a mesh of faces for a given number of directions and tilt angle.

    This is used to creation the radiation rose, with one face per direction.

    Args:
        directions (int, optional):
            The number of directions to divide the rose into.
            Default is 36.
        tilt_angle (float, optional):
            The tilt angle in degrees from horizontal. 0 is horizontal, 90 is upwards.
            Default is 0.

    Returns:
        Mesh3D:
            A ladybug Mesh3D object

    """
    angles = np.linspace(0, 360, directions, endpoint=False)
    base_face = Face3D.from_extrusion(
        line_segment=LineSegment3D(p=Point3D(0.05, 0.1, 0), v=Vector3D(-0.1, 0, 0)),
        extrusion_vector=Vector3D(0, 0, 0.1),
    ).rotate(axis=Vector3D(1, 0, 0), angle=np.deg2rad(tilt_angle), origin=Point3D())
    faces = [
        base_face.rotate_xy(angle=np.deg2rad(-a), origin=Point3D()) for a in angles
    ]
    return Mesh3D.from_face_vertices(faces=faces)


def icosphere(
    resolution: int = 1, radius: float = 0.5, origin: Point3D = Point3D()
) -> Mesh3D:
    """Create an icosphere mesh with the given resolution.

    Args:
        resolution (int, optional):
            The number of subdivisions for the icosphere.
            Default is 1.
        radius (float, optional):
            The radius of the icosphere.
            Default is 0.5.

    Returns:
        Mesh3D:
            The resulting icosphere mesh.

    """
    if resolution <= 0:
        raise ValueError("Resolution must be greater than 0.")
    if not isinstance(resolution, int):
        raise ValueError("Resolution must be an integer.")
    if radius <= 0:
        raise ValueError("Radius must be greater than 0.")
    if not isinstance(origin, Point3D):
        raise ValueError("Origin must be a ladybug-geometry Point3D object.")

    def slerp(start: Vector3D, end: Vector3D, t: float) -> Vector3D:
        """Spherical linear interpolation."""
        dot = max(-1.0, min(1.0, start.dot(end)))  # Clamp dot product to avoid errors
        theta = acos(dot) * t
        relative_vec = end - start * dot
        relative_vec = relative_vec.normalize()
        return start * cos(theta) + relative_vec * sin(theta)

    # Base vertices of an icosahedron
    phi = (1 + 5**0.5) / 2  # Golden ratio
    base_vertices = [
        Vector3D(-1, phi, 0).normalize(),
        Vector3D(1, phi, 0).normalize(),
        Vector3D(-1, -phi, 0).normalize(),
        Vector3D(1, -phi, 0).normalize(),
        Vector3D(0, -1, phi).normalize(),
        Vector3D(0, 1, phi).normalize(),
        Vector3D(0, -1, -phi).normalize(),
        Vector3D(0, 1, -phi).normalize(),
        Vector3D(phi, 0, -1).normalize(),
        Vector3D(phi, 0, 1).normalize(),
        Vector3D(-phi, 0, -1).normalize(),
        Vector3D(-phi, 0, 1).normalize(),
    ]

    # Base faces of an icosahedron (triangles)
    base_faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    # Initialize vertices and faces
    vertices = base_vertices[:]
    faces = base_faces[:]

    # Subdivide each face
    for _ in range(resolution):
        new_faces = []
        midpoints = {}

        def get_midpoint(v1, v2):
            """Get or create the midpoint of two vertices."""
            smaller, larger = min(v1, v2), max(v1, v2)
            key = (smaller, larger)
            if key not in midpoints:
                midpoint = slerp(vertices[smaller], vertices[larger], 0.5).normalize()
                midpoints[key] = len(vertices)
                vertices.append(midpoint)
            return midpoints[key]

        for v1, v2, v3 in faces:
            # Split each edge of the triangle
            a = get_midpoint(v1, v2)
            b = get_midpoint(v2, v3)
            c = get_midpoint(v3, v1)

            # Create four new faces
            new_faces.append((v1, a, c))
            new_faces.append((a, v2, b))
            new_faces.append((c, b, v3))
            new_faces.append((a, b, c))

        faces = new_faces

    # Convert vertices to Point3D and create the mesh
    points = [Point3D(v.x * radius, v.y * radius, v.z * radius) for v in vertices]
    return Mesh3D(vertices=points, faces=faces).move(moving_vec=origin)


def points_to_mesh3d(points: list[Point3D], alpha: float) -> Mesh3D:
    """Convert a list of points to a ladybug-geometry Mesh3D.

    Args:
        points (list[Point3D]):
            A list of ladybug-geometry Point3D objects.
        alpha (float):
            The alpha value for the mesh. Mesh faces with edges not within this
            tolerance will be removed.

    Returns:
        Mesh3D: A ladybug-geometry Mesh3D.

    """
    # TODO - implement this function
    # - find the most planar plane, and try to create the mesh around that
    raise NotImplementedError()


def project_mesh3d_to_mesh2d(mesh: Mesh3D, plane: Plane = Plane()) -> Mesh2D:
    """Project a mesh to a 2D plane.

    Args:
        mesh (Mesh3D):
            A ladybug-geometry Mesh3D object.
        plane (Plane):
            The plane to project the mesh to.
            Defaults to the XY plane.

    Returns:
        Mesh2D: A ladybug-geometry Mesh2D object.

    """
    projected_vertices = [plane.project_point(pt) for pt in mesh.vertices]
    projected_vertices_2d = [Point2D(*i.to_array()[:-1]) for i in projected_vertices]
    return Mesh2D(vertices=projected_vertices_2d, faces=mesh.faces, colors=mesh.colors)


def scale_mesh_non_uniform(
    mesh: Mesh3D, x: float = 1, y: float = 1, z: float = 1, origin: Point3D = None
) -> Mesh3D:
    """Scale a mesh non-uniformly in the x, y, and z directions.

    Args:
        mesh (Mesh3D):
            A ladybug-geometry Mesh3D object.
        x (float):
            The scale factor in the x direction.
            Defaults to 1.
        y (float):
            The scale factor in the y direction.
            Defaults to 1.
        z (float):
            The scale factor in the z direction.
            Defaults to 1.
        origin (Point3D, optional):
            The origin point for scaling.
            If None, the mesh will be scaled from its centroid.
            Defaults to None.

    Returns:
        Mesh3D: A ladybug-geometry Mesh3D object.

    """
    if origin is None:
        origin = mesh.center
    if not isinstance(origin, Point3D):
        raise ValueError("The origin must be a ladybug-geometry Point3D object.")
    if not isinstance(mesh, Mesh3D):
        raise ValueError("The mesh must be a ladybug-geometry Mesh3D object.")
    if any(i == 0 for i in (x, y, z)):
        raise ValueError("Scale factors must not be 0.")

    new_vertices = []
    for vert in mesh.vertices:
        new_vertices.append(
            Point3D(
                (vert.x - origin.x) * x + origin.x,
                (vert.y - origin.y) * y + origin.y,
                (vert.z - origin.z) * z + origin.z,
            )
        )

    return Mesh3D(vertices=new_vertices, faces=mesh.faces)

def cull_mesh_faces(mesh: Mesh3D, mask: Sequence[bool]) -> Mesh3D:
    """Cull mesh faces by mask and remove unused vertices.

    Args:
        mesh: Mesh3D object.
        mask: List of booleans, same length as mesh.faces.

    Returns:
        Mesh3D: New mesh with culled faces and unused vertices removed.
    """
    # Keep only faces where mask is True
    new_faces = [face for face, keep in zip(mesh.faces, mask) if keep]

    # Find all used vertex indices
    used_indices = set(i for face in new_faces for i in face)

    # Map old indices to new indices
    index_map = {
        old_idx: new_idx for new_idx, old_idx in enumerate(sorted(used_indices))
    }

    # Create new vertex list
    new_vertices = [mesh.vertices[i] for i in sorted(used_indices)]

    # Remap faces to new indices
    remapped_faces = [tuple(index_map[i] for i in face) for face in new_faces]

    # Optionally, filter colors if mesh.colors is by face
    colors = None
    if mesh.colors is not None and mesh.is_color_by_face:
        colors = [c for c, keep in zip(mesh.colors, mask) if keep]

    return Mesh3D(new_vertices, remapped_faces, colors)