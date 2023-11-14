"""Methods for handling Honeybee sensorgrids."""
# pylint: disable=E0401
from warnings import warn

import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from honeybee_radiance.sensorgrid import SensorGrid
from ladybug_geometry.geometry3d import Plane, Point3D, Vector3D
from scipy.spatial.distance import cdist

from ...ladybug_geometry_extension import mesh3d_get_plane
from ...plot.utilities import create_triangulation

# pylint: enable=E0401


def is_planar(sensorgrid: SensorGrid) -> bool:
    """Check if a sensorgrid is planar.

    Args:
        sensorgrid (SensorGrid): A honeybee-radiance SensorGrid.

    Returns:
        bool: True if the sensorgrid is planar, False otherwise.
    """

    plane = Plane.from_three_points(
        *[Point3D.from_array(i.pos) for i in sensorgrid.sensors[0:3]]
    )
    for sensor in sensorgrid.sensors:
        if not np.isclose(
            a=plane.distance_to_point(point=Point3D.from_array(sensor.pos)), b=0
        ):
            return False
    return True


def get_plane(sensorgrid: SensorGrid) -> Plane:
    """Create a plane from a sensorgrid.

    Args:
        sensorgrid (SensorGrid):
            A honeybee-radiance SensorGrid.

    Returns:
        Plane:
            A ladybug-geometry Plane.
    """

    if len(sensorgrid.sensors) < 3:
        raise ValueError(
            "sensor grid must contain at least 3 points to create a plane."
        )

    for sensor in sensorgrid.sensors[2:]:
        plane = Plane.from_three_points(
            o=Point3D(*sensorgrid.sensors[0].pos),
            p2=Point3D(*sensorgrid.sensors[1].pos),
            p3=Point3D(*sensor.pos),
        )
        if sum(plane.n.to_array()) != 0:
            break

    if sum(plane.n.to_array()) == 0:
        raise ValueError(
            "sensor grid contains points which are colinear and cannot create a triangle."
        )

    if plane.n.z == -1:
        plane = plane.flip()

    origin = list(plane.o.to_array())
    normal = list(plane.n.to_array())

    for n, i in enumerate(origin):
        if abs(i) == i:
            origin[n] = abs(i)
    for n, i in enumerate(normal):
        if abs(i) == i:
            normal[n] = abs(i)

    if not is_planar(sensorgrid):
        raise ValueError("sensorgrid must be planar to create a plane.")

    return Plane(o=Point3D(*origin), n=Vector3D(*normal))


def groupby_level(
    sensorgrids: list[SensorGrid],
) -> dict[float, list[SensorGrid]]:
    """Group sensorgrids by their level.

    Args:
        sensorgrids (list[SensorGrid]):
            A list of honeybee-radiance SensorGrids.

    Returns:
        dict[float, list[SensorGrid]]:
            A dictionary of lists of honeybee-radiance SensorGrids.
    """

    d = {}
    for grid in sensorgrids:
        level = grid.sensors[0].pos[-1]
        if not is_planar(grid):
            warn(
                f"{grid} is not planar. This may cause issues when grouping by level. It will be assumed to be at level {level}."
            )
        if level not in d:
            d[level] = [grid]
        else:
            d[level].append(grid)

    return d


def position_array(sensorgrid: SensorGrid) -> np.ndarray:
    """Convert a honeybee-radiance SensorGrid to a numpy array of XYZ coordinates.

    Args:
        sensorgrid (SensorGrid):
            A honeybee-radiance SensorGrid.

    Returns:
        np.ndarray:
            A numpy array of the sensorgrid.
    """

    return np.array([i.pos for i in sensorgrid.sensors])


def vector_array(sensorgrid: SensorGrid) -> np.ndarray:
    """Convert a honeybee-radiance SensorGrid to a numpy array of XYZ coordinates.

    Args:
        sensorgrid (SensorGrid):
            A honeybee-radiance SensorGrid.

    Returns:
        np.ndarray:
            A numpy array of the sensorgrid.
    """

    return np.array([i.dir for i in sensorgrid.sensors])


def estimate_spacing(sensorgrid: SensorGrid) -> float:
    """Estimate the spacing of a sensorgrid.

    Args:
        sensorgrid (SensorGrid):
            A honeybee-radiance SensorGrid.

    Returns:
        float:
            The estimated spacing (between sensors) of the sensorgrid.
    """
    if not isinstance(sensorgrid, SensorGrid):
        raise ValueError("sensorgrid must be a honeybee-radiance SensorGrid.")

    if not is_planar(sensorgrid):
        raise ValueError("sensorgrid must be planar.")

    pts = position_array(sensorgrid)

    return np.sort(np.sort(cdist(pts, pts))[:, 1])[-1]


def as_triangulation(
    sensorgrid: SensorGrid,
    alpha_adjust: float = 0.1,
) -> mtri.Triangulation:
    """Create matploltib triangulation from a SensorGrid object.

    Args:
        sensorgrid (SensorGrid):
            A honeybee-radiance SensorGrid.
        alpha_adjust (float, optional):
            A value to adjust the alpha value of the triangulation. Defaults to 0.1.

    Returns:
        Triangulation: A matplotlib triangulation.
    """

    alpha = estimate_spacing(sensorgrid) + alpha_adjust
    plane = get_plane(sensorgrid)

    if not any([(plane.n.z == 1), (plane.n.z == -1)]):
        warn(
            "The sensorgrid given is planar, but not in the XY plane. You may need to rotate this when visualising it!"
        )

    x, y = np.array(
        [
            plane.xyz_to_xy(Point3D(*sensor.pos)).to_array()
            for sensor in sensorgrid.sensors
        ]
    ).T

    return create_triangulation(x, y, alpha=alpha)


def as_patchcollection(
    sensorgrid: SensorGrid, **kwargs
) -> mcollections.PatchCollection:
    """Get the matplotlib PatchCollection for a set of sensor grids.

    Args:
        sensorgrid (SensorGrid):
            A honeybee-radiance SensorGrid.
        **kwargs:
            Additional keyword arguments to pass to the PatchCollection objects created.

    Returns:
        mcollections.PatchCollection:
            The matplotlib PatchCollection.
    """

    if sensorgrid.mesh is None:
        raise ValueError(
            f"{sensorgrid} does not have a mesh. This should have been assigned when the sensorgrid was created."
        )

    # flatten the mesh to 2D in XY plane, raising warnings if the mesh is not 2D planar
    mesh3d_get_plane(mesh=sensorgrid.mesh)

    patches = []
    for face in sensorgrid.mesh.face_vertices:
        patches.append(
            mpatches.Polygon(np.array([i.to_array()[:2] for i in face]), closed=False)
        )
    return mcollections.PatchCollection(patches, **kwargs)


def plot_values(
    sensorgrid: SensorGrid, values: list[float], ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    """Plot a sensorgrid with values.

    Args:
        sensorgrid (SensorGrid):
            A honeybee-radiance SensorGrid.
        values (List[float]):
            A list of values to plot.
        ax (plt.Axes, optional):
            A matplotlib Axes object. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the PatchCollection objects created.

    Returns:
        mcollections.PatchCollection: The matplotlib PatchCollection.
    """

    if ax is None:
        ax = plt.gca()

    pc = as_patchcollection(sensorgrid, **kwargs)
    pc.set_array(values)
    ax.add_collection(pc)

    ax.autoscale_view()
    ax.set_aspect("equal")

    return ax


def get_limits(sensorgrids: list[SensorGrid], buffer: float = 2) -> tuple[tuple[float]]:
    """Get the X/Y limits from a set of sensor grids.

    Args:
        sensorgrids (list[SensorGrid]):
            A list of honeybee-radiance SensorGrids.
        buffer (float, optional):
            A buffer to add to the limits. Defaults to 2.

    Returns:
        tuple[tuple[float]]: The X/Y limits.
    """

    x_low = []
    x_high = []
    y_low = []
    y_high = []
    for grid in sensorgrids:
        x, y, _ = position_array(grid).T
        x_low.append(np.min(x))
        x_high.append(np.max(x))
        y_low.append(np.min(y))
        y_high.append(np.max(y))

    return (
        (np.min(x_low) - buffer, np.max(x_high) + buffer),
        (np.min(y_low) - buffer, np.max(y_high) + buffer),
    )
