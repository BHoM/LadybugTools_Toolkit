"""Methods for handling Ladybug geometry."""
from warnings import warn  # pylint: disable=E0401

from ladybug_geometry.bounding import bounding_rectangle
from ladybug_geometry.geometry3d import Plane, Point3D, Vector3D, Mesh3D
from ladybug_geometry.geometry2d import Mesh2D, Point2D


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
