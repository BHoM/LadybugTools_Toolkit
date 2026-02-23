import copy
from typing import Any, Union

import numpy as np
from ladybug_geometry.geometry2d import Mesh2D
from ladybug_geometry.geometry3d import Mesh3D
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation


def estimate_mesh_spacing(mesh: Mesh3D) -> float:
    if not isinstance(mesh, Mesh3D):
        raise TypeError("mesh must be a Mesh3D object")

    x = abs(
        np.diff(
            np.array([i.to_array() for i in mesh.face_centroids])[:, :-1], axis=0
        ).flatten()
    )

    vals, counts = np.unique(x[x != 0], return_counts=True)
    return float(vals[np.argmax(counts)])


def create_triangulation(
    mesh: Mesh3D,
    max_iterations: int = 250,
    increment: float = 0.01,
) -> Triangulation:
    """Create a matplotlib Triangulation from a mesh.

    Returns:
        Triangulation:
            A matplotlib Triangulation object.

    """
    # check that mesh is triangulated
    if any([len(i) > 3 for i in mesh.faces]):
        raise ValueError("Mesh must be triangulated")

    alpha = np.sqrt((estimate_mesh_spacing(mesh) ** 2) * 2)

    xs = np.array([pt.x for pt in mesh.face_centroids])
    ys = np.array([pt.y for pt in mesh.face_centroids])

    tri = Triangulation(x=xs, y=ys)

    x_tri = xs[tri.triangles] - np.roll(xs[tri.triangles], 1, axis=1)
    y_tri = ys[tri.triangles] - np.roll(ys[tri.triangles], 1, axis=1)
    maxi = np.max(np.sqrt(x_tri**2 + y_tri**2), axis=1)

    # Iterate triangulation masking until a possible mask is found
    count = 0
    fig, ax = plt.subplots(1, 1)
    synthetic_values = range(len(xs))
    success = False
    while not success:
        count += 1
        try:
            tr = copy.deepcopy(tri)
            tr.set_mask(maxi > alpha)
            ax.tricontour(tr, synthetic_values)
            success = True
        except ValueError:
            alpha += increment
        else:
            break
        if count > max_iterations:
            plt.close(fig)
            raise ValueError(
                f"Could not create a valid triangulation mask within {max_iterations}"
            )
    plt.close(fig)
    tri.set_mask(maxi > alpha)
    return tri


def triangulate_mesh3d(
    mesh: Mesh3D, aligned_data: Union[list[Any], None] = None
) -> tuple[Mesh3D, Union[list[Any], None]]:
    """Convert a mesh to a triangulated mesh.

    Args:
    mesh: Mesh3D
        The mesh to triangulate.
    aligned_data : list, optional
        Optional data to be used for the triangulated faces.
        This is one-value-per-face.
        Defaults to None, which will return an array of ones.

    Returns
    -------
    tuple[Mesh3D, np.array]:
        - The triangulated mesh.
        - The aligned data for the triangulated faces.

    """
    if not isinstance(mesh, Mesh3D):
        raise ValueError("mesh must be a Mesh3D object")

    if aligned_data is None:
        aligned_data = np.ones_like(mesh.faces)

    # find faces that are quads
    is_quad_face = [len(i) == 4 for i in mesh.faces]
    if sum(is_quad_face) == 0:
        return mesh, aligned_data

    tri_data = []
    if aligned_data is not None:
        if len(aligned_data) != len(is_quad_face):
            raise ValueError(
                f"aligned_data must be the same length as the number of faces in the mesh ({len(aligned_data)} != {len(mesh.faces)})"
            )

        # reshape the aligned_data to duplicate values being converted to tri's
        for is_quad, val in zip(*[is_quad_face, aligned_data]):
            if is_quad:
                tri_data.extend([val, val])
            else:
                tri_data.append(val)

    # for each quad face, create two triangles
    tri_faces = []
    for is_quad, face in zip(*[is_quad_face, mesh.faces]):
        if is_quad:
            tri_faces.append(tuple([face[0], face[1], face[2]]))
            tri_faces.append(tuple([face[0], face[2], face[3]]))
        else:
            tri_faces.append(face)

    return Mesh3D(vertices=mesh.vertices, faces=tri_faces), np.array(tri_data)


def triangulate_mesh2d(
    mesh: Mesh2D, aligned_data: Union[list[Any], None] = None
) -> tuple[Mesh2D, Union[list[Any], None]]:
    """Convert a 2D mesh to a triangulated mesh.

    Args:
        mesh:
            The mesh to triangulate.
        aligned_data:
            Optional data to be used for the triangulated faces.
            This is one-value-per-face.
            Defaults to None, which will return an array of ones.

    Returns:
        - The triangulated mesh.
        - The aligned data for the triangulated faces.

    """
    # FIXME - untested!!!!!!!!!
    if not isinstance(mesh, Mesh2D):
        raise ValueError("mesh must be a Mesh2D object")

    if aligned_data is None:
        aligned_data = np.ones_like(mesh.faces)

    # find faces that are quads
    is_quad_face = [len(i) == 4 for i in mesh.faces]
    if sum(is_quad_face) == 0:
        return mesh, aligned_data

    tri_data = []
    if aligned_data is not None:
        if len(aligned_data) != len(is_quad_face):
            raise ValueError(
                f"aligned_data must be the same length as the number of faces in the mesh ({len(aligned_data)} != {len(mesh.faces)})"
            )
        # reshape the aligned_data to duplicate values being converted to tri's
        for is_quad, val in zip(is_quad_face, aligned_data):
            if is_quad:
                tri_data.extend([val, val])
            else:
                tri_data.append(val)

    # for each quad face, create two triangles
    tri_faces = []
    for is_quad, face in zip(is_quad_face, mesh.faces):
        if is_quad:
            tri_faces.append(tuple([face[0], face[1], face[2]]))
            tri_faces.append(tuple([face[0], face[2], face[3]]))
        else:
            tri_faces.append(face)

    return Mesh2D(vertices=mesh.vertices, faces=tri_faces), np.array(tri_data)
