"""Methods for handling Ladybug geometry."""
from warnings import warn  # pylint: disable=E0401

from ladybug_geometry.bounding import bounding_rectangle
from ladybug_geometry.geometry3d import Plane, Point3D, Vector3D, Mesh3D
from ladybug_geometry.geometry2d import Mesh2D, Point2D, Polygon2D
from scipy.spatial.distance import cdist

from .util import (
    mesh3d_isplanar,
    mesh3d_get_plane,
    pt_distances
)
