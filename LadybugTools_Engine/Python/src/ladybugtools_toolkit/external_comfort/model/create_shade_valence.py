from typing import List

from honeybee.model import Shade
from ladybug_geometry.geometry3d import Point3D


def create_shade_valence(
    width: float = 10, depth: float = 10, shade_height: float = 3
) -> List[Shade]:
    """Create a massless shade around the location being assessed for a shaded external comfort
        condition.

    Args:
        width (float, optional):
            The width (x-dimension) of the shaded zone. Defaults to 10m.
        depth (float, optional):
            The depth (y-dimension) of the shaded zone. Defaults to 10m.
        shade_height (float, optional):
            The height of the shade. Default is 3m.

    Returns:
        List[Shade]:
            A list of shading surfaces.
    """

    shades = [
        Shade.from_vertices(
            identifier="SHADE_VALENCE_SOUTH",
            vertices=[
                Point3D(-width / 2, -depth / 2, 0),
                Point3D(-width / 2, -depth / 2, shade_height),
                Point3D(width / 2, -depth / 2, shade_height),
                Point3D(width / 2, -depth / 2, 0),
            ],
        ),
        Shade.from_vertices(
            identifier="SHADE_VALENCE_NORTH",
            vertices=[
                Point3D(width / 2, depth / 2, 0),
                Point3D(width / 2, depth / 2, shade_height),
                Point3D(-width / 2, depth / 2, shade_height),
                Point3D(-width / 2, depth / 2, 0),
            ],
        ),
        Shade.from_vertices(
            identifier="SHADE_VALENCE_WEST",
            vertices=[
                Point3D(-width / 2, depth / 2, 0),
                Point3D(-width / 2, depth / 2, shade_height),
                Point3D(-width / 2, -depth / 2, shade_height),
                Point3D(-width / 2, -depth / 2, 0),
            ],
        ),
        Shade.from_vertices(
            identifier="SHADE_VALENCE_EAST",
            vertices=[
                Point3D(width / 2, -depth / 2, 0),
                Point3D(width / 2, -depth / 2, shade_height),
                Point3D(width / 2, depth / 2, shade_height),
                Point3D(width / 2, depth / 2, 0),
            ],
        ),
    ]

    return shades
