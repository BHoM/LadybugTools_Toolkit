import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from typing import List

from external_comfort.model import _DEPTH, _SHADE_HEIGHT_ABOVE_GROUND, _WIDTH
from honeybee.model import Shade
from ladybug_geometry.geometry3d import Point3D


def create_shade_valence() -> List[Shade]:
    """Create a massless shade around the location being assessed for a shaded external comfort condition.

    Returns:
        List[Shade]: A list of shading surfaces.
    """

    shades = [
        Shade.from_vertices(
            identifier=f"SHADE_VALENCE_SOUTH",
            vertices=[
                Point3D(-_WIDTH / 2, -_DEPTH / 2, 0),
                Point3D(-_WIDTH / 2, -_DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(_WIDTH / 2, -_DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(_WIDTH / 2, -_DEPTH / 2, 0),
            ],
        ),
        Shade.from_vertices(
            identifier=f"SHADE_VALENCE_NORTH",
            vertices=[
                Point3D(_WIDTH / 2, _DEPTH / 2, 0),
                Point3D(_WIDTH / 2, _DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(-_WIDTH / 2, _DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(-_WIDTH / 2, _DEPTH / 2, 0),
            ],
        ),
        Shade.from_vertices(
            identifier=f"SHADE_VALENCE_WEST",
            vertices=[
                Point3D(-_WIDTH / 2, _DEPTH / 2, 0),
                Point3D(-_WIDTH / 2, _DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(-_WIDTH / 2, -_DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(-_WIDTH / 2, -_DEPTH / 2, 0),
            ],
        ),
        Shade.from_vertices(
            identifier=f"SHADE_VALENCE_EAST",
            vertices=[
                Point3D(_WIDTH / 2, -_DEPTH / 2, 0),
                Point3D(_WIDTH / 2, -_DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(_WIDTH / 2, _DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(_WIDTH / 2, _DEPTH / 2, 0),
            ],
        ),
    ]

    return shades
