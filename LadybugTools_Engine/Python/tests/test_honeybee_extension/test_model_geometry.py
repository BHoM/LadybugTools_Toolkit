from ladybug_geometry.geometry3d import Vector3D, Point3D, Plane

from ladybugtools_toolkit.honeybee_extension.model import (
    plane_intersections,
    HbModelGeometry,
    linesegments_to_polycollection,
    slice_geometry,
    slice_model,
    PolyCollection,
)
from matplotlib import pyplot as plt

from .. import TEST_DAYLIGHT_MODEL


def test_slice_model():
    """_"""
    assert isinstance(
        slice_model(
            model=TEST_DAYLIGHT_MODEL,
            plane=Plane(o=Point3D(0, 0, 1), n=Vector3D(0, 0, 1)),
        ),
        plt.Axes,
    )


def test_slice_geometry():
    """_"""
    assert isinstance(
        slice_geometry(
            hb_objects=TEST_DAYLIGHT_MODEL.faces,
            plane=Plane(o=Point3D(0, 0, 1), n=Vector3D(0, 0, 1)),
        ),
        plt.Axes,
    )


def test_plane_intersections():
    """_"""
    assert (
        len(
            plane_intersections(
                [i.geometry for i in TEST_DAYLIGHT_MODEL.faces],
                plane=Plane(o=Point3D(0, 0, 2), n=Vector3D(0, 1, 1)),
            )
        )
        == 8
    )


def test_linesegments_to_polycollection():
    """_"""
    assert isinstance(
        linesegments_to_polycollection(
            plane_intersections(
                [i.geometry for i in TEST_DAYLIGHT_MODEL.faces],
                plane=Plane(o=Point3D(0, 0, 2), n=Vector3D(0, 1, 1)),
            )
        ),
        PolyCollection,
    )


def test_model_get_geometry():
    """_"""

    assert len(HbModelGeometry.AIRBOUNDARY.get_geometry(TEST_DAYLIGHT_MODEL)) == 0
    assert len(HbModelGeometry.APERTURE.get_geometry(TEST_DAYLIGHT_MODEL)) == 32
    assert len(HbModelGeometry.DOOR.get_geometry(TEST_DAYLIGHT_MODEL)) == 0
    assert len(HbModelGeometry.FLOOR.get_geometry(TEST_DAYLIGHT_MODEL)) == 8
    assert len(HbModelGeometry.ROOFCEILING.get_geometry(TEST_DAYLIGHT_MODEL)) == 8
    assert len(HbModelGeometry.SHADE.get_geometry(TEST_DAYLIGHT_MODEL)) == 192
    assert len(HbModelGeometry.WALL.get_geometry(TEST_DAYLIGHT_MODEL)) == 32

    assert HbModelGeometry.WALL.alpha == 1
    assert HbModelGeometry.WALL.edgecolor == "#E6B43C"
    assert HbModelGeometry.WALL.facecolor == "#E6B43C"
    assert HbModelGeometry.WALL.linewidth == 1
    assert HbModelGeometry.WALL.zorder == 6

    assert (
        len(
            HbModelGeometry.WALL.slice_polycollection(
                model=TEST_DAYLIGHT_MODEL,
                plane=Plane(o=Point3D(0, 0, 3), n=Vector3D(0, 0, 1)),
            ).get_paths()
        )
        == 32
    )
    assert (
        len(
            HbModelGeometry.APERTURE.slice_polycollection(
                model=TEST_DAYLIGHT_MODEL,
                plane=Plane(o=Point3D(0, 0, 2), n=Vector3D(0, 1, 1)),
            ).get_paths()
        )
        == 6
    )
