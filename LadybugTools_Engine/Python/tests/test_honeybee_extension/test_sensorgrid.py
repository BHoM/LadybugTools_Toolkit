import pytest
from ladybugtools_toolkit.honeybee_extension.simulation.sensorgrids import (
    is_planar,
    get_plane,
    position_array,
    vector_array,
    estimate_spacing,
    as_triangulation,
    as_patchcollection,
    plot_values,
    get_limits,
)
from matplotlib.collections import PatchCollection
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import numpy as np
from ladybug_geometry.geometry3d import Vector3D, Plane, Point3D

from .. import TEST_DAYLIGHT_MODEL

TEST_SENSORGRID = TEST_DAYLIGHT_MODEL.properties.radiance.sensor_grids[0]


def test_is_planar():
    """_"""
    assert is_planar(TEST_SENSORGRID)


def test_get_plane():
    """_"""
    assert isinstance(get_plane(TEST_SENSORGRID), Plane)


def test_position_array():
    """_"""
    assert position_array(TEST_SENSORGRID).sum() == pytest.approx(145, rel=1)
    assert position_array(TEST_SENSORGRID).shape == (25, 3)


def test_vector_array():
    """_"""
    assert vector_array(TEST_SENSORGRID).sum() == pytest.approx(25, rel=0.01)
    assert vector_array(TEST_SENSORGRID).shape == (25, 3)


def test_estimate_spacing():
    """_"""
    assert estimate_spacing(TEST_SENSORGRID) == 1


def test_as_triangulation():
    """_"""
    assert isinstance(
        as_triangulation(TEST_SENSORGRID, alpha_adjust=0.1), Triangulation
    )


def test_as_patchcollection():
    """_"""
    assert isinstance(as_patchcollection(TEST_SENSORGRID), PatchCollection)


def test_plot_values():
    """_"""
    assert isinstance(
        plot_values(sensorgrid=TEST_SENSORGRID, values=np.random.randint(0, 100, 25)),
        plt.Axes,
    )


def test_get_limits():
    """_"""
    assert get_limits(sensorgrids=[TEST_SENSORGRID]) == ((-1.5, 6.5), (-1.5, 6.5))
