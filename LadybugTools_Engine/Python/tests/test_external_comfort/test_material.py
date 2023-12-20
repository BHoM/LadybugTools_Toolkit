"""Test functions for the "material" module."""

import numpy as np
import pytest
from ladybugtools_toolkit.external_comfort.material import (
    _custom_materials,
    _ice_tool_materials,
    _lbt_materials,
    get_material,
    materials,
)

TEST_GROUND_MATERIAL = get_material("Concrete Pavement")
TEST_SHADE_MATERIAL = get_material("Fabric")


def test_lbt_materials():
    """_"""
    _materials = _lbt_materials()
    assert len(_materials) == 114
    assert _materials[0].identifier == "Generic Roof Membrane"
    assert _materials[-1].identifier == "Grassy Lawn"
    assert np.array([i.conductivity for i in _materials]).mean() == pytest.approx(
        4.06154874650346
    )


def test_ice_tool_materials():
    """_"""
    _materials = _ice_tool_materials()
    assert len(_materials) == 39
    assert _materials[0].identifier == "SD1 - Quartzite (Beige/brown/black New/Rough)"
    assert _materials[-1].identifier == "WT1 - Water small (- -)"
    assert np.array([i.conductivity for i in _materials]).mean() == pytest.approx(
        1.498974358974359
    )


def test_custom_materials():
    """_"""
    _materials = _custom_materials()
    assert len(_materials) == 3
    assert _materials[0].identifier == "Fabric"
    assert _materials[-1].identifier == "Travertine"
    assert np.array([i.conductivity for i in _materials]).mean() == pytest.approx(
        1.2033333333333334
    )


def test_materials():
    """_"""
    _materials = materials()
    assert len(_materials) == 154
    assert _materials[0].identifier == "Generic Roof Membrane"
    assert _materials[-1].identifier == "WT1 - Water small (- -)"
    assert np.array([i.conductivity for i in _materials]).mean() == pytest.approx(
        3.4017114097493146
    )


def test_get_material():
    """_"""
    with pytest.raises(KeyError):
        assert "Did you mean " in get_material("Dry Grass")

    assert get_material("Dry Sand").identifier == "Dry Sand"
    assert get_material("M01 100mm brick").specific_heat == pytest.approx(
        789.4906206515565
    )
