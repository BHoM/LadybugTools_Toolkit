"""Test functions for the "model" module."""

import pytest
from honeybee.model import Model
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.shade import ShadeConstruction
from ladybugtools_toolkit.external_comfort.model import (
    _GROUND_THICKNESS,
    _SHADE_HEIGHT_ABOVE_GROUND,
    _SHADE_THICKNESS,
    _ZONE_DEPTH,
    _ZONE_WIDTH,
    _ground_zone,
    _shade_valence,
    _shade_zone,
    create_model,
    get_ground_material,
    get_ground_reflectance,
    get_shade_material,
    model_equality,
    opaque_to_shade,
    single_layer_construction,
)

from .test_material import TEST_GROUND_MATERIAL, TEST_SHADE_MATERIAL


def test_single_layer_construction():
    """_"""
    construction = single_layer_construction(TEST_SHADE_MATERIAL)
    assert isinstance(construction, OpaqueConstruction)
    assert construction.identifier == TEST_SHADE_MATERIAL.identifier
    assert construction.materials[0].identifier == TEST_SHADE_MATERIAL.identifier

    with pytest.raises(TypeError):
        single_layer_construction("not_a_material")


def test_opaque_to_shade():
    """_"""
    opaque = single_layer_construction(TEST_GROUND_MATERIAL)
    shade = opaque_to_shade(opaque)
    assert isinstance(shade, ShadeConstruction)
    assert shade.identifier == f"{TEST_GROUND_MATERIAL.identifier}_shade"
    assert shade.solar_reflectance == opaque.outside_solar_reflectance


def test_ground_zone():
    """_"""
    assert (
        _ground_zone(single_layer_construction(TEST_GROUND_MATERIAL)).volume
        == _GROUND_THICKNESS * _ZONE_DEPTH * _ZONE_WIDTH
    )


def test_ground_zone_bad():
    """_"""
    with pytest.raises((AssertionError, TypeError)):
        _ground_zone("not_a_construction")


def test_shade_valence():
    """_"""
    assert (
        _shade_valence(single_layer_construction(TEST_SHADE_MATERIAL))[0].area
        == _ZONE_DEPTH * _SHADE_HEIGHT_ABOVE_GROUND
    )


def test_shade_valence_bad():
    with pytest.raises((AssertionError, TypeError)):
        _shade_valence("not_a_construction")


def test_shade_zone():
    """_"""
    assert (
        _shade_zone(single_layer_construction(TEST_SHADE_MATERIAL)).volume
        == _ZONE_DEPTH * _ZONE_WIDTH * _SHADE_THICKNESS
    )


def test_shade_zone_bad():
    with pytest.raises((AssertionError, TypeError)):
        _shade_zone("not_a_construction")


def test_create_model():
    """_"""
    default_model = create_model(
        identifier="test_name",
        ground_material=TEST_GROUND_MATERIAL,
        shade_material=TEST_SHADE_MATERIAL,
    )
    assert isinstance(default_model, Model)
    assert default_model.identifier == "test_name"


def test_get_ground_material():
    """_"""
    model = create_model(
        identifier="test_name",
        ground_material=TEST_GROUND_MATERIAL,
        shade_material=TEST_SHADE_MATERIAL,
    )
    assert get_ground_material(model) == TEST_GROUND_MATERIAL

    with pytest.raises(TypeError):
        get_ground_material("not_a_model")


def test_get_ground_reflectance():
    """_"""
    model = create_model(
        identifier="test_name",
        ground_material=TEST_GROUND_MATERIAL,
        shade_material=TEST_SHADE_MATERIAL,
    )
    assert get_ground_reflectance(model) == TEST_GROUND_MATERIAL.solar_reflectance

    with pytest.raises(TypeError):
        get_ground_reflectance("not_a_model")


def test_get_shade_material():
    """_"""
    model = create_model(
        identifier="test_name",
        ground_material=TEST_GROUND_MATERIAL,
        shade_material=TEST_SHADE_MATERIAL,
    )
    assert get_shade_material(model) == TEST_SHADE_MATERIAL

    with pytest.raises(TypeError):
        get_shade_material("not_a_model")


def test_model_equality():
    """_"""
    model_1 = create_model(
        identifier="test_name",
        ground_material=TEST_GROUND_MATERIAL,
        shade_material=TEST_SHADE_MATERIAL,
    )
    model_2 = create_model(
        identifier="other_name",
        ground_material=TEST_GROUND_MATERIAL,
        shade_material=TEST_SHADE_MATERIAL,
    )
    model_3 = create_model(
        identifier="test_name",
        ground_material=TEST_GROUND_MATERIAL,
        shade_material=TEST_GROUND_MATERIAL,
    )
    model_4 = create_model(
        identifier="test_name",
        ground_material=TEST_SHADE_MATERIAL,
        shade_material=TEST_SHADE_MATERIAL,
    )
    model_5 = create_model(
        identifier="test_name",
        ground_material=TEST_SHADE_MATERIAL,
        shade_material=TEST_GROUND_MATERIAL,
    )

    with pytest.raises(TypeError):
        model_equality("not_a_model", model_1)

    assert model_equality(model_1, model_1)  # equal
    assert not model_equality(model_1, model_2)  # identifier different
    assert not model_equality(model_1, model_4)  # ground material different
    assert not model_equality(model_1, model_3)  # shade material different
    assert not model_equality(model_1, model_5)  # both materials different
