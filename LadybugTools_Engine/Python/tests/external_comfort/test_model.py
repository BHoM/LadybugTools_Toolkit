"""Test functions for the "model" module."""

import pytest
from honeybee.model import Model
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.shade import ShadeConstruction
from ladybugtools_toolkit.new_external_comfort.material import get_material
from ladybugtools_toolkit.new_external_comfort.model import (
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


def test_single_layer_construction():
    """_"""
    material = get_material("Fabric")
    construction = single_layer_construction(material)
    assert isinstance(construction, OpaqueConstruction)
    assert construction.identifier == "Fabric"
    assert construction.materials[0].identifier == "Fabric"


def test_opaque_to_shade():
    """_"""
    opaque = single_layer_construction(get_material("Concrete Pavement"))
    shade = opaque_to_shade(opaque)
    assert isinstance(shade, ShadeConstruction)
    assert shade.identifier == "Concrete Pavement"
    assert shade.solar_reflectance == opaque.outside_solar_reflectance


def test_ground_zone():
    """_"""
    assert (
        _ground_zone(
            single_layer_construction(get_material("Concrete Pavement"))
        ).volume
        == _GROUND_THICKNESS * _ZONE_DEPTH * _ZONE_WIDTH
    )
    with pytest.raises(AssertionError):
        _ground_zone("not_a_construction")


def test_shade_valence():
    """_"""
    assert (
        _shade_valence(single_layer_construction(get_material("Fabric")))[0].area
        == _ZONE_DEPTH * _SHADE_HEIGHT_ABOVE_GROUND
    )
    with pytest.raises(AssertionError):
        _shade_valence("not_a_construction")


def test_shade_zone():
    """_"""
    assert (
        _shade_zone(single_layer_construction(get_material("Fabric"))).volume
        == _ZONE_DEPTH * _ZONE_WIDTH * _SHADE_THICKNESS
    )
    with pytest.raises(AssertionError):
        _shade_zone("not_a_construction")


def test_create_model():
    """_"""
    default_model = create_model(
        ground_material=get_material("Concrete Pavement"),
        shade_material=get_material("Fabric"),
    )
    assert isinstance(default_model, Model)
    assert default_model.identifier == "unnamed"

    named_model = create_model(
        ground_material=get_material("Concrete Pavement"),
        shade_material=get_material("Fabric"),
        identifier="test_name",
    )
    assert named_model.identifier == "test_name"


def test_get_ground_material():
    """_"""
    ground_material = get_material("Concrete Pavement")
    model = create_model(
        ground_material=ground_material,
        shade_material=get_material("Fabric"),
    )
    assert get_ground_material(model) == ground_material


def test_get_ground_reflectance():
    """_"""
    ground_material = get_material("Concrete Pavement")
    model = create_model(
        ground_material=ground_material,
        shade_material=get_material("Fabric"),
    )
    assert get_ground_reflectance(model) == ground_material.solar_reflectance


def test_get_shade_material():
    """_"""
    shade_material = get_material("Fabric")
    model = create_model(
        ground_material=get_material("Concrete Pavement"),
        shade_material=shade_material,
    )
    assert get_shade_material(model) == shade_material


def test_model_equality():
    """_"""
    model_1 = create_model(
        ground_material=get_material("Concrete Pavement"),
        shade_material=get_material("Fabric"),
    )
    model_2 = create_model(
        ground_material=get_material("Concrete Pavement"),
        shade_material=get_material("Fabric"),
        identifier="text_name",
    )
    model_3 = create_model(
        ground_material=get_material("Concrete Pavement"),
        shade_material=get_material("Concrete Pavement"),
    )

    assert model_equality(model_1, model_2, include_identifier=False)
    assert not model_equality(model_1, model_2, include_identifier=True)
    assert not model_equality(model_1, model_3, include_identifier=False)
