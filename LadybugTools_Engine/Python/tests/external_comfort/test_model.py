import pytest
from honeybee_energy.material.opaque import EnergyMaterial
from ladybugtools_toolkit.external_comfort._model import (
    _create_ground_zone,
    _create_shade_valence,
    _create_shade_zone,
    create_model,
    single_layer_construction,
)
from ladybugtools_toolkit.external_comfort.material import Materials

from .. import BASE_IDENTIFIER

GROUND_MATERIAL = Materials.LBT_AsphaltPavement.value
SHADE_MATERIAL = Materials.FABRIC.value


def test_single_layer_construction():
    """_"""
    material = EnergyMaterial(
        identifier="TEST_MATERIAL",
        thickness=1,
        conductivity=1,
        density=1,
        specific_heat=500,
    )
    construction = single_layer_construction(material)
    assert construction.identifier == "TEST_MATERIAL"
    assert construction.materials[0].identifier == "TEST_MATERIAL"


def test_create_ground_zone():
    """_"""
    assert (
        _create_ground_zone(single_layer_construction(GROUND_MATERIAL.to_lbt())).volume
        == 10 * 10 * 1
    )


def test_create_ground_zone_material():
    """_"""
    with pytest.raises(AssertionError):
        _create_ground_zone("not_a_material")


def test_create_shade_valence():
    """_"""
    assert (
        _create_shade_valence(single_layer_construction(SHADE_MATERIAL.to_lbt()))[
            0
        ].area
        == 10 * 3
    )


def test_create_shade_zone():
    """_"""
    assert (
        _create_shade_zone(single_layer_construction(SHADE_MATERIAL.to_lbt())).volume
        == 10 * 10 * 0.2
    )


def test_create_shade_zone_construction():
    """_"""
    with pytest.raises(AssertionError):
        _create_shade_zone("not_a_construction")


def test_create_model():
    """_"""
    model = create_model(
        GROUND_MATERIAL.to_lbt(),
        SHADE_MATERIAL.to_lbt(),
        identifier=BASE_IDENTIFIER,
    )
    assert model.identifier == BASE_IDENTIFIER
