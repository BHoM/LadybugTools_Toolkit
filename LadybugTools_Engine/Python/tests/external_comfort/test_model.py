import pytest
from ladybugtools_toolkit.external_comfort.material import Materials
from ladybugtools_toolkit.external_comfort.model import (
    _create_ground_zone,
    _create_shade_valence,
    _create_shade_zone,
    create_model,
)

from .. import BASE_IDENTIFIER

GROUND_MATERIAL = Materials.LBT_AsphaltPavement.value
SHADE_MATERIAL = Materials.FABRIC.value


def test_create_ground_zone():
    """_"""
    assert _create_ground_zone(GROUND_MATERIAL.to_lbt()).volume == 10 * 10 * 1


def test_create_ground_zone_material():
    """_"""
    with pytest.raises(AssertionError):
        _create_ground_zone("not_a_material")


def test_create_shade_valence():
    """_"""
    assert _create_shade_valence()[0].area == 10 * 3


def test_create_shade_zone():
    """_"""
    assert _create_shade_zone(SHADE_MATERIAL.to_lbt()).volume == 10 * 10 * 0.2


def test_create_shade_zone_material():
    """_"""
    with pytest.raises(AssertionError):
        _create_shade_zone("not_a_material")


def test_create_model():
    """_"""
    model = create_model(
        GROUND_MATERIAL.to_lbt(),
        SHADE_MATERIAL.to_lbt(),
        identifier=BASE_IDENTIFIER,
    )
    assert model.identifier == BASE_IDENTIFIER
