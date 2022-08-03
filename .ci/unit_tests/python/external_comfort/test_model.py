import pytest
from ladybugtools_toolkit.external_comfort.materials.materials import Materials
from ladybugtools_toolkit.external_comfort.model.create_ground_zone import (
    create_ground_zone,
)
from ladybugtools_toolkit.external_comfort.model.create_model import create_model
from ladybugtools_toolkit.external_comfort.model.create_shade_valence import (
    create_shade_valence,
)
from ladybugtools_toolkit.external_comfort.model.create_shade_zone import (
    create_shade_zone,
)

from .. import IDENTIFIER


def test_create_ground_zone():
    assert create_ground_zone(Materials.ASPHALT_PAVEMENT.value).volume == 10 * 10 * 1


def test_create_ground_zone_material():
    with pytest.raises(AssertionError):
        create_ground_zone("not_a_material")


def test_create_shade_valence():
    assert create_shade_valence()[0].area == 10 * 3


def test_create_shade_zone():
    assert create_shade_zone(Materials.FABRIC.value).volume == 10 * 10 * 0.2


def test_create_shade_zone_material():
    with pytest.raises(AssertionError):
        create_shade_zone("not_a_material")


def test_create_model():
    model = create_model(
        Materials.ASPHALT_PAVEMENT.value, Materials.FABRIC.value, identifier=IDENTIFIER
    )
    assert model.identifier == IDENTIFIER
