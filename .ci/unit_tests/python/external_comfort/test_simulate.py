import pytest
from ladybugtools_toolkit.external_comfort.materials.materials import Materials
from ladybugtools_toolkit.external_comfort.model.create_model import create_model
from ladybugtools_toolkit.external_comfort.simulate.longwave_mean_radiant_temperature import (
    longwave_mean_radiant_temperature,
)
from ladybugtools_toolkit.external_comfort.simulate.solar_radiation import (
    solar_radiation,
)
from ladybugtools_toolkit.external_comfort.simulate.surface_temperature import (
    surface_temperature,
)

from .. import EPW_OBJ, GROUND_MATERIAL, IDENTIFIER, SHADE_MATERIAL

model = create_model(
    ground_material=GROUND_MATERIAL,
    shade_material=SHADE_MATERIAL,
    identifier=IDENTIFIER,
)


def test_solar_radiation():
    result = solar_radiation(model, EPW_OBJ)
    assert result["unshaded_total_radiation"].average == pytest.approx(203, rel=1.5)


def test_surface_temperature():
    result = surface_temperature(model, EPW_OBJ)
    assert (
        result["unshaded_above_temperature"].average == EPW_OBJ.sky_temperature.average
    )


def test_longwave_mean_radiant_temperature():
    assert (
        longwave_mean_radiant_temperature(
            [EPW_OBJ.dry_bulb_temperature * 0.5, EPW_OBJ.dry_bulb_temperature * 2],
            view_factors=[0.25, 0.75],
        ).average
        == 16.924573102344482
    )
