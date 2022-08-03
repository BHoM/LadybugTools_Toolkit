import pytest
from ladybugtools_toolkit.external_comfort.wind.target_wind_speed_collection import (
    target_wind_speed_collection,
)
from ladybugtools_toolkit.external_comfort.wind.wind_speed_at_height import (
    wind_speed_at_height,
)

from .. import EPW_OBJ


def test_target_wind_speed_collection():
    assert target_wind_speed_collection(EPW_OBJ, 3, 10).average == pytest.approx(
        3, rel=0.0001
    )


def test_wind_speed_at_height():
    assert wind_speed_at_height(
        reference_wind_speed=2,
        reference_height=10,
        target_height=2,
        terrain_roughness_length=0.5,
        log_function=False,
    ) == pytest.approx(1.5891948094037045, rel=0.0001)
