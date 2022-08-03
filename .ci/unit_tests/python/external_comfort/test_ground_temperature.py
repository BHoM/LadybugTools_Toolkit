from ladybugtools_toolkit.external_comfort.ground_temperature.ground_temperature_at_depth import (
    ground_temperature_at_depth,
)
from ladybugtools_toolkit.external_comfort.ground_temperature.hourly_ground_temperature import (
    hourly_ground_temperature,
)

from .. import EPW_OBJ


def test_ground_temperature_at_depth():
    assert ground_temperature_at_depth(EPW_OBJ, 0.5).average == 10.108297296260833


def test_hourly_ground_temperature():
    hourly_ground_temperature(EPW_OBJ)
