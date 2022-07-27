from ladybugtools_toolkit.external_comfort.ground_temperature.ground_temperature_at_depth import (
    ground_temperature_at_depth,
)

from .. import EPW_OBJ


def test_ground_temperature_at_depth():
    assert ground_temperature_at_depth(EPW_OBJ, 0.5).average == 10.108297296260833
