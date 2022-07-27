import pytest
from ladybugtools_toolkit.external_comfort.moisture.evaporative_cooling_effect import (
    evaporative_cooling_effect,
)


def test_evaporative_cooling_effect():
    dbt, rh = evaporative_cooling_effect(20, 50, 0.5)
    assert (dbt == pytest.approx(16.9, rel=0.1)) and (rh == pytest.approx(75, rel=0.1))
