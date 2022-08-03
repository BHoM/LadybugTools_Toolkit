import pytest
from ladybugtools_toolkit.external_comfort.moisture.evaporative_cooling_effect import (
    evaporative_cooling_effect,
)
from ladybugtools_toolkit.external_comfort.moisture.evaporative_cooling_effect_collection import (
    evaporative_cooling_effect_collection,
)

from .. import EPW_OBJ


def test_evaporative_cooling_effect():
    dbt, rh = evaporative_cooling_effect(20, 50, 0.5)
    assert (dbt == pytest.approx(16.9, rel=0.1)) and (rh == pytest.approx(75, rel=0.1))


def test_evaporative_cooling_effect_collection():
    dbt, rh = evaporative_cooling_effect_collection(
        EPW_OBJ, evaporative_cooling_effectiveness=0.3
    )
    assert (pytest.approx(dbt.average, rel=0.1) == 9.638794225575671) and (
        pytest.approx(rh.average, rel=0.1) == 85.50440639269416
    )


def test_evaporative_cooling_effect_collection_bad():
    with pytest.raises(ValueError):
        evaporative_cooling_effect_collection(
            EPW_OBJ, evaporative_cooling_effectiveness=1.2
        )
