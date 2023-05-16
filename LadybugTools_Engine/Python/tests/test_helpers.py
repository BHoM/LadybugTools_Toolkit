import pandas as pd
import pytest
from ladybug.epw import EPW
from ladybugtools_toolkit.helpers import (
    angle_from_north,
    cardinality,
    decay_rate_smoother,
    evaporative_cooling_effect,
    evaporative_cooling_effect_collection,
    proximity_decay,
    target_wind_speed_collection,
    wind_speed_at_height,
)

from . import EPW_FILE

EPW_OBJ = EPW(EPW_FILE)


def test_angle_from_north():
    """_"""
    assert angle_from_north([0.5, 0.5]) == 45


def test_cardinality_good():
    """_"""
    assert cardinality(22.5, directions=16) == "NNE"


def test_cardinality_bad():
    """_"""
    with pytest.raises(ValueError):
        cardinality(370, directions=16)


def test_decay_rate_smoother():
    """_"""
    s = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    assert decay_rate_smoother(
        s, difference_threshold=2, transition_window=4, ewm_span=4
    ).sum() == pytest.approx(99.37681225198287, rel=0.1)


def test_proximity_decay_good():
    """_"""
    assert (
        proximity_decay(
            value=10,
            distance_to_value=5,
            max_distance=10,
            decay_method="parabolic",
        )
        == 7.5
    )


def test_proximity_decay_bad():
    """_"""
    with pytest.raises(ValueError):
        proximity_decay(
            value=10,
            distance_to_value=5,
            max_distance=10,
            decay_method="unknown",
        )


def test_evaporative_cooling_effect():
    """_"""
    dbt, rh = evaporative_cooling_effect(20, 50, 0.5)
    assert (dbt == pytest.approx(16.9, rel=0.1)) and (rh == pytest.approx(75, rel=0.1))


def test_evaporative_cooling_effect_collection():
    """_"""
    dbt, rh = evaporative_cooling_effect_collection(
        EPW_OBJ, evaporative_cooling_effectiveness=0.3
    )
    assert (pytest.approx(dbt.average, rel=0.1) == 9.638794225575671) and (
        pytest.approx(rh.average, rel=0.1) == 85.50440639269416
    )


def test_evaporative_cooling_effect_collection_bad():
    """_"""
    with pytest.raises(ValueError):
        evaporative_cooling_effect_collection(
            EPW_OBJ, evaporative_cooling_effectiveness=1.2
        )


def test_target_wind_speed_collection():
    """_"""
    assert target_wind_speed_collection(EPW_OBJ, 3, 10).average == pytest.approx(
        3, rel=0.0001
    )


def test_wind_speed_at_height():
    """_"""
    assert wind_speed_at_height(
        reference_value=2,
        reference_height=10,
        target_height=2,
        terrain_roughness_length=0.5,
        log_function=False,
    ) == pytest.approx(1.5891948094037045, rel=0.0001)
