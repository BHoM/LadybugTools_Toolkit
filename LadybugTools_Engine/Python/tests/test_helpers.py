import pandas as pd
import pytest
from ladybugtools_toolkit.helpers import (
    angle_from_north,
    cardinality,
    decay_rate_smoother,
    proximity_decay,
)


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
