import pandas as pd
import pytest
from ladybug.epw import EPW
from ladybugtools_toolkit.ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)

from .. import EPW_OBJ

SERIES_GOOD = pd.Series(
    data=EPW_OBJ.dry_bulb_temperature.values,
    name="Dry Bulb Temperature (C)",
    index=pd.date_range("2007-01-01 00:30:00", freq="60min", periods=8760),
)


def test_to_series():
    """_"""
    assert collection_to_series(EPW_OBJ.dry_bulb_temperature).mean() == pytest.approx(
        EPW_OBJ.dry_bulb_temperature.average, rel=0.01
    )


def test_from_series_good_1():
    """_"""
    assert collection_from_series(SERIES_GOOD).average == pytest.approx(
        EPW_OBJ.dry_bulb_temperature.average, rel=0.01
    )


def test_from_series_good_2():
    """_"""
    assert collection_from_series(
        SERIES_GOOD.resample("MS").mean()
    ).average == pytest.approx(EPW_OBJ.dry_bulb_temperature.average, rel=0.01)


def test_from_series_bad_1():
    """_"""
    with pytest.raises(ValueError):
        collection_from_series(SERIES_GOOD.sample(4000))
