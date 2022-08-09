import pandas as pd
import pytest
from ladybugtools_toolkit.ladybug_extension.datacollection.from_series import (
    from_series,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series

from .. import EPW_OBJ

SERIES_GOOD = pd.Series(
    data=EPW_OBJ.dry_bulb_temperature.values,
    name="Dry Bulb Temperature (C)",
    index=pd.date_range("2007-01-01 00:30:00", freq="60T", periods=8760),
)


def test_to_series():
    assert to_series(EPW_OBJ.dry_bulb_temperature).mean() == pytest.approx(
        EPW_OBJ.dry_bulb_temperature.average, rel=0.01
    )


def test_from_series_good_1():
    assert from_series(SERIES_GOOD).average == pytest.approx(
        EPW_OBJ.dry_bulb_temperature.average, rel=0.01
    )


def test_from_series_good_2():
    assert from_series(SERIES_GOOD.resample("MS").mean()).average == pytest.approx(
        EPW_OBJ.dry_bulb_temperature.average, rel=0.01
    )


def test_from_series_bad_1():
    with pytest.raises(ValueError):
        from_series(SERIES_GOOD.sample(4000))
