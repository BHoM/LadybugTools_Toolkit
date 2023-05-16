import pandas as pd
import pytest
from ladybug.epw import EPW
from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
from ladybugtools_toolkit.wind.wind import DirectionBins, Wind

from . import EPW_FILE

EPW_OBJ = EPW(EPW_FILE)


def test_direction_bins():
    """."""
    db = DirectionBins()
    assert db.bin_width == 45
    assert db.directions == 8
    assert db.bins.sum() == 3240
    assert db.cardinal_directions == ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    db = DirectionBins(directions=16)
    assert db.bin_width == 22.5
    assert db.directions == 16
    assert db.bins.sum() == 6120
    assert db.cardinal_directions == [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]

    db = DirectionBins(directions=7, centered=False)
    assert db.bin_width == pytest.approx(51.42857142857143, rel=0.01)
    assert db.directions == 7
    assert db.bins.sum() == 2520.0
    assert db.cardinal_directions == ["NNE", "EbN", "SEbE", "S", "SWbW", "WbN", "NNW"]


def test_wind_good():
    """."""
    ws = collection_to_series(EPW_OBJ.wind_speed)
    wd = collection_to_series(EPW_OBJ.wind_direction)
    w = Wind(
        wind_speeds=ws.values,
        wind_directions=wd.values,
        datetimes=ws.index,
        height_above_ground=10,
    )
    assert isinstance(w, Wind)


def test_wind_bad():
    """."""
    ws = collection_to_series(EPW_OBJ.wind_speed)
    wd = collection_to_series(EPW_OBJ.wind_direction)
    with pytest.raises(ValueError):
        Wind(
            wind_speeds=ws.values[0:10],
            wind_directions=wd.values,
            datetimes=ws.index,
            height_above_ground=0,
        )


def test_wind_from_epw():
    """."""
    assert isinstance(Wind.from_epw(EPW_OBJ), Wind)


def test_wind_from_dataframe():
    """."""
    ws = collection_to_series(EPW_OBJ.wind_speed)
    wd = collection_to_series(EPW_OBJ.wind_direction)
    df = pd.concat([ws, wd], axis=1, keys=["speed", "direction"])

    assert isinstance(
        Wind.from_dataframe(
            df=df,
            wind_speed_column="speed",
            wind_direction_column="direction",
            height_above_ground=10,
        ),
        Wind,
    )


def test_wind_from_openmeteo():
    """."""
    assert isinstance(
        Wind.from_openmeteo(
            latitude=51.5074,
            longitude=0.1278,
            start_date="2020-01-01",
            end_date="2020-01-02",
        ),
        Wind,
    )
