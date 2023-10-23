import matplotlib.pyplot as plt
import pandas as pd
import pytest
from ladybug.epw import AnalysisPeriod
from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
from ladybugtools_toolkit.wind import DirectionBins, Wind

from . import EPW_OBJ


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


def test_wind_from_uv():
    """_"""
    w = Wind.from_epw(EPW_OBJ)
    u, v = w.uv.values.T
    with pytest.warns(UserWarning):
        assert isinstance(Wind.from_uv(u=u, v=v, datetimes=w.datetimes), Wind)


def test_wind_functions():
    """."""
    w = Wind.from_epw(EPW_OBJ)

    assert (
        len(w.filter_by_analysis_period(AnalysisPeriod(st_month=3, end_month=5)))
        == 2208
    )
    assert w.calm(threshold=1) == pytest.approx(0.19988584474885845, rel=0.01)
    assert w.exceedance(limit_value=3.5).values.sum() == pytest.approx(
        51.16172035633674, rel=0.01
    )
    assert len(w.filter_by_direction(left_angle=350, right_angle=94)) == 3185
    assert len(w.filter_by_speed(min_speed=0.5, max_speed=4.5)) == 5806
    assert w.frequency_table().values.sum() == 8162
    assert w.max() == 17.5
    assert w.min() == 0
    assert w.mean() == pytest.approx(3.2418835616438364, rel=0.01)
    assert w.median() == 3.1
    assert w.percentile(0.25) == 1.5
    assert w.prevailing()[0] == "SW"
    assert w.wind_matrix().shape == (24, 24)
    assert w.to_height(target_height=52).ws.sum() == pytest.approx(
        48732.60735555029, rel=0.01
    )

    db = DirectionBins(directions=4)
    assert (
        w.apply_directional_factors(direction_bins=db, factors=[0, 1, 20, 0]).ws.sum()
        == 201354.6
    )


def test_plot_windrose():
    """_"""
    w = Wind.from_epw(EPW_OBJ)
    assert isinstance(w.plot_windrose(), plt.Axes)
    plt.close("all")


def test_plot_timeseries():
    """_"""
    w = Wind.from_epw(EPW_OBJ)
    assert isinstance(w.plot_timeseries(), plt.Axes)
    plt.close("all")


def test_plot_windhist():
    """_"""
    w = Wind.from_epw(EPW_OBJ)
    assert isinstance(w.plot_windhist(), plt.Axes)
    plt.close("all")


def test_plot_windhist_radial():
    """_"""
    w = Wind.from_epw(EPW_OBJ)
    assert isinstance(w.plot_windhist_radial(), plt.Axes)
    plt.close("all")


def test_plot_wind_matrix():
    """_"""
    w = Wind.from_epw(EPW_OBJ)
    assert isinstance(w.plot_wind_matrix(), plt.Axes)
    plt.close("all")


def test_plot_speed_frequency():
    """_"""
    w = Wind.from_epw(EPW_OBJ)
    assert isinstance(w.plot_speed_frequency(), plt.Axes)
    plt.close("all")


def test_plot_cumulative_density():
    """_"""
    w = Wind.from_epw(EPW_OBJ)
    assert isinstance(w.plot_cumulative_density(), plt.Axes)
    plt.close("all")
