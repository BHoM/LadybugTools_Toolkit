import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tempfile import gettempdir
import numpy as np
import pytest
from ladybug.epw import AnalysisPeriod
from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
from ladybugtools_toolkit.wind import DirectionBins, Wind

from . import EPW_OBJ, EPW_FILE

TEST_WIND = Wind.from_epw(EPW_OBJ)


def test_round_trip():
    """_"""
    tempfile = Path(gettempdir()) / "pytest_wind.json"
    Wind.from_dict(TEST_WIND.to_dict())
    Wind.from_json(TEST_WIND.to_json())
    Wind.from_file(TEST_WIND.to_file(tempfile))

    tempfile.unlink()

    with pytest.raises(ValueError):
        TEST_WIND.to_file("./not_a_json_file.txt")


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
            wind_speeds=ws.values,
            wind_directions=wd.values,
            datetimes=ws.index,
            height_above_ground=0,
        )
        Wind(
            wind_speeds=ws.values[0:10],
            wind_directions=wd.values,
            datetimes=ws.index,
            height_above_ground=10,
        )
        Wind(
            wind_speeds=ws.values[0:1],
            wind_directions=wd.values[0:1],
            datetimes=ws.index,
            height_above_ground=10,
        )
        Wind(
            wind_speeds=[1, 1, 1, 1, 1],
            wind_directions=[1, 1, 1, 1, 1],
            datetimes=[ws.index[0]] * 5,
            height_above_ground=10,
        )
        Wind(
            wind_speeds=np.where(ws < 5, np.nan, ws),
            wind_directions=wd.values,
            datetimes=ws.index,
            height_above_ground=0,
        )
        Wind(
            wind_speeds=ws.values,
            wind_directions=np.where(wd < 180, np.nan, wd),
            datetimes=ws.index,
            height_above_ground=0,
        )
        Wind(
            wind_speeds=ws.values,
            wind_directions=np.where(wd < 180, -1, wd),
            datetimes=ws.index,
            height_above_ground=0,
        )


def test_wind_from_epw():
    """."""
    assert isinstance(Wind.from_epw(EPW_OBJ), Wind)
    assert isinstance(Wind.from_epw(EPW_FILE), Wind)
    assert Wind.from_epw(EPW_FILE).source == EPW_FILE.name


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

    with pytest.raises(ValueError):
        Wind.from_dataframe(
            df="not_a_dataframe",
            wind_speed_column="speed",
            wind_direction_column="direction",
            height_above_ground=10,
        )
        Wind.from_dataframe(
            df=df.reset_index(drop=True),
            wind_speed_column="speed",
            wind_direction_column="direction",
            height_above_ground=10,
        )


def test_wind_from_openmeteo():
    """."""
    w = Wind.from_openmeteo(
        latitude=51.5074,
        longitude=0.1278,
        start_date="2020-01-01",
        end_date="2020-01-02",
    )
    assert isinstance(
        w,
        Wind,
    )
    assert w.source == "OpenMeteo"


def test_wind_from_uv():
    """_"""
    u, v = TEST_WIND.uv.values.T
    with pytest.warns(UserWarning):
        assert isinstance(Wind.from_uv(u=u, v=v, datetimes=TEST_WIND.datetimes), Wind)


def test_wind_functions():
    """_"""
    assert isinstance(TEST_WIND.calm_datetimes, list)
    assert TEST_WIND.calm(threshold=1) == pytest.approx(0.19988584474885845, rel=0.01)
    assert TEST_WIND.exceedance(limit_value=3.5).values.sum() == pytest.approx(
        51.16172035633674, rel=0.01
    )
    assert TEST_WIND.frequency_table().values.sum() == 7979
    assert TEST_WIND.max() == 17.5
    assert TEST_WIND.min() == 0
    assert TEST_WIND.mean() == pytest.approx(3.2418835616438364, rel=0.01)
    assert TEST_WIND.median() == 3.1
    assert TEST_WIND.percentile(0.25) == 1.5
    assert TEST_WIND.prevailing()[0] == "SW"
    assert TEST_WIND.wind_matrix().shape == (24, 24)
    assert TEST_WIND.to_height(target_height=52).ws.sum() == pytest.approx(
        48732.60735555029, rel=0.01
    )

    db = DirectionBins(directions=4)
    assert (
        TEST_WIND.apply_directional_factors(
            direction_bins=db, factors=[0, 1, 20, 0]
        ).ws.sum()
        == 201354.6
    )


def test_filters():
    """_"""
    assert (
        len(
            TEST_WIND.filter_by_analysis_period(AnalysisPeriod(st_month=3, end_month=5))
        )
        == 2208
    )
    assert len(TEST_WIND.filter_by_direction(left_angle=350, right_angle=94)) == 3185
    assert len(TEST_WIND.filter_by_speed(min_speed=0.5, max_speed=4.5)) == 5806
    assert (
        len(
            TEST_WIND.filter_by_boolean_mask(
                [i % 25 == 0 for i in range(len(TEST_WIND))]
            )
        )
        == 351
    )
    assert (
        len(TEST_WIND.filter_by_time(months=[3, 4], hours=[4, 6, 19, 21], years=[2017]))
        == 244
    )


def test_plot_windrose():
    """_"""
    assert isinstance(TEST_WIND.plot_windrose(), plt.Axes)
    plt.close("all")


def test_plot_timeseries():
    """_"""
    assert isinstance(TEST_WIND.plot_timeseries(), plt.Axes)
    plt.close("all")


def test_plot_windhist():
    """_"""
    assert isinstance(TEST_WIND.plot_windhist(), plt.Axes)
    plt.close("all")


def test_plot_windhist_radial():
    """_"""
    assert isinstance(TEST_WIND.plot_windhist_radial(), plt.Axes)
    plt.close("all")


def test_plot_wind_matrix():
    """_"""
    assert isinstance(TEST_WIND.plot_wind_matrix(), plt.Axes)
    plt.close("all")


def test_plot_speed_frequency():
    """_"""
    assert isinstance(TEST_WIND.plot_speed_frequency(), plt.Axes)
    plt.close("all")


def test_plot_cumulative_density():
    """_"""
    assert isinstance(TEST_WIND.plot_cumulative_density(), plt.Axes)
    plt.close("all")
