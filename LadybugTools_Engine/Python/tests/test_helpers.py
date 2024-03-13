from datetime import timedelta
import warnings  # pylint: disable=E0401


import numpy as np
import pandas as pd
import pytest
from ladybugtools_toolkit.helpers import (
    air_pressure_at_height,
    AnalysisPeriod,
    angle_from_cardinal,
    angle_from_north,
    cardinality,
    chunks,
    circular_weighted_mean,
    decay_rate_smoother,
    DecayMethod,
    default_analysis_periods,
    default_combined_analysis_periods,
    default_hour_analysis_periods,
    default_month_analysis_periods,
    dry_bulb_temperature_at_height,
    epw_wind_vectors,
    evaporative_cooling_effect_collection,
    evaporative_cooling_effect,
    month_hour_binned_series,
    OpenMeteoVariable,
    proximity_decay,
    radiation_at_height,
    remove_leap_days,
    rolling_window,
    sanitise_string,
    scrape_openmeteo,
    sunrise_sunset,
    target_wind_speed_collection,
    temperature_at_height,
    timedelta_tostring,
    validate_timeseries,
    wind_speed_at_height,
)

from . import EPW_OBJ


def test_sunrise_sunset():
    """_"""
    assert sunrise_sunset(EPW_OBJ.location).shape == (365, 9)


def test_openmeteo_variable():
    """_"""
    assert (
        OpenMeteoVariable.from_string("winddirection_100m")
        == OpenMeteoVariable.WINDDIRECTION_100M
    )
    assert OpenMeteoVariable.TEMPERATURE_2M.openmeteo_unit == "°C"
    assert OpenMeteoVariable.TEMPERATURE_2M.target_name == "Dry Bulb Temperature"
    assert OpenMeteoVariable.TEMPERATURE_2M.target_unit == "C"
    assert OpenMeteoVariable.TEMPERATURE_2M.target_multiplier == 1
    assert OpenMeteoVariable.TEMPERATURE_2M.openmeteo_name == "temperature_2m"
    assert (
        OpenMeteoVariable.TEMPERATURE_2M.target_table_name == "Dry Bulb Temperature (C)"
    )
    assert (
        OpenMeteoVariable.TEMPERATURE_2M.openmeteo_table_name == "temperature_2m (°C)"
    )
    assert OpenMeteoVariable.TEMPERATURE_2M.convert(1) == 1


def test_epw_wind_vectors():
    """_"""
    assert len(epw_wind_vectors(EPW_OBJ)) == 8760
    assert len(epw_wind_vectors(EPW_OBJ, normalise=True)) == 8760


def test_scrape_openmeteo():
    """_"""
    with pytest.raises(ValueError):
        # latitude out of range
        scrape_openmeteo(
            latitude=180,
            longitude=EPW_OBJ.location.longitude,
            convert_units=True,
            start_date="2021-01-01",
            end_date="2021-02-01",
            variables=[
                OpenMeteoVariable.DEWPOINT_2M,
            ],
        )
        # longitude out of range
        scrape_openmeteo(
            latitude=EPW_OBJ.location.latitude,
            longitude=-1000,
            convert_units=True,
            start_date="2021-01-01",
            end_date="2021-02-01",
            variables=[
                OpenMeteoVariable.DEWPOINT_2M,
            ],
        )
        # start date after end date
        scrape_openmeteo(
            latitude=EPW_OBJ.location.latitude,
            longitude=EPW_OBJ.location.longitude,
            convert_units=True,
            start_date="2021-02-01",
            end_date="2021-01-01",
            variables=[
                OpenMeteoVariable.DEWPOINT_2M,
                OpenMeteoVariable.SOIL_MOISTURE_28_TO_100CM,
                OpenMeteoVariable.TEMPERATURE_2M,
            ],
        )
        # invalid variable
        scrape_openmeteo(
            latitude=EPW_OBJ.location.latitude,
            longitude=EPW_OBJ.location.longitude,
            convert_units=True,
            start_date="2021-01-01",
            end_date="2021-02-01",
            variables=[
                "not a variable",
            ],
        )

    assert scrape_openmeteo(
        latitude=EPW_OBJ.location.latitude,
        longitude=EPW_OBJ.location.longitude,
        convert_units=True,
        start_date="2021-01-01",
        end_date="2021-02-01",
        variables=[
            OpenMeteoVariable.WINDSPEED_10M,
            OpenMeteoVariable.WINDDIRECTION_10M,
            OpenMeteoVariable.TEMPERATURE_2M,
        ],
    ).shape == (768, 3)


def test_chunks():
    """_"""
    assert len(list(chunks([1, 2, 3, 4, 5, 6, 7, 8, 9], 2))) == 5


def test_radiation_at_height():
    """_"""
    # Test with default lapse rate
    assert radiation_at_height(100, 1000, 10) == pytest.approx(107.92, rel=1e-2)
    assert radiation_at_height(200, 1000, 10) == pytest.approx(215.84, rel=1e-2)
    assert radiation_at_height(300, 1000, 10) == pytest.approx(323.76, rel=1e-2)

    # Test with custom lapse rate
    assert radiation_at_height(100, 1000, 10, lapse_rate=0.1) == pytest.approx(
        109.9, rel=1e-2
    )
    assert radiation_at_height(200, 2000, 10, lapse_rate=0.2) == pytest.approx(
        279.6, rel=1e-2
    )
    assert radiation_at_height(300, 10000, 10, lapse_rate=0.01) == pytest.approx(
        329.97, rel=1e-2
    )


def test_temperature_at_height():
    """_"""
    # Test case 1: normal input
    assert temperature_at_height(10, 10, 200) == 8.765

    # Test case 2: input beyond troposphere
    with pytest.warns(UserWarning):
        temperature_at_height(10, 10, 9000)

    # Test case 3: input with kwargs
    assert temperature_at_height(10, 10, 200, lapse_rate=0.009) == 8.29


def test_circular_weighted_mean():
    """_"""

    # Test with angles outside of expected range
    angles = [0, 90, 180, 270, 361]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    with pytest.raises(ValueError):
        circular_weighted_mean(angles, weights)

    # Test with number of weights not equal to number of angles
    angles = [0, 90, 180, 270]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    with pytest.raises(ValueError):
        circular_weighted_mean(angles, weights)

    # Test with negative angles
    angles = [-90, 0, 90, 180, 270]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    with pytest.raises(ValueError):
        circular_weighted_mean(angles, weights)

    # Test with equal weights
    angles = [90, 180, 270]
    weights = [1 / 3, 1 / 3, 1 / 3]
    assert np.isclose(circular_weighted_mean(angles, weights), 180)

    # Test equally distributed, with equal weighting
    angles = [45, 135, 225, 315]
    with pytest.warns(UserWarning):
        circular_weighted_mean(angles)

    # Test without weights specified
    angles = [0, 90, 180, 270]
    with pytest.warns(UserWarning):
        assert isinstance(circular_weighted_mean(angles), float)

    # Test with different weights
    angles = [90, 180, 270]
    weights = [0.3, 0.3, 0.4]
    assert np.isclose(circular_weighted_mean(angles, weights), 198.43, rtol=0.1)

    # Test about 0
    angles = [355, 5]
    weights = [0.5, 0.5]
    assert np.isclose(circular_weighted_mean(angles, weights), 0)

    # Test opposing
    angles = [0, 180]
    weights = [0.5, 0.5]
    with pytest.warns(UserWarning):
        assert isinstance(circular_weighted_mean(angles, weights), float)


def test_sanitise_string():
    """_"""
    assert sanitise_string("Hello World!") == "Hello_World_"
    assert sanitise_string("My file name.txt") == "My_file_name.txt"
    assert sanitise_string("This is a % test") == "This_is_a__test"
    assert sanitise_string("12345") == "12345"
    assert sanitise_string("") == ""


def test_angle_from_cardinal():
    """_"""
    assert angle_from_cardinal("N") == 0
    assert angle_from_cardinal("E") == 90
    assert angle_from_cardinal("S") == 180
    assert angle_from_cardinal("W") == 270
    assert angle_from_cardinal("NE") == 45
    assert angle_from_cardinal("SE") == 135
    assert angle_from_cardinal("SW") == 225
    assert angle_from_cardinal("NW") == 315
    with pytest.raises(ValueError):
        angle_from_cardinal("Z")


def test_timedelta_tostring():
    """_"""
    assert timedelta_tostring(timedelta(seconds=3661)) == "01:01"
    assert timedelta_tostring(timedelta(seconds=7200)) == "02:00"
    assert timedelta_tostring(timedelta(seconds=60)) == "00:01"
    assert timedelta_tostring(timedelta(seconds=0)) == "00:00"


def test_rolling_window():
    """_"""
    with pytest.raises(ValueError):
        rolling_window([2], 3)

    array = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    window = 2
    expected_output = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]
    assert rolling_window(array, window).tolist() == expected_output

    array = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    window = 3
    expected_output = [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
    ]
    assert rolling_window(array, window).tolist() == expected_output

    array = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    window = 4
    expected_output = [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
    ]
    assert rolling_window(array, window).tolist() == expected_output


def test_default_hour_analysis_periods():
    """_"""
    aps = default_hour_analysis_periods()
    assert len(aps) == 4
    assert isinstance(aps[0], AnalysisPeriod)
    assert isinstance(aps[1], AnalysisPeriod)
    assert isinstance(aps[2], AnalysisPeriod)
    assert isinstance(aps[3], AnalysisPeriod)
    assert aps[0].st_hour == 5
    assert aps[0].end_hour == 12
    assert aps[0].timestep == 1
    assert aps[1].st_hour == 13
    assert aps[1].end_hour == 17
    assert aps[1].timestep == 1
    assert aps[2].st_hour == 18
    assert aps[2].end_hour == 21
    assert aps[2].timestep == 1
    assert aps[3].st_hour == 22
    assert aps[3].end_hour == 4
    assert aps[3].timestep == 1


def test_default_month_analysis_periods():
    """_"""
    aps = default_month_analysis_periods()
    assert len(aps) == 4
    assert isinstance(aps[0], AnalysisPeriod)
    assert isinstance(aps[1], AnalysisPeriod)
    assert isinstance(aps[2], AnalysisPeriod)
    assert isinstance(aps[3], AnalysisPeriod)
    assert aps[0].st_month == 12
    assert aps[0].end_month == 2
    assert aps[1].st_month == 3
    assert aps[1].end_month == 5
    assert aps[2].st_month == 6
    assert aps[2].end_month == 8
    assert aps[3].st_month == 9
    assert aps[3].end_month == 11


def test_default_combined_analysis_periods():
    """_"""
    aps = default_combined_analysis_periods()
    assert len(aps) == 16


def test_default_analysis_periods():
    """_"""
    aps = default_analysis_periods()
    assert len(aps) == 1 + 4 + 4 + 16


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
        cardinality(12, directions=7)


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
            value=3,
            distance_to_value=1.5,
            max_distance=3,
            decay_method=DecayMethod.LINEAR,
        )
        == 1.5
    )

    assert (
        proximity_decay(
            value=3,
            distance_to_value=1.5,
            max_distance=3,
            decay_method=DecayMethod.SIGMOID,
        )
        == 1.5
    )
    assert (
        proximity_decay(
            value=3,
            distance_to_value=1.5,
            max_distance=3,
            decay_method=DecayMethod.PARABOLIC,
        )
        == 2.25
    )


def test_proximity_decay_bad():
    """_"""
    with pytest.raises(ValueError):
        proximity_decay(
            value=10,
            distance_to_value=5,
            max_distance=10,
            decay_method="not a decay method",
        )


def test_evaporative_cooling_effect():
    """_"""
    dbt, rh = evaporative_cooling_effect(20, 50, 0.5)
    assert (dbt == pytest.approx(16.9, rel=0.1)) and (rh == pytest.approx(75, rel=0.1))

    dbt, rh = evaporative_cooling_effect(20, 110, 0.5)
    assert (dbt == pytest.approx(20, rel=0.1)) and (rh == pytest.approx(100, rel=0.1))


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


def test_air_pressure_at_height():
    """_"""
    assert air_pressure_at_height(
        reference_value=2,
        reference_height=10,
        target_height=2,
    ) == pytest.approx(2.001897379377387, rel=0.0001)


def test_dry_bulb_temperature_at_height():
    """_"""
    assert sum(
        dry_bulb_temperature_at_height(epw=EPW_OBJ, target_height=50).values
    ) == pytest.approx(87388.7999999973, rel=0.0001)


def test_remove_leap_days():
    """_"""
    # Test with a DataFrame containing leap days
    df = pd.DataFrame(
        {
            "value": np.random.rand(366),
        },
        index=pd.date_range("2020-01-01", "2020-12-31", freq="D"),
    )
    df = pd.concat([df, df])  # Test with duplicated data
    df = remove_leap_days(df)
    assert len(df) == 365 * 2

    # Test with a Series containing leap days
    s = pd.Series(
        np.random.rand(366),
        index=pd.date_range("2020-01-01", "2020-12-31", freq="D"),
    )
    s = pd.concat([s, s])  # Test with duplicated data
    s = remove_leap_days(s)
    assert len(s) == 365 * 2

    # Test with a DataFrame without leap days
    df = pd.DataFrame(
        {
            "value": np.random.rand(365),
        },
        index=pd.date_range("2020-01-01", "2020-12-30", freq="D"),
    )
    df = pd.concat([df, df])  # Test with duplicated data
    df = remove_leap_days(df)
    assert len(df) == 728

    # Test with a Series without leap days
    s = pd.Series(
        np.random.rand(365),
        index=pd.date_range("2020-01-01", "2020-12-30", freq="D"),
    )
    s = pd.concat([s, s])  # Test with duplicated data
    s = remove_leap_days(s)
    assert len(s) == 728

    # Test with a series without a datetime index
    with pytest.raises(ValueError):
        remove_leap_days(pd.Series([1, 2, 3]))


def test_month_hour_binned_series():
    """_"""
    s = pd.Series(
        index=pd.date_range(start="2017-01-01 00:00:00", freq="60T", periods=8760),
        data=range(8760),
    )

    # test that the function returns a dataframe
    assert isinstance(month_hour_binned_series(s), pd.DataFrame)

    # test that the function raises an error if the series is not a time series
    with pytest.raises(ValueError):
        month_hour_binned_series([1, 2, 3])

    # test that the function raises an error if the series is empty
    with pytest.raises(ValueError):
        month_hour_binned_series(pd.Series(dtype=float))

    # test that the function raises an error if the series does not contain at least 12 months of data
    with pytest.raises(ValueError):
        month_hour_binned_series(
            pd.Series(
                index=pd.date_range(
                    start="2017-01-01 00:00:00", freq="60T", periods=5000
                ),
                data=range(5000),
            )
        )

    # test that the function raises an error if the series does not have at least 24 values per day
    with pytest.raises(ValueError):
        month_hour_binned_series(
            pd.Series(
                index=pd.date_range(
                    start="2017-01-01 00:00:00", freq="120T", periods=8760 * 3
                ),
                data=range(8760),
            )
        )

    # test that the function raises an error if the length of hour-bin-labels does not match that of hour-bins
    with pytest.raises(ValueError):
        month_hour_binned_series(s, hour_labels=["Morning", "Afternoon"])

    # test that the function raises an error if the length of month-bin-labels does not match that of month-bins
    with pytest.raises(ValueError):
        month_hour_binned_series(s, month_labels=["Q1", "Q2", "Q3"])

    # test that the function raises an error if hour bins do not contain all hours [0-23]
    with pytest.raises(ValueError):
        month_hour_binned_series(s, hour_bins=[[0, 1, 2], [3, 4, 5]])

    # test that the function raises an error if month bins do not contain all months [1-12]
    with pytest.raises(ValueError):
        month_hour_binned_series(s, month_bins=[[1, 2, 3], [4, 5, 6]])

    # test that the function raises an error if month bins overlap
    with pytest.raises(ValueError):
        month_hour_binned_series(
            s, month_bins=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 11, 12]]
        )

    # test that the function raises an error if hour bins overlap
    with pytest.raises(ValueError):
        month_hour_binned_series(
            s,
            hour_bins=[
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            ],
        )

    # test that the function returns a dataframe with the expected shape
    assert month_hour_binned_series(
        s,
        month_bins=[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        hour_bins=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        ],
    ).shape == (2, 2)
