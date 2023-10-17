from copy import deepcopy

import pandas as pd
import pytest
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.ladybug_extension.epw import (
    clearness_index,
    enthalpy,
    epw_from_dataframe,
    epw_to_dataframe,
    equality,
    equation_of_time,
    humidity_ratio,
    solar_altitude,
    solar_altitude_radians,
    solar_azimuth,
    solar_azimuth_radians,
    solar_declination,
    solar_time_datetime,
    solar_time_hour,
    sun_position_collection,
    sun_position_list,
    wet_bulb_temperature,
)

from .. import EPW_OBJ


def test_to_dataframe():
    """_"""
    df = epw_to_dataframe(EPW_OBJ, include_additional=False)
    assert len(df.columns) == 30


def test_from_dataframe():
    """_"""
    df = epw_to_dataframe(EPW_OBJ, include_additional=False)
    assert isinstance(epw_from_dataframe(df), EPW)


def test_equality_good():
    """_"""
    assert equality(EPW_OBJ, deepcopy(EPW_OBJ))


def test_equality_bad():
    """_"""
    bad_epw = deepcopy(EPW_OBJ)
    bad_epw.dry_bulb_temperature.values = [0] * 8760
    with pytest.warns(UserWarning):
        assert not equality(EPW_OBJ, bad_epw)


def test_clearness_index():
    """_"""
    assert len(clearness_index(EPW_OBJ)) == 8760


def test_enthalpy():
    """_"""
    assert isinstance(enthalpy(EPW_OBJ), HourlyContinuousCollection)


def test_equation_of_time():
    """_"""
    assert isinstance(equation_of_time(EPW_OBJ), HourlyContinuousCollection)


def test_humidity_ratio():
    """_"""
    assert isinstance(humidity_ratio(EPW_OBJ), HourlyContinuousCollection)


def test_solar_altitude_radians():
    """_"""
    assert isinstance(solar_altitude_radians(EPW_OBJ), HourlyContinuousCollection)


def test_solar_altitude():
    """_"""
    assert isinstance(solar_altitude(EPW_OBJ), HourlyContinuousCollection)


def test_solar_azimuth_radians():
    """_"""
    assert isinstance(solar_azimuth_radians(EPW_OBJ), HourlyContinuousCollection)


def test_solar_azimuth():
    """_"""
    assert isinstance(solar_azimuth(EPW_OBJ), HourlyContinuousCollection)


def test_solar_declination():
    """_"""
    assert isinstance(solar_declination(EPW_OBJ), HourlyContinuousCollection)


def test_solar_time_datetime():
    """_"""
    assert isinstance(solar_time_datetime(EPW_OBJ), HourlyContinuousCollection)


def test_solar_time_hour():
    """_"""
    assert isinstance(solar_time_hour(EPW_OBJ), HourlyContinuousCollection)


def test_sun_position_collection():
    """_"""
    assert isinstance(sun_position_collection(EPW_OBJ), HourlyContinuousCollection)


def test_sun_position_list():
    """_"""
    assert isinstance(sun_position_list(EPW_OBJ)[0].altitude, float)


def test_wet_bulb_temperature():
    """_"""
    assert isinstance(wet_bulb_temperature(EPW_OBJ), HourlyContinuousCollection)
