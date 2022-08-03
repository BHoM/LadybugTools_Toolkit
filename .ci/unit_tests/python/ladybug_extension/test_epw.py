from copy import deepcopy

from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.ladybug_extension.epw.clearness_index import clearness_index
from ladybugtools_toolkit.ladybug_extension.epw.enthalpy import enthalpy
from ladybugtools_toolkit.ladybug_extension.epw.equality import equality
from ladybugtools_toolkit.ladybug_extension.epw.equation_of_time import equation_of_time
from ladybugtools_toolkit.ladybug_extension.epw.from_dataframe import from_dataframe
from ladybugtools_toolkit.ladybug_extension.epw.humidity_ratio import humidity_ratio
from ladybugtools_toolkit.ladybug_extension.epw.solar_altitude import solar_altitude
from ladybugtools_toolkit.ladybug_extension.epw.solar_altitude_radians import (
    solar_altitude_radians,
)
from ladybugtools_toolkit.ladybug_extension.epw.solar_azimuth import solar_azimuth
from ladybugtools_toolkit.ladybug_extension.epw.solar_azimuth_radians import (
    solar_azimuth_radians,
)
from ladybugtools_toolkit.ladybug_extension.epw.solar_declination import (
    solar_declination,
)
from ladybugtools_toolkit.ladybug_extension.epw.solar_time_datetime import (
    solar_time_datetime,
)
from ladybugtools_toolkit.ladybug_extension.epw.solar_time_hour import solar_time_hour
from ladybugtools_toolkit.ladybug_extension.epw.sun_position_collection import (
    sun_position_collection,
)
from ladybugtools_toolkit.ladybug_extension.epw.sun_position_list import (
    sun_position_list,
)
from ladybugtools_toolkit.ladybug_extension.epw.to_dataframe import to_dataframe
from ladybugtools_toolkit.ladybug_extension.epw.wet_bulb_temperature import (
    wet_bulb_temperature,
)

from .. import EPW_DF, EPW_OBJ


def test_to_dataframe():
    assert len(to_dataframe(EPW_OBJ).columns) == 47


def test_from_dataframe():
    assert isinstance(from_dataframe(EPW_DF), EPW)


def test_equality_good():
    assert equality(EPW_OBJ, deepcopy(EPW_OBJ))


def test_equality_bad():
    bad_epw = deepcopy(EPW_OBJ)
    bad_epw.dry_bulb_temperature.values = [0] * 8760
    assert not equality(EPW_OBJ, bad_epw)


def test_clearness_index():
    assert len(clearness_index(EPW_OBJ)) == 8760


def test_enthalpy():
    assert isinstance(enthalpy(EPW_OBJ), HourlyContinuousCollection)


def test_equation_of_time():
    assert isinstance(equation_of_time(EPW_OBJ), HourlyContinuousCollection)


def test_humidity_ratio():
    assert isinstance(humidity_ratio(EPW_OBJ), HourlyContinuousCollection)


def test_solar_altitude_radians():
    assert isinstance(solar_altitude_radians(EPW_OBJ), HourlyContinuousCollection)


def test_solar_altitude():
    assert isinstance(solar_altitude(EPW_OBJ), HourlyContinuousCollection)


def test_solar_azimuth_radians():
    assert isinstance(solar_azimuth_radians(EPW_OBJ), HourlyContinuousCollection)


def test_solar_azimuth():
    assert isinstance(solar_azimuth(EPW_OBJ), HourlyContinuousCollection)


def test_solar_declination():
    assert isinstance(solar_declination(EPW_OBJ), HourlyContinuousCollection)


def test_solar_time_datetime():
    assert isinstance(solar_time_datetime(EPW_OBJ), HourlyContinuousCollection)


def test_solar_time_hour():
    assert isinstance(solar_time_hour(EPW_OBJ), HourlyContinuousCollection)


def test_sun_position_collection():
    assert isinstance(sun_position_collection(EPW_OBJ), HourlyContinuousCollection)


def test_sun_position_list():
    assert isinstance(sun_position_list(EPW_OBJ)[0].altitude, float)


def test_wet_bulb_temperature():
    assert isinstance(wet_bulb_temperature(EPW_OBJ), HourlyContinuousCollection)
