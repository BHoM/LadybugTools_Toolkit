import pytest
from ladybug.epw import EPW

from ladybugtools_toolkit.external_comfort.shelter import Shelter
from ladybugtools_toolkit.ladybug_extension.epw import sun_position_list

from ...tests import EPW_FILE

EPW_OBJ = EPW(EPW_FILE)
SUNS = sun_position_list(EPW_OBJ)


def test_to_dict():
    """Test whether an object can be converted to a dictionary."""
    obj = Shelter()
    obj_dict = obj.to_dict()
    assert "_t" in obj_dict.keys()


def test_to_json():
    """Test whether an object can be converted to a json string."""
    obj = Shelter()
    obj_json = obj.to_json()
    assert '"_t":' in obj_json


def test_from_dict_native():
    """Test whether an object can be converted from a dictionary directly."""
    obj = Shelter()
    obj_dict = obj.to_dict()
    assert isinstance(Shelter.from_dict(obj_dict), Shelter)


def test_from_json_native():
    """Test whether an object can be converted from a json string directly."""
    obj = Shelter()
    obj_dict = obj.to_json()
    assert isinstance(Shelter.from_json(obj_dict), Shelter)


def test_sun_blocked_south():
    """Test whether sun is blocked."""
    south_shelter = Shelter(
        radiation_porosity=0, altitude_range=(0, 90), azimuth_range=(90, 270)
    )
    south_suns = [
        i for i in SUNS if (i.azimuth > 90) and (i.azimuth < 270) and (i.altitude > 0)
    ]
    assert sum(south_shelter.sun_blocked(south_suns)) == len(south_suns)


def test_sun_blocked_north():
    """Test whether sun is blocked."""
    north_shelter = Shelter(
        radiation_porosity=0, altitude_range=(0, 90), azimuth_range=(270, 90)
    )
    north_suns = [
        i for i in SUNS if ((i.azimuth < 90) or (i.azimuth > 270)) and (i.altitude > 0)
    ]
    assert sum(north_shelter.sun_blocked(north_suns)) == len(north_suns)


def test_sky_blocked_opaque():
    """Test whether sun is blocked."""
    shelter = Shelter(
        radiation_porosity=0, altitude_range=(0, 90), azimuth_range=(0, 360)
    )
    assert shelter.sky_blocked() == 1


def test_sky_blocked_porous():
    """Test whether sky is blocked."""
    shelter = Shelter(
        radiation_porosity=0.5, altitude_range=(0, 90), azimuth_range=(0, 360)
    )
    assert shelter.sky_blocked() == 0.5


def test_effective_wind_speed():
    """Test effective wind speed."""
    shelter = Shelter(wind_porosity=0.5, altitude_range=(0, 90), azimuth_range=(0, 360))
    assert shelter.effective_wind_speed(EPW_OBJ).average == pytest.approx(
        0.40605379566207317
    )
