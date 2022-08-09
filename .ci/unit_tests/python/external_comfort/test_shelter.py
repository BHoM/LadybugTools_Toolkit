import pytest
from ladybugtools_toolkit.external_comfort.shelter.shelter import Shelter
from ladybugtools_toolkit.ladybug_extension.epw.sun_position_list import (
    sun_position_list,
)

from .. import EPW_OBJ

SUNS = sun_position_list(EPW_OBJ)


def test_sun_blocked_south():
    south_shelter = Shelter(porosity=0, altitude_range=(0, 90), azimuth_range=(90, 270))
    south_suns = [
        i for i in SUNS if (i.azimuth > 90) and (i.azimuth < 270) and (i.altitude > 0)
    ]
    assert sum(south_shelter.sun_blocked(south_suns)) == len(south_suns)


def test_sun_blocked_north():
    north_shelter = Shelter(porosity=0, altitude_range=(0, 90), azimuth_range=(270, 90))
    north_suns = [
        i for i in SUNS if ((i.azimuth < 90) or (i.azimuth > 270)) and (i.altitude > 0)
    ]
    assert sum(north_shelter.sun_blocked(north_suns)) == len(north_suns)


def test_sky_blocked_opaque():
    shelter = Shelter(porosity=0, altitude_range=(0, 90), azimuth_range=(0, 360))
    assert shelter.sky_blocked() == 1


def test_sky_blocked_porous():
    shelter = Shelter(porosity=0.5, altitude_range=(0, 90), azimuth_range=(0, 360))
    assert shelter.sky_blocked() == 0.5


def test_effective_wind_speed():
    shelter = Shelter(porosity=0.5, altitude_range=(0, 90), azimuth_range=(0, 360))
    assert shelter.effective_wind_speed(EPW_OBJ).average == pytest.approx(
        0.40605379566207317
    )
