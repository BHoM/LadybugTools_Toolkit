import numpy as np
import pytest
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort._shelterbase import Point3D, Shelter
from ladybugtools_toolkit.external_comfort.shelter import TreeSpecies
from ladybugtools_toolkit.ladybug_extension.epw import sun_position_list

from .. import EPW_FILE

EPW_OBJ = EPW(EPW_FILE)
TEST_SHELTER = Shelter(
    Vertices=[
        Point3D(-10, -10, 5),
        Point3D(-10, 10, 5),
        Point3D(10, 10, 5),
        Point3D(10, -10, 5),
    ]
)
SUNS = sun_position_list(EPW_OBJ)


def test_tree_species():
    """_"""
    for species in TreeSpecies:
        assert isinstance(species.shelter(), Shelter)


def test_round_trip():
    """Test whether an object can be converted to a dictionary, and json and back."""
    assert isinstance(Shelter(**TEST_SHELTER.dict()), Shelter)
    assert isinstance(Shelter.parse_raw(TEST_SHELTER.json(by_alias=True)), Shelter)


def test_set_porosity():
    """Test whether porosity setting works."""
    assert list(TEST_SHELTER.wind_porosity) == [0] * 8760
    assert list(TEST_SHELTER.radiation_porosity) == [0] * 8760
    assert list(TEST_SHELTER.set_porosity([0.5] * 8760).wind_porosity) == [0.5] * 8760
    assert (
        list(TEST_SHELTER.set_porosity([0.5] * 8760).radiation_porosity) == [0.5] * 8760
    )


def test_sky_exposure():
    """Test amount of sky exposure."""
    assert sum(TEST_SHELTER.annual_sky_exposure()) == pytest.approx(2732.7556325820833)


def test_sun_exposure():
    """Test sun exposure"""
    assert np.nansum(TEST_SHELTER.annual_sun_exposure(EPW_OBJ)) == 2041


def test_wind_adjustment():
    """Test wind speed adjustment"""
    assert sum(TEST_SHELTER.annual_wind_speed(EPW_OBJ)) == pytest.approx(
        28398.899999998048
    )
