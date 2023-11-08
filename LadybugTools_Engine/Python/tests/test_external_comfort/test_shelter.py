from pathlib import Path
from tempfile import gettempdir
import numpy as np
import pytest
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort._shelterbase import (
    Point3D,
    Shelter,
    annual_sky_exposure,
    annual_sun_exposure,
    annual_wind_speed,
    write_shelters_to_hbjson,
)


from ladybugtools_toolkit.external_comfort.shelter import TreeShelter
from ladybugtools_toolkit.ladybug_extension.epw import sun_position_list

from .. import EPW_FILE

EPW_OBJ = EPW(EPW_FILE)
TEST_SHELTER = Shelter(
    vertices=[
        Point3D(-10, -10, 5),
        Point3D(-10, 10, 5),
        Point3D(10, 10, 5),
        Point3D(10, -10, 5),
    ]
)
SUNS = sun_position_list(EPW_OBJ)


def test_tree_species():
    """_"""
    for species in TreeShelter:
        assert isinstance(species.shelter(), Shelter)

    assert isinstance(TreeShelter.ACER_PLATANOIDES.shelter(), Shelter)
    assert isinstance(
        TreeShelter.ACER_PLATANOIDES.shelter(northern_hemisphere=False), Shelter
    )


def test_round_trip():
    """_"""
    tempfile = Path(gettempdir()) / "pytest_shelter.json"
    Shelter.from_dict(TEST_SHELTER.to_dict())
    Shelter.from_json(TEST_SHELTER.to_json())
    Shelter.from_file(TEST_SHELTER.to_file(tempfile))

    tempfile.unlink()


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


def test_annual_sky_exposure():
    """_"""
    assert sum(annual_sky_exposure([TEST_SHELTER])) == pytest.approx(2732.7556325820833)


def test_annual_wind_speed():
    """_"""
    assert sum(annual_wind_speed([TEST_SHELTER], EPW_OBJ)) == pytest.approx(
        28398.899999998048
    )


def test_annual_sun_exposure():
    """_"""
    assert np.nansum(annual_sun_exposure([TEST_SHELTER], EPW_OBJ)) == pytest.approx(
        2041
    )


def test_write_shelters_to_hbjson():
    """_"""
    tempfile = Path(gettempdir()) / "pytest_shelter.hbjson"
    write_shelters_to_hbjson([TEST_SHELTER], tempfile)
    assert tempfile.exists()

    tempfile.unlink()
