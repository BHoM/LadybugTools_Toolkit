import numpy as np
import pytest
from ladybug.epw import EPW
from ladybug_geometry.geometry3d.pointvector import Point3D
from ladybugtools_toolkit.external_comfort._shelterbase import Shelter
from ladybugtools_toolkit.external_comfort._simulatebase import SimulationResult
from ladybugtools_toolkit.external_comfort.material import get_material
from ladybugtools_toolkit.external_comfort.typology import (
    Typology,
    combine_typologies,
    east_shelter,
    east_shelter_with_canopy,
    east_west_linear_shelter,
    enclosed,
    fritted_sky_shelter,
    misting,
    near_water,
    north_shelter,
    north_shelter_with_canopy,
    north_south_linear_shelter,
    northeast_shelter,
    northeast_shelter_with_canopy,
    northeast_southwest_linear_shelter,
    northwest_shelter,
    northwest_shelter_with_canopy,
    northwest_southeast_linear_shelter,
    openfield,
    pdec,
    porous_enclosure,
    sky_shelter,
    south_shelter,
    south_shelter_with_canopy,
    southeast_shelter,
    southeast_shelter_with_canopy,
    southwest_shelter,
    southwest_shelter_with_canopy,
    west_shelter,
    west_shelter_with_canopy,
)

from .. import EPW_FILE, EXTERNAL_COMFORT_IDENTIFIER
from .test_shelter import TEST_SHELTER
from .test_simulate import TEST_SIMULATION_RESULT

EPW_OBJ = EPW(EPW_FILE)

TEST_TYPOLOGY = Typology(
    Name=EXTERNAL_COMFORT_IDENTIFIER,
    Shelters=[TEST_SHELTER],
    EvaporativeCoolingEffect=[0.1] * 8760,
)


def test_combine_typologies():
    """_"""
    with pytest.warns(UserWarning):
        assert isinstance(
            combine_typologies([sky_shelter(), south_shelter()]), Typology
        )


def test_dry_bulb_temperature():
    """_"""
    assert TEST_TYPOLOGY.dry_bulb_temperature(EPW_OBJ).average == pytest.approx(
        10.036858349164525, rel=0.1
    )


def test_relative_humidity():
    """_"""
    assert TEST_TYPOLOGY.relative_humidity(EPW_OBJ).average == pytest.approx(
        81.36280821917846, rel=0.1
    )


def test_wind_speed():
    """_"""
    assert TEST_TYPOLOGY.wind_speed(EPW_OBJ).average == pytest.approx(
        3.2418835616436126, rel=0.1
    )


def test_mean_radiant_temperature():
    """_"""
    assert TEST_TYPOLOGY.mean_radiant_temperature(
        TEST_SIMULATION_RESULT
    ).average == pytest.approx(15.64262493993061, rel=0.5)


def test_predefined_typologies():
    """_"""
    for typology in [
        east_shelter_with_canopy,
        east_shelter,
        east_west_linear_shelter,
        enclosed,
        fritted_sky_shelter,
        misting,
        near_water,
        north_shelter_with_canopy,
        north_shelter,
        north_south_linear_shelter,
        northeast_shelter_with_canopy,
        northeast_shelter,
        northeast_southwest_linear_shelter,
        northwest_shelter_with_canopy,
        northwest_shelter,
        northwest_southeast_linear_shelter,
        openfield,
        pdec,
        porous_enclosure,
        sky_shelter,
        south_shelter_with_canopy,
        south_shelter,
        southeast_shelter_with_canopy,
        southeast_shelter,
        southwest_shelter_with_canopy,
        southwest_shelter,
        west_shelter_with_canopy,
        west_shelter,
    ]:
        assert isinstance(typology(), Typology)
