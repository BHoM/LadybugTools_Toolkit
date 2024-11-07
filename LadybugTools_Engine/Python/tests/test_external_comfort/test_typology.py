from pathlib import Path
from tempfile import gettempdir

import pytest
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.typology import (Typologies,
                                                            Typology,
                                                            combine_typologies)

from .. import EPW_FILE, EXTERNAL_COMFORT_IDENTIFIER
from .test_shelter import TEST_SHELTER
from .test_simulate import TEST_SIMULATION_RESULT

EPW_OBJ = EPW(EPW_FILE)

TEST_TYPOLOGY = Typology(
    identifier=EXTERNAL_COMFORT_IDENTIFIER,
    shelters=[TEST_SHELTER],
    evaporative_cooling_effect=[0.1] * 8760,
)


def test_round_trip():
    """_"""
    tempfile = Path(gettempdir()) / "pytest_typology.json"
    Typology.from_dict(TEST_TYPOLOGY.to_dict())
    Typology.from_json(TEST_TYPOLOGY.to_json())
    Typology.from_file(TEST_TYPOLOGY.to_file(tempfile))
    tempfile.unlink()


def test_combine_typologies():
    """_"""
    assert isinstance(
        combine_typologies(
            [Typologies.SKY_SHELTER.value, Typologies.SOUTH_SHELTER.value]
        ),
        Typology,
    )


def test_dry_bulb_temperature():
    """_"""
    assert TEST_TYPOLOGY.dry_bulb_temperature(
        EPW_OBJ).average == pytest.approx(10.036858349164525, rel=0.1)


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
