import numpy as np
import pytest
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW

from ladybugtools_toolkit.external_comfort.material import Materials
from ladybugtools_toolkit.external_comfort.typology import (
    Shelter,
    SimulationResult,
    Typologies,
    Typology,
)

from ...tests import EPW_FILE, EXTERNAL_COMFORT_IDENTIFIER

EPW_OBJ = EPW(EPW_FILE)
GROUND_MATERIAL = Materials.ASPHALT_PAVEMENT.value
SHADE_MATERIAL = Materials.FABRIC.value

TYPOLOGY = Typology(
    name=EXTERNAL_COMFORT_IDENTIFIER,
    shelters=[
        Shelter(
            wind_porosity=0.5,
            radiation_porosity=0.5,
            altitude_range=(45, 90),
            azimuth_range=(90, 270),
        )
    ],
    evaporative_cooling_effectiveness=0.1,
)


def test_sky_exposure():
    """_"""
    assert TYPOLOGY.sky_exposure() == pytest.approx(0.9267766952966369, rel=0.1)


def test_sun_exposure():
    """_"""
    x = np.array(TYPOLOGY.sun_exposure(EPW_OBJ))
    assert x[~np.isnan(x)].sum() == 4093


def test_dry_bulb_temperature():
    """_"""
    assert TYPOLOGY.dry_bulb_temperature(EPW_OBJ).average == pytest.approx(
        10.036858349164525, rel=0.1
    )


def test_relative_humidity():
    """_"""
    assert TYPOLOGY.relative_humidity(EPW_OBJ).average == pytest.approx(
        81.36280821917846, rel=0.1
    )


def test_wind_speed():
    """_"""
    assert TYPOLOGY.wind_speed(EPW_OBJ).average == pytest.approx(
        2.026758704337967, rel=0.1
    )


def test_mean_radiant_temperature():
    """_"""
    sim_result = SimulationResult(
        EPW_FILE, GROUND_MATERIAL, SHADE_MATERIAL, EXTERNAL_COMFORT_IDENTIFIER
    ).run()
    assert TYPOLOGY.mean_radiant_temperature(sim_result).average == pytest.approx(
        15.64262493993061, rel=0.5
    )


def test_typologies():
    """_"""
    for typology in Typologies:
        assert isinstance(typology.value, Typology)


def test_universal_thermal_climate_index():
    """_"""
    sim_result = SimulationResult(
        EPW_FILE, GROUND_MATERIAL, SHADE_MATERIAL, EXTERNAL_COMFORT_IDENTIFIER
    ).run()
    assert isinstance(
        TYPOLOGY.universal_thermal_climate_index(sim_result), HourlyContinuousCollection
    )
