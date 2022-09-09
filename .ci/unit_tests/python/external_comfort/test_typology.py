import numpy as np
import pytest
from ladybugtools_toolkit.external_comfort.shelter.shelter import Shelter
from ladybugtools_toolkit.external_comfort.simulate.simulation_result import (
    SimulationResult,
)
from ladybugtools_toolkit.external_comfort.typology.typology import Typology

from .. import EPW_OBJ, GROUND_MATERIAL, IDENTIFIER, SHADE_MATERIAL

TYPOLOGY = Typology(
    name=IDENTIFIER,
    shelters=[Shelter(porosity=0.5, altitude_range=(45, 90), azimuth_range=(90, 270))],
    evaporative_cooling_effectiveness=0.1,
)


def test_sky_exposure():
    assert TYPOLOGY.sky_exposure() == pytest.approx(0.9267766952966369, rel=0.1)


def test_sun_exposure():
    x = np.array(TYPOLOGY.sun_exposure(EPW_OBJ))
    assert x[~np.isnan(x)].sum() == 4093


def test_dry_bulb_temperature():
    assert TYPOLOGY.dry_bulb_temperature(EPW_OBJ).average == pytest.approx(
        10.036858349164525, rel=0.1
    )


def test_relative_humidity():
    assert TYPOLOGY.relative_humidity(EPW_OBJ).average == pytest.approx(
        81.36280821917846, rel=0.1
    )


def test_wind_speed():
    assert TYPOLOGY.wind_speed(EPW_OBJ).average == pytest.approx(
        2.026758704337967, rel=0.1
    )


def test_mean_radiant_temperature():
    sim_result = SimulationResult(EPW_OBJ, GROUND_MATERIAL, SHADE_MATERIAL, IDENTIFIER)
    assert TYPOLOGY.mean_radiant_temperature(sim_result).average == pytest.approx(
        15.64262493993061, rel=0.5
    )
