import matplotlib.pyplot as plt
from ladybugtools_toolkit.solar import Solar
import pytest

from . import EPW_OBJ, EPW_FILE


def test_solar_creation():
    """_"""

    assert isinstance(Solar(EPW_OBJ), Solar)
    assert isinstance(Solar(EPW_FILE), Solar)


def test_validation():
    """_"""

    obj = Solar(EPW_OBJ)

    with pytest.raises(ValueError):
        obj.altitudes(n_altitudes="not_an_int")
        obj.altitudes(n_altitudes=2)
        obj.azimuths(n_azimuths="not_an_int")
        obj.azimuths(n_azimuths=2)


def test_solar_calc():
    solar = Solar(EPW_OBJ)

    assert solar.detailed_irradiance_matrix().shape == (8760, 36)
    assert solar.directional_irradiance_matrix().shape == (8760, 128)


def test_solar_plot():
    solar = Solar(EPW_OBJ)

    assert isinstance(solar.plot_tilt_orientation_factor(), plt.Axes)
    assert isinstance(solar.plot_directional_irradiance(), plt.Axes)
