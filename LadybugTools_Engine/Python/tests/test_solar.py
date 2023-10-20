import matplotlib.pyplot as plt
from ladybugtools_toolkit.solar import Solar

from . import EPW_OBJ, EPW_FILE


def test_solar_creation():
    """_"""

    assert isinstance(Solar.from_epw(EPW_OBJ), Solar)
    assert isinstance(Solar(EPW_FILE), Solar)


def test_solar_calc():
    solar = Solar.from_epw(EPW_OBJ)

    assert solar.detailed_irradiance_matrix().shape == (8760, 36)
    assert solar.directional_irradiance_matrix().shape == (8760, 128)


def test_solar_plot():
    solar = Solar.from_epw(EPW_OBJ)

    assert isinstance(solar.plot_tilt_orientation_factor(), plt.Axes)
    assert isinstance(solar.plot_directional_irradiance(), plt.Axes)
