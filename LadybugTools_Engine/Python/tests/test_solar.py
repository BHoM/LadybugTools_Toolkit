import matplotlib.pyplot as plt
from ladybugtools_toolkit.solar import radiation_rose, tilt_orientation_factor

from . import EPW_FILE


def test_tilt_orientation_factor():
    """_"""
    assert isinstance(tilt_orientation_factor(EPW_FILE), plt.Axes)


def test_radiation_rose():
    """_"""
    assert isinstance(radiation_rose(EPW_FILE), plt.Axes)


# def test_solar_from_epw():
#     """_"""

#     assert isinstance(Solar.from_epw(EPW_OBJ), Solar)
#     assert isinstance(Solar.from_epw(EPW_FILE), Solar)


# def test_altitudes():
#     """_"""

#     with pytest.raises(ValueError):
#         Solar.altitudes(n="not_an_int")
#         Solar.altitudes(n=2)


# def test_azimuths():
#     """_"""

#     with pytest.raises(ValueError):
#         Solar.azimuths(n="not_an_int")
#         Solar.azimuths(n=2)


# def test_solar_calc():
#     solar = Solar.from_epw(EPW_OBJ)

#     assert solar.detailed_irradiance_matrix(location=EPW_OBJ.location).shape == (
#         8760,
#         36,
#     )
#     assert solar.directional_irradiance_matrix(location=EPW_OBJ.location).shape == (
#         8760,
#         128,
#     )


# def test_solar_plot():
#     solar = Solar.from_epw(EPW_OBJ)

#     assert isinstance(
#         solar.plot_tilt_orientation_factor(location=EPW_OBJ.location), plt.Axes
#     )
#     assert isinstance(
#         solar.plot_directional_irradiance(location=EPW_OBJ.location), plt.Axes
#     )
