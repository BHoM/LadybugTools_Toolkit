import matplotlib.pyplot as plt
from ladybugtools_toolkit.external_comfort.external_comfort import ExternalComfort
from ladybugtools_toolkit.external_comfort.simulate.simulation_result import (
    SimulationResult,
)
from ladybugtools_toolkit.external_comfort.typology.typologies import Typologies
from matplotlib import pyplot as plt

from .. import EPW_OBJ, GROUND_MATERIAL, IDENTIFIER, SHADE_MATERIAL

SIM_RESULT = SimulationResult(EPW_OBJ, GROUND_MATERIAL, SHADE_MATERIAL, IDENTIFIER)
EC_RESULT = ExternalComfort(SIM_RESULT, Typologies.OPENFIELD.value)


def test_to_dict():
    keys = [
        "universal_thermal_climate_index",
        "dry_bulb_temperature",
        "relative_humidity",
        "mean_radiant_temperature",
        "wind_speed",
        "name",
        "shelters",
        "evaporative_cooling_effectiveness",
        "epw",
        "ground_material",
        "shade_material",
        "model",
        "shaded_above_temperature",
        "shaded_below_temperature",
        "shaded_diffuse_radiation",
        "shaded_direct_radiation",
        "shaded_longwave_mean_radiant_temperature",
        "shaded_mean_radiant_temperature",
        "shaded_total_radiation",
        "unshaded_above_temperature",
        "unshaded_below_temperature",
        "unshaded_diffuse_radiation",
        "unshaded_direct_radiation",
        "unshaded_longwave_mean_radiant_temperature",
        "unshaded_mean_radiant_temperature",
        "unshaded_total_radiation",
    ]
    d = EC_RESULT.to_dict()
    for key in keys:
        assert key in d.keys()


def test_to_dataframe():
    assert EC_RESULT.to_dataframe().shape == (8760, 5)


def test_plot_utci_day_comfort_metrics():
    assert isinstance(EC_RESULT.plot_utci_day_comfort_metrics(), plt.Figure)
    plt.close("all")


def test_plot_utci_heatmap():
    assert isinstance(EC_RESULT.plot_utci_heatmap(), plt.Figure)
    plt.close("all")


def test_plot_utci_heatmap_histogram():
    assert isinstance(EC_RESULT.plot_utci_heatmap_histogram(), plt.Figure)
    plt.close("all")


def test_plot_utci_distance_to_comfortable():
    assert isinstance(EC_RESULT.plot_utci_distance_to_comfortable(), plt.Figure)
    plt.close("all")


def test_plot_dbt_heatmap():
    assert isinstance(EC_RESULT.plot_dbt_heatmap(), plt.Figure)
    plt.close("all")


def test_plot_rh_heatmap():
    assert isinstance(EC_RESULT.plot_rh_heatmap(), plt.Figure)
    plt.close("all")


def test_plot_ws_heatmap():
    assert isinstance(EC_RESULT.plot_ws_heatmap(), plt.Figure)
    plt.close("all")


def test_plot_mrt_heatmap():
    assert isinstance(EC_RESULT.plot_mrt_heatmap(), plt.Figure)
    plt.close("all")
