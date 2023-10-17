import matplotlib.pyplot as plt
import pandas as pd
import pytest
from ladybugtools_toolkit.external_comfort.externalcomfort import ExternalComfort

from .test_simulate import TEST_SIMULATION_RESULT
from .test_typology import TEST_TYPOLOGY

TEST_EXTERNAL_COMFORT = ExternalComfort(
    SimulationResult=TEST_SIMULATION_RESULT, Typology=TEST_TYPOLOGY
)


def test_externalcomfort():
    """_"""

    assert isinstance(TEST_EXTERNAL_COMFORT, ExternalComfort)


def test_to_dataframe():
    """_"""
    assert isinstance(TEST_EXTERNAL_COMFORT.to_dataframe(), pd.DataFrame)


def test_plot_utci_day_comfort_metrics():
    """_"""

    assert isinstance(TEST_EXTERNAL_COMFORT.plot_utci_day_comfort_metrics(), plt.Axes)
    plt.close("all")


def test_plot_utci_heatmap():
    """_"""

    assert isinstance(TEST_EXTERNAL_COMFORT.plot_utci_heatmap(), plt.Axes)
    plt.close("all")


def test_plot_utci_heatmap_histogram():
    """_"""

    assert isinstance(TEST_EXTERNAL_COMFORT.plot_utci_heatmap_histogram(), plt.Figure)
    plt.close("all")


def test_plot_utci_distance_to_comfortable():
    """_"""

    assert isinstance(
        TEST_EXTERNAL_COMFORT.plot_utci_distance_to_comfortable(), plt.Axes
    )
    plt.close("all")


def test_plot_dbt_heatmap():
    """_"""

    assert isinstance(TEST_EXTERNAL_COMFORT.plot_dbt_heatmap(), plt.Axes)
    plt.close("all")


def test_plot_rh_heatmap():
    """_"""

    assert isinstance(TEST_EXTERNAL_COMFORT.plot_rh_heatmap(), plt.Axes)
    plt.close("all")


def test_plot_ws_heatmap():
    """_"""

    assert isinstance(TEST_EXTERNAL_COMFORT.plot_ws_heatmap(), plt.Axes)
    plt.close("all")


def test_plot_mrt_heatmap():
    """_"""

    assert isinstance(TEST_EXTERNAL_COMFORT.plot_mrt_heatmap(), plt.Axes)
    plt.close("all")


def test_round_trip():
    """Test whether an object can be converted to a dictionary, and json and back."""

    assert isinstance(ExternalComfort(**TEST_EXTERNAL_COMFORT.dict()), ExternalComfort)
    assert isinstance(
        ExternalComfort.parse_raw(TEST_EXTERNAL_COMFORT.json(by_alias=True)),
        ExternalComfort,
    )
