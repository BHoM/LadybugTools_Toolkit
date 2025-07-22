from pathlib import Path
from tempfile import gettempdir
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from ladybugtools_toolkit.external_comfort.externalcomfort import ExternalComfort
from ladybugtools_toolkit.external_comfort.typology import Typologies
from ladybug.datacollection import HourlyContinuousCollection

from .test_simulate import TEST_SIMULATION_RESULT
from .test_typology import TEST_TYPOLOGY
from .. import EPW_OBJ

TEST_EXTERNAL_COMFORT = ExternalComfort(
    simulation_result=TEST_SIMULATION_RESULT, typology=TEST_TYPOLOGY
)


def test_externalcomfort():
    """_"""
    assert isinstance(TEST_EXTERNAL_COMFORT, ExternalComfort)

    assert isinstance(
        ExternalComfort(
            simulation_result=TEST_SIMULATION_RESULT, typology=Typologies.OPENFIELD
        ),
        ExternalComfort,
    )

    with pytest.raises(ValueError):
        ExternalComfort(
            simulation_result="not_a_simulation_result", typology=TEST_TYPOLOGY
        )
        ExternalComfort(
            simulation_result=TEST_SIMULATION_RESULT, typology="not_a_typology"
        )
        ExternalComfort(
            simulation_result=TEST_SIMULATION_RESULT,
            typology=TEST_TYPOLOGY,
            dry_bulb_temperature="not_a_collection",
        )

    assert isinstance(repr(TEST_EXTERNAL_COMFORT), str)


def test_round_trip():
    """_"""
    tempfile = Path(gettempdir()) / "pytest_external_comfort.json"
    ExternalComfort.from_dict(TEST_EXTERNAL_COMFORT.to_dict())
    ExternalComfort.from_json(TEST_EXTERNAL_COMFORT.to_json())
    ExternalComfort.from_file(TEST_EXTERNAL_COMFORT.to_file(tempfile))

    tempfile.unlink()

    with pytest.raises(ValueError):
        TEST_EXTERNAL_COMFORT.to_file("./not_a_json_file.txt")


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


def test_plot_utci_histogram():
    """_"""

    assert isinstance(TEST_EXTERNAL_COMFORT.plot_utci_histogram(), plt.Axes)
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

def test_set_all_but_wind_speed():
    """Test that the wind speed method for getting the typology wind speed works correctly when dbt and rh are set."""
    temp_ec_test = ExternalComfort(TEST_SIMULATION_RESULT, typology=TEST_TYPOLOGY, dry_bulb_temperature=EPW_OBJ.dry_bulb_temperature, relative_humidity=EPW_OBJ.relative_humidity)
    assert isinstance(temp_ec_test, ExternalComfort)
    assert isinstance(temp_ec_test.wind_speed, HourlyContinuousCollection)