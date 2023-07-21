import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from ladybugtools_toolkit.external_comfort.external_comfort import (
    ExternalComfort,
    SimulationResult,
)
from ladybugtools_toolkit.external_comfort.material import Materials
from ladybugtools_toolkit.external_comfort.shelter import Shelters
from ladybugtools_toolkit.external_comfort.typology import Typologies, Typology

from .. import EPW_FILE, EXTERNAL_COMFORT_IDENTIFIER

GROUND_MATERIAL = Materials.LBT_AsphaltPavement.value
SHADE_MATERIAL = Materials.FABRIC.value

SIMULATION_RESULT = SimulationResult(
    EPW_FILE,
    GROUND_MATERIAL,
    SHADE_MATERIAL,
    EXTERNAL_COMFORT_IDENTIFIER,
).run()
TYPOLOGY = Typologies.EAST_SHELTER_WITH_CANOPY.value


def test_external_comfort():
    """_"""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    assert isinstance(ext_comf, ExternalComfort)


def test_external_comfort_array():
    """_"""
    typ = Typology(
        Name="example",
        Shelters=[Shelters.EAST.value],
        EvaporativeCoolingEffect=np.where(
            np.array(range(8760)) % 8 == 0, np.ones(8760) * 0.85, np.zeros(8760)
        ),
        WindSpeedMultiplier=np.where(
            np.array(range(8760)) % 2 == 0, np.ones(8760) - 0.5, np.zeros(8760) + 1.5
        ),
        RadiantTemperatureAdjustment=np.where(
            np.array(range(8760)) % 3 == 0, np.zeros(8760) - 1.2, np.zeros(8760)
        ),
    )
    ext_comf = ExternalComfort(SimulationResult=SIMULATION_RESULT, Typology=typ)

    assert (
        pytest.approx(ext_comf.universal_thermal_climate_index.average, rel=0.1)
        == 7.331275990178459
    )


def test_to_dataframe():
    """_"""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    assert isinstance(ext_comf.to_dataframe(), pd.DataFrame)


def test_plot_utci_day_comfort_metrics():
    """_"""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    assert isinstance(ext_comf.plot_utci_day_comfort_metrics(), plt.Axes)
    plt.close("all")


def test_plot_utci_heatmap():
    """_"""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    assert isinstance(ext_comf.plot_utci_heatmap(), plt.Axes)
    plt.close("all")


def test_plot_utci_heatmap_histogram():
    """_"""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    assert isinstance(ext_comf.plot_utci_heatmap_histogram(), plt.Figure)
    plt.close("all")


def test_plot_utci_distance_to_comfortable():
    """_"""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    assert isinstance(ext_comf.plot_utci_distance_to_comfortable(), plt.Axes)
    plt.close("all")


def test_plot_dbt_heatmap():
    """_"""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    assert isinstance(ext_comf.plot_dbt_heatmap(), plt.Axes)
    plt.close("all")


def test_plot_rh_heatmap():
    """_"""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    assert isinstance(ext_comf.plot_rh_heatmap(), plt.Axes)
    plt.close("all")


def test_plot_ws_heatmap():
    """_"""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    assert isinstance(ext_comf.plot_ws_heatmap(), plt.Axes)
    plt.close("all")


def test_plot_mrt_heatmap():
    """_"""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    assert isinstance(ext_comf.plot_mrt_heatmap(), plt.Axes)
    plt.close("all")


def test_to_dict():
    """Test whether an object can be converted to a dictionary."""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    for obj in [ext_comf]:
        obj_dict = obj.to_dict()
        assert "_t" in obj_dict


def test_to_json():
    """Test whether an object can be converted to a json string."""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    for obj in [ext_comf]:
        obj_json = obj.to_json()
        assert '"_t":' in obj_json


def test_from_dict_native():
    """Test whether an object can be converted from a dictionary directly."""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    for obj in [ext_comf]:
        new_obj = type(obj).from_dict(obj.to_dict())
        assert isinstance(new_obj, type(obj))


def test_from_json_native():
    """Test whether an object can be converted from a json string directly."""
    ext_comf = ExternalComfort(SIMULATION_RESULT, TYPOLOGY)
    for obj in [ext_comf]:
        new_obj = type(obj).from_json(obj.to_json())
        assert isinstance(new_obj, type(obj))
