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

from ...tests import EPW_FILE, EXTERNAL_COMFORT_IDENTIFIER

GROUND_MATERIAL = Materials.LBT_AsphaltPavement.value
SHADE_MATERIAL = Materials.FABRIC.value


def test_external_comfort():
    """_"""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)

    assert isinstance(ext_comf, ExternalComfort)


def test_external_comfort_array():
    """_"""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typology(
        name="example",
        shelters=[Shelters.EAST.value],
        evaporative_cooling_effect=np.where(
            np.array(range(8760)) % 8 == 0, np.ones(8760) * 0.85, np.zeros(8760)
        ),
        wind_speed_multiplier=np.where(
            np.array(range(8760)) % 2 == 0, np.ones(8760) - 0.5, np.zeros(8760) + 1.5
        ),
        radiant_temperature_adjustment=np.where(
            np.array(range(8760)) % 3 == 0, np.zeros(8760) - 1.2, np.zeros(8760)
        ),
    )
    ext_comf = ExternalComfort(simulation_result=sim_res, typology=typ)

    assert pytest.approx(ext_comf.average, rel=0.1) == 7.331275990178459


def test_to_dataframe():
    """_"""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)
    assert isinstance(ext_comf.to_dataframe(), pd.DataFrame)


def test_plot_utci_day_comfort_metrics():
    """_"""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)
    assert isinstance(ext_comf.plot_utci_day_comfort_metrics(), plt.Figure)
    plt.close("all")


def test_plot_utci_heatmap():
    """_"""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)
    assert isinstance(ext_comf.plot_utci_heatmap(), plt.Figure)
    plt.close("all")


def test_plot_utci_heatmap_histogram():
    """_"""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)
    assert isinstance(ext_comf.plot_utci_heatmap_histogram(), plt.Figure)
    plt.close("all")


def test_plot_utci_distance_to_comfortable():
    """_"""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)
    assert isinstance(ext_comf.plot_utci_distance_to_comfortable(), plt.Figure)
    plt.close("all")


def test_plot_dbt_heatmap():
    """_"""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)
    assert isinstance(ext_comf.plot_dbt_heatmap(), plt.Figure)
    plt.close("all")


def test_plot_rh_heatmap():
    """_"""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)
    assert isinstance(ext_comf.plot_rh_heatmap(), plt.Figure)
    plt.close("all")


def test_plot_ws_heatmap():
    """_"""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)
    assert isinstance(ext_comf.plot_ws_heatmap(), plt.Figure)
    plt.close("all")


def test_plot_mrt_heatmap():
    """_"""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)
    assert isinstance(ext_comf.plot_mrt_heatmap(), plt.Figure)
    plt.close("all")


def test_to_dict():
    """Test whether an object can be converted to a dictionary."""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)
    for obj in [ext_comf]:
        obj_dict = obj.to_dict()
        assert "_t" in obj_dict.keys()


def test_to_json():
    """Test whether an object can be converted to a json string."""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)
    for obj in [ext_comf]:
        obj_json = obj.to_json()
        assert '"_t":' in obj_json


def test_from_dict_native():
    """Test whether an object can be converted from a dictionary directly."""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)
    for obj in [ext_comf]:
        new_obj = type(obj).from_dict(obj.to_dict())
        assert isinstance(new_obj, type(obj))


def test_from_json_native():
    """Test whether an object can be converted from a json string directly."""
    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()
    typ = Typologies.EAST_SHELTER_WITH_CANOPY.value
    ext_comf = ExternalComfort(sim_res, typ)
    for obj in [ext_comf]:
        new_obj = type(obj).from_json(obj.to_json())
        assert isinstance(new_obj, type(obj))
