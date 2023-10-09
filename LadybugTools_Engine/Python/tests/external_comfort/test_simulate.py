from pathlib import Path

import pytest
from honeybee.config import folders as hb_folders
from ladybug.epw import HourlyContinuousCollection
from ladybugtools_toolkit.new_external_comfort.material import get_material
from ladybugtools_toolkit.new_external_comfort.model import create_model
from ladybugtools_toolkit.new_external_comfort.simulate import (
    SimulationResult,
    radiant_temperature,
    simulate_surface_temperatures,
    simulation_directory,
)

from .. import EPW_FILE, EPW_OBJ, EXTERNAL_COMFORT_IDENTIFIER


def test_simulation_directory():
    """_"""
    model = create_model(
        ground_material=get_material("Concrete Pavement"),
        shade_material=get_material("Fabric"),
        identifier=EXTERNAL_COMFORT_IDENTIFIER,
    )
    wd = simulation_directory(model=model)
    assert wd.exists()
    assert wd.is_dir()
    assert wd.name == "test_simulation_directory"
    wd.unlink()


def test_radiant_temperature():
    """_"""
    assert (
        radiant_temperature(
            [EPW_OBJ.dry_bulb_temperature * 0.5, EPW_OBJ.dry_bulb_temperature * 2],
            view_factors=[0.25, 0.75],
        ).average
        == 16.924573102344482
    )


def test_simulate_surface_temperature():
    """_"""
    model = create_model(
        ground_material=get_material("Concrete Pavement"),
        shade_material=get_material("Fabric"),
        identifier=EXTERNAL_COMFORT_IDENTIFIER,
    )
    result = simulate_surface_temperatures(
        model=model, epw_file=EPW_FILE, remove_dir=True
    )
    assert isinstance(result, dict)
    for key in [
        "unshaded_up_temperature",
        "shaded_up_temperature",
        "unshaded_down_temperature",
        "shaded_down_temperature",
    ]:
        assert key in result
        assert isinstance(result[key], HourlyContinuousCollection)
    assert result["unshaded_up_temperature"] == EPW_OBJ.sky_temperature

    # reload old results and check if they are the same
    reloaded_result = simulate_surface_temperatures(
        model=model, epw_file=EPW_FILE, remove_dir=False
    )
    assert (
        reloaded_result["unshaded_up_temperature"] == result["unshaded_up_temperature"]
    )
    assert reloaded_result["shaded_up_temperature"] == result["shaded_up_temperature"]
    assert (
        reloaded_result["unshaded_down_temperature"]
        == result["unshaded_down_temperature"]
    )
    assert (
        reloaded_result["shaded_down_temperature"] == result["shaded_down_temperature"]
    )


def test_simulation_result():
    """_"""

    res = SimulationResult(
        epw_file=EPW_FILE,
        ground_material=get_material("Concrete Pavement"),
        shade_material=get_material("Fabric"),
        identifier=EXTERNAL_COMFORT_IDENTIFIER,
    )

    # round trip test
    dict_obj = res.to_dict()
    assert isinstance(SimulationResult.from_dict(dict_obj), SimulationResult)
    assert dict_obj["_t"] == "BH.oM.LadybugTools.SimulationResult"

    json_str = res.to_json()
    assert isinstance(SimulationResult.from_json(json_str), SimulationResult)

    # partial reconstruction test
    res_custom = SimulationResult(
        epw_file=EPW_FILE,
        ground_material=get_material("Concrete Pavement"),
        shade_material=get_material("Fabric"),
        identifier=EXTERNAL_COMFORT_IDENTIFIER,
        shaded_down_temperature=EPW_OBJ.dry_bulb_temperature.get_aligned_collection(5),
        unshaded_down_temperature=None,
        shaded_up_temperature=EPW_OBJ.dry_bulb_temperature.get_aligned_collection(5),
        unshaded_up_temperature=None,
    )
    assert res_custom.shaded_down_temperature == res_custom.shaded_up_temperature
    assert res_custom.shaded_mean_radiant_temperature.average == 5
