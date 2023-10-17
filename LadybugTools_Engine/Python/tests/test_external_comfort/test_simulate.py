from ladybug.epw import HourlyContinuousCollection
from ladybugtools_toolkit.external_comfort._simulatebase import (
    SimulationResult,
    radiant_temperature,
    simulate_surface_temperatures,
    simulation_directory,
)
from ladybugtools_toolkit.external_comfort.model import create_model

from .. import EPW_FILE, EPW_OBJ, EXTERNAL_COMFORT_IDENTIFIER
from .test_material import TEST_GROUND_MATERIAL, TEST_SHADE_MATERIAL

TEST_SIMULATION_RESULT = SimulationResult(
    EpwFile=EPW_FILE,
    GroundMaterial=TEST_GROUND_MATERIAL,
    ShadeMaterial=TEST_SHADE_MATERIAL,
    Identifier=EXTERNAL_COMFORT_IDENTIFIER,
)
TEST_MODEL = create_model(
    ground_material=TEST_GROUND_MATERIAL,
    shade_material=TEST_SHADE_MATERIAL,
    identifier=EXTERNAL_COMFORT_IDENTIFIER,
)


def test_simulation_directory():
    """_"""

    wd = simulation_directory(model=TEST_MODEL)
    assert wd.exists()
    assert wd.is_dir()
    assert wd.name == EXTERNAL_COMFORT_IDENTIFIER


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
    result = simulate_surface_temperatures(
        model=TEST_MODEL, epw_file=EPW_FILE, remove_dir=True
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
        model=TEST_MODEL, epw_file=EPW_FILE, remove_dir=False
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


def test_round_trip():
    """Test whether an object can be converted to a dictionary, and json and back."""
    assert isinstance(
        SimulationResult(**TEST_SIMULATION_RESULT.dict()), SimulationResult
    )
    assert isinstance(
        SimulationResult.parse_raw(TEST_SIMULATION_RESULT.json(by_alias=True)),
        SimulationResult,
    )
