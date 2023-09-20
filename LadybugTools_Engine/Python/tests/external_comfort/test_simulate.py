import pytest
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.material import Materials
from ladybugtools_toolkit.external_comfort.model import create_model
from ladybugtools_toolkit.external_comfort.simulate import (
    SimulationResult,
    radiant_temperature,
    solar_radiation,
    surface_temperature,
)

from .. import EPW_FILE, EXTERNAL_COMFORT_IDENTIFIER

EPW_OBJ = EPW(EPW_FILE)
GROUND_MATERIAL = Materials.LBT_AsphaltPavement.value
SHADE_MATERIAL = Materials.FABRIC.value


def test_solar_radiation():
    """_"""
    model = create_model(
        ground_material=GROUND_MATERIAL.to_lbt(),
        shade_material=SHADE_MATERIAL.to_lbt(),
        identifier=EXTERNAL_COMFORT_IDENTIFIER,
    )
    result = solar_radiation(model, EPW_OBJ)
    assert result["UnshadedUpTotalIrradiance"].average == pytest.approx(203, rel=1.5)


def test_radiant_temperature():
    """_"""
    assert (
        radiant_temperature(
            [EPW_OBJ.dry_bulb_temperature * 0.5, EPW_OBJ.dry_bulb_temperature * 2],
            view_factors=[0.25, 0.75],
        ).average
        == 16.924573102344482
    )


def test_surface_temperature():
    """_"""
    model = create_model(
        ground_material=GROUND_MATERIAL.to_lbt(),
        shade_material=SHADE_MATERIAL.to_lbt(),
        identifier=EXTERNAL_COMFORT_IDENTIFIER,
    )
    result = surface_temperature(model, EPW_OBJ)
    assert result["UnshadedUpTemperature"].average == EPW_OBJ.sky_temperature.average


def test_simulation_result():
    """_"""
    res = SimulationResult(
        EPW_FILE, GROUND_MATERIAL, SHADE_MATERIAL, EXTERNAL_COMFORT_IDENTIFIER
    ).run()

    # TODO - add value checks here for the result, to ensure variation is
    # captured when upstream methods change

    assert res.is_run()


def test_to_dict():
    """Test whether an object can be converted to a dictionary."""

    res = SimulationResult(
        EPW_FILE, GROUND_MATERIAL, SHADE_MATERIAL, EXTERNAL_COMFORT_IDENTIFIER
    )
    res_run = res.run()
    for obj in [res, res_run]:
        obj_dict = obj.to_dict()
        assert "_t" in obj_dict


def test_to_json():
    """Test whether an object can be converted to a json string."""
    res = SimulationResult(
        EPW_FILE, GROUND_MATERIAL, SHADE_MATERIAL, EXTERNAL_COMFORT_IDENTIFIER
    )
    res_run = res.run()
    for obj in [res, res_run]:
        obj_json = obj.to_json()
        assert '"_t":' in obj_json


def test_from_dict_native():
    """Test whether an object can be converted from a dictionary directly."""
    res = SimulationResult(
        EPW_FILE, GROUND_MATERIAL, SHADE_MATERIAL, EXTERNAL_COMFORT_IDENTIFIER
    )
    res_run = res.run()
    for obj in [res, res_run]:
        new_obj = type(obj).from_dict(obj.to_dict())
        assert isinstance(new_obj, type(obj))


def test_from_json_native():
    """Test whether an object can be converted from a json string directly."""
    res = SimulationResult(
        EPW_FILE, GROUND_MATERIAL, SHADE_MATERIAL, EXTERNAL_COMFORT_IDENTIFIER
    )
    res_run = res.run()
    for obj in [res, res_run]:
        new_obj = type(obj).from_json(obj.to_json())
        assert isinstance(new_obj, type(obj))
