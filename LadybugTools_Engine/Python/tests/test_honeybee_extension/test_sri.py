"""Test methods for SRI calculations."""

import pytest
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.material.opaque import EnergyMaterial
from ladybugtools_toolkit.honeybee_extension.sri import (
    calculate_sri, construction_sri, estimate_sri_from_color,
    estimate_sri_properties, material_sri)

MATERIAL = EnergyMaterial("sri_material", 1, 1, 100, 100, solar_absorptance=0.5, thermal_absorptance=0.9)
CONSTRUCTION = OpaqueConstruction("sri_construction", [MATERIAL])


def test_calculate_sri() -> None:
    """_"""
    assert calculate_sri(solar_reflectance=0.35, thermal_emittance=0.85, wind_speed=4) == pytest.approx(36.354029062087186, 0.001)
    assert calculate_sri(solar_reflectance=0.35, thermal_emittance=0.85, wind_speed=0, air_temperature=40) == pytest.approx(
        33.394936240990575, 0.001
    )
    assert calculate_sri(
        solar_reflectance=0.35,
        thermal_emittance=0.85,
        wind_speed=0,
        air_temperature=40,
        sky_temperature=0.5,
        insolation=540,
    ) == pytest.approx(33.83853104051297, 0.001)


def test_construction_sri() -> None:
    """_"""
    assert construction_sri(CONSTRUCTION) == pytest.approx(58.414795244385736, 0.001)


def test_material_sri() -> None:
    """_"""
    assert material_sri(MATERIAL) == pytest.approx(58.414795244385736, 0.001)


def test_sri_from_color() -> None:
    """_"""
    color_hexs = [
        "#82898F",
        "#82898F",
        "#A18594",
        "#A18594",
        "#FFFFFF",
        "#FFFFFF",
        "#C1876B",
        "#C1876B",
        "#000000",
        "#000000",
    ]
    thermal_emittances = [
        0.1,
        0.9,
        0.1,
        0.5,
        0.5,
        0.9,
        0.1,
        0.5,
        0.1,
        0.5,
    ]
    results = [
        0,
        44.94055482166446,
        0.607661822985469,
        28.480845442536328,
        79.7886393659181,
        89.6433289299868,
        3.5667107001321003,
        30.779392338177015,
        0,
        6.737120211360634,
    ]
    for color_hex, thermal_emittance, result in zip(color_hexs, thermal_emittances, results):
        assert estimate_sri_from_color(color_hex=color_hex, thermal_emittance=thermal_emittance) == pytest.approx(result, 0.001)


def test_estimate_sri_properties() -> None:
    """_"""
    for target_sri, target_emittance, result in [
        (10, 0.1, (0.7, 0.1)),
        (10, 0.5, (0.785, 0.5)),
        (10, 0.9, (0.87, 0.9)),
        (50, 0.1, (0.405, 0.1)),
        (50, 0.5, (0.49, 0.5)),
        (50, 0.9, (0.575, 0.9)),
        (100, 0.1, (0.03, 0.1)),
        (100, 0.5, (0.115, 0.5)),
        (100, 0.9, (0.2, 0.9)),
    ]:
        assert estimate_sri_properties(target_sri=target_sri, target_emittance=target_emittance) == pytest.approx(result, 0.001)
