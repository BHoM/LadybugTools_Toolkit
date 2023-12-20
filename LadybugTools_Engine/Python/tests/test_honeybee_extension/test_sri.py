import pytest
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.material.opaque import EnergyMaterial
from ladybugtools_toolkit.honeybee_extension.sri import (
    calculate_sri,
    construction_sri,
    material_sri,
)

MATERIAL = EnergyMaterial(
    "sri_material", 1, 1, 100, 100, solar_absorptance=0.5, thermal_absorptance=0.9
)
CONSTRUCTION = OpaqueConstruction("sri_construction", [MATERIAL])


def test_calculate_sri() -> None:
    """_"""
    assert calculate_sri(
        solar_reflectance=0.35, thermal_emittance=0.85, wind_speed=4
    ) == pytest.approx(36.354029062087186, 0.001)
    assert calculate_sri(
        solar_reflectance=0.35, thermal_emittance=0.85, wind_speed=0, air_temperature=40
    ) == pytest.approx(33.394936240990575, 0.001)
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
