"""Methods for creating and manipulating materials.
"""

# pylint: disable=E0401
from difflib import get_close_matches
from pathlib import Path

# pylint: enable=E0401

import pandas as pd
from honeybee_energy.lib.materials import (
    OPAQUE_MATERIALS,
    opaque_material_by_identifier,
)
from honeybee_energy.material.opaque import (
    EnergyMaterial,
    EnergyMaterialVegetation,
    _EnergyMaterialOpaqueBase,
)
from ..bhom import decorator_factory


def _ice_tool_materials(
    path: Path = Path(__file__).parent.parent.parent
    / "data"
    / r"ICE_database_sources.xlsx",
) -> list[_EnergyMaterialOpaqueBase]:
    """Load predefined materials from the ICE tool database.

    Args:
        path (Path):
            Path to the ICE tool database.
    Returns:
        List of EnergyMaterial objects.
    """

    df = pd.read_excel(path, "Basic")

    _materials = []
    for _, row in df.iterrows():
        try:
            mat = EnergyMaterial(
                identifier=f"{row['id']} - {row['specific name']} ({row['color']} {row['age/rugosite']})".replace(
                    ",", " "
                ),
                thickness=0.01,
                conductivity=row["Thermal condutivity (W/m.K)"],
                density=row["density (kg/m³)"],
                specific_heat=row["Specific heat capacity  Cp (J/kg.K)"],
                roughness="MediumRough",
                thermal_absorptance=row["emissivitty"],
                solar_absorptance=1 - row["albedo"],
                visible_absorptance=1 - row["albedo"],
            )
            _materials.append(mat)
        except AssertionError as _:
            pass
    return _materials


def _lbt_materials() -> list[_EnergyMaterialOpaqueBase]:
    """Load predefined LBT materials into a list."""
    _materials = [
        opaque_material_by_identifier(material_name)
        for material_name in OPAQUE_MATERIALS
    ]
    return [
        i
        for i in _materials
        if isinstance(i, (EnergyMaterial, EnergyMaterialVegetation))
    ]


def _custom_materials() -> list[_EnergyMaterialOpaqueBase]:
    """Create a list of custom materials."""
    return [
        EnergyMaterial(
            identifier="Fabric",
            roughness="Smooth",
            thickness=0.002,
            conductivity=0.06,
            density=500.0,
            specific_heat=1800.0,
            thermal_absorptance=0.89,
            solar_absorptance=0.5,
            visible_absorptance=0.5,
        ),
        EnergyMaterialVegetation(
            identifier="Shrubs",
            roughness="Rough",
            thickness=0.1,
            conductivity=0.35,
            density=1260,
            specific_heat=1100,
            soil_thermal_absorptance=0.9,
            soil_solar_absorptance=0.7,
            soil_visible_absorptance=0.7,
            plant_height=0.2,
            leaf_area_index=2.08,
            leaf_reflectivity=0.21,
            leaf_emissivity=0.95,
            min_stomatal_resist=180,
        ),
        EnergyMaterial(
            identifier="Travertine",
            roughness="MediumRough",
            thickness=0.2,
            conductivity=3.2,
            density=2700.0,
            specific_heat=790.0,
            thermal_absorptance=0.96,
            solar_absorptance=0.55,
            visible_absorptance=0.55,
        ),
    ]


def _material_equality(
    material1: _EnergyMaterialOpaqueBase, material2: _EnergyMaterialOpaqueBase
) -> bool:
    """Determine if two materials are equal.

    Args:
        material1 (_EnergyMaterialOpaqueBase): The first material.
        material2 (_EnergyMaterialOpaqueBase): The second material.

    Returns:
        bool: True if the materials are equal, False otherwise.
    """

    return str(material1) == str(material2)


@decorator_factory()
def materials() -> list[_EnergyMaterialOpaqueBase]:
    """Return a list all the materials in the library.

    Notes:
        This function is used to create the Materials enum which is a combination of all
        LBT provided generic materials, custom materials, and materials from the ICE tool.
    """

    all_materials = []
    all_materials.extend(_lbt_materials())
    all_materials.extend(_custom_materials())
    all_materials.extend(_ice_tool_materials())

    # remove dodgy materials
    filtered_materials = []
    for mat in all_materials:
        if " air " in mat.identifier.lower():
            continue
        if " gap " in mat.identifier.lower():
            continue
        if " void " in mat.identifier.lower():
            continue
        filtered_materials.append(mat)
    return filtered_materials


@decorator_factory()
def get_material(material_identifier: str) -> _EnergyMaterialOpaqueBase:
    """Get a material from its name.

    Args:
        material_identifier (str): The identifier of the material.

    Returns:
        _EnergyMaterialBase: The material.
    """
    # create dict of materials
    d = {mat.identifier: mat for mat in materials()}
    try:
        return d[material_identifier]
    except KeyError:
        try:
            return opaque_material_by_identifier(material_identifier)
        except ValueError as exc:
            possible_materials = get_close_matches(
                material_identifier, d.keys(), cutoff=0.1
            )
            raise KeyError(
                f"Unknown material name: '{material_identifier}'. Did you mean {possible_materials}"
            ) from exc
