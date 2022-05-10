from enum import Enum
from typing import Dict

from honeybee_energy.lib.materials import opaque_material_by_identifier
from honeybee_energy.material.opaque import EnergyMaterial, EnergyMaterialVegetation

"""
Material properties defined here are from a range of sources. 

If they cannot be found in the locations given below, then the material must have 
been added based on precedent, project requirement and research into representative 
properties of the named material:

- ASHRAE_2005_HOF_Materials.idf (ASHRAE Handbook of Fundamentals, 2005, Chapter 30, Table 19 and Table 22)
- https://github.com/ladybug-tools/honeybee-energy-standards
"""


class Material(Enum):
    """Enum of predefined materials for use in External Comfort simulation workflow."""

    AsphaltPavement: EnergyMaterial = opaque_material_by_identifier("Asphalt Pavement")
    ConcretePavement: EnergyMaterial = opaque_material_by_identifier(
        "Concrete Pavement"
    )
    DryDust: EnergyMaterial = opaque_material_by_identifier("Dry Dust")
    DrySand: EnergyMaterial = opaque_material_by_identifier("Dry Sand")
    # GrassyLawn: EnergyMaterial = opaque_material_by_identifier("Grassy Lawn")
    MetalSurface: EnergyMaterial = opaque_material_by_identifier("Metal Surface")
    MetalSurfaceHighlyReflective: EnergyMaterial = opaque_material_by_identifier(
        "Metal Roof Surface - Highly Reflective"
    )
    MoistSoil: EnergyMaterial = opaque_material_by_identifier("Moist Soil")
    Mud: EnergyMaterial = opaque_material_by_identifier("Mud")
    SolidRock: EnergyMaterial = opaque_material_by_identifier("Solid Rock")
    WoodSiding: EnergyMaterial = opaque_material_by_identifier("Wood Siding")
    # CUSTOM MATERIALS
    Fabric: EnergyMaterial = EnergyMaterial(
        identifier="Fabric",
        roughness="Smooth",
        thickness=0.002,
        conductivity=0.06,
        density=500.0,
        specific_heat=1800.0,
        thermal_absorptance=0.89,
        solar_absorptance=0.5,
        visible_absorptance=0.5,
    )
    Shrubs: EnergyMaterialVegetation = EnergyMaterialVegetation(
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
    )
    Travertine: EnergyMaterial = EnergyMaterial(
        identifier="Travertine",
        roughness="MediumRough",
        thickness=0.2,
        conductivity=3.2,
        density=2700.0,
        specific_heat=790.0,
        thermal_absorptance=0.96,
        solar_absorptance=0.55,
        visible_absorptance=0.55,
    )
