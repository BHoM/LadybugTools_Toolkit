from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Union

from honeybee_energy.lib.materials import opaque_material_by_identifier
from honeybee_energy.material.opaque import EnergyMaterial, EnergyMaterialVegetation

from ..bhomutil.bhom_object import BHoMObject, bhom_dict_to_dict


@dataclass(init=True, repr=True, eq=True)
class OpaqueMaterial(BHoMObject):  # pylint: disable=invalid-name
    """An object representing a material."""

    identifier: str = field(repr=True, compare=True)
    thickness: float = field(repr=False, compare=True)
    conductivity: float = field(repr=False, compare=True)
    density: float = field(repr=False, compare=True)
    specific_heat: float = field(repr=False, compare=True)

    roughness: str = field(repr=False, compare=True, default="MediumRough")
    thermal_absorptance: float = field(repr=False, compare=True, default=0.9)
    solar_absorptance: float = field(repr=False, compare=True, default=0.7)
    visible_absorptance: float = field(repr=False, compare=True, default=0.7)

    _t: str = field(
        init=False,
        repr=False,
        compare=True,
        default="BH.oM.LadybugTools.OpaqueMaterial",
    )

    def to_lbt(self) -> EnergyMaterial:
        """Return this object as its LBT equivalent."""
        return EnergyMaterial(
            identifier=self.identifier,
            thickness=self.thickness,
            conductivity=self.conductivity,
            density=self.density,
            specific_heat=self.specific_heat,
            roughness=self.roughness,
            thermal_absorptance=self.thermal_absorptance,
            solar_absorptance=self.solar_absorptance,
            visible_absorptance=self.visible_absorptance,
        )

    @classmethod
    def from_lbt(cls, lbt_material: EnergyMaterial) -> OpaqueMaterial:
        """Create this object from its LBT equivalent."""
        return cls(
            identifier=lbt_material.identifier,
            thickness=lbt_material.thickness,
            conductivity=lbt_material.conductivity,
            density=lbt_material.density,
            specific_heat=lbt_material.specific_heat,
            roughness=lbt_material.roughness,
            thermal_absorptance=lbt_material.thermal_absorptance,
            solar_absorptance=lbt_material.solar_absorptance,
            visible_absorptance=lbt_material.visible_absorptance,
        )

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> OpaqueMaterial:
        """Create this object from a dictionary."""

        sanitised_dict = bhom_dict_to_dict(dictionary)
        sanitised_dict.pop("_t", None)

        return cls(
            identifier=sanitised_dict["identifier"],
            thickness=sanitised_dict["thickness"],
            conductivity=sanitised_dict["conductivity"],
            density=sanitised_dict["density"],
            specific_heat=sanitised_dict["specific_heat"],
            roughness=sanitised_dict["roughness"],
            thermal_absorptance=sanitised_dict["thermal_absorptance"],
            solar_absorptance=sanitised_dict["solar_absorptance"],
            visible_absorptance=sanitised_dict["visible_absorptance"],
        )

    @classmethod
    def from_json(cls, json_string: str) -> OpaqueMaterial:
        """Create this object from a JSON string."""

        dictionary = json.loads(json_string)

        return cls.from_dict(dictionary)


@dataclass(init=True, repr=True, eq=True)
class OpaqueVegetationMaterial(BHoMObject):  # pylint: disable=invalid-name
    """An object representing a vegetation material."""

    identifier: float = field(repr=True, compare=True)

    thickness: float = field(repr=False, compare=True, default=0.1)
    conductivity: float = field(repr=False, compare=True, default=0.35)
    density: float = field(repr=False, compare=True, default=1100)
    specific_heat: float = field(repr=False, compare=True, default=1200)
    roughness: float = field(repr=False, compare=True, default="MediumRough")
    soil_thermal_absorptance: float = field(repr=False, compare=True, default=0.9)
    soil_solar_absorptance: float = field(repr=False, compare=True, default=0.7)
    soil_visible_absorptance: float = field(repr=False, compare=True, default=0.7)
    plant_height: float = field(repr=False, compare=True, default=0.2)
    leaf_area_index: float = field(repr=False, compare=True, default=1)
    leaf_reflectivity: float = field(repr=False, compare=True, default=0.22)
    leaf_emissivity: float = field(repr=False, compare=True, default=0.95)
    min_stomatal_resist: float = field(repr=False, compare=True, default=180)

    _t: str = field(
        init=False,
        repr=False,
        compare=True,
        default="BH.oM.LadybugTools.OpaqueVegetationMaterial",
    )

    def to_lbt(self) -> EnergyMaterialVegetation:
        """Return this object as its LBT equivalent."""
        return EnergyMaterialVegetation(
            identifier=self.identifier,
            thickness=self.thickness,
            conductivity=self.conductivity,
            density=self.density,
            specific_heat=self.specific_heat,
            roughness=self.roughness,
            soil_thermal_absorptance=self.soil_thermal_absorptance,
            soil_solar_absorptance=self.soil_solar_absorptance,
            soil_visible_absorptance=self.soil_visible_absorptance,
            plant_height=self.plant_height,
            leaf_area_index=self.leaf_area_index,
            leaf_reflectivity=self.leaf_reflectivity,
            leaf_emissivity=self.leaf_emissivity,
            min_stomatal_resist=self.min_stomatal_resist,
        )

    @classmethod
    def from_lbt(cls, material: EnergyMaterialVegetation):
        """Create this object from its LBT equivalent."""
        return cls(
            identifier=material.identifier,
            thickness=material.thickness,
            conductivity=material.conductivity,
            density=material.density,
            specific_heat=material.specific_heat,
            roughness=material.roughness,
            soil_thermal_absorptance=material.soil_thermal_absorptance,
            soil_solar_absorptance=material.soil_solar_absorptance,
            soil_visible_absorptance=material.soil_visible_absorptance,
            plant_height=material.plant_height,
            leaf_area_index=material.leaf_area_index,
            leaf_reflectivity=material.leaf_reflectivity,
            leaf_emissivity=material.leaf_emissivity,
            min_stomatal_resist=material.min_stomatal_resist,
        )

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> OpaqueVegetationMaterial:
        """Create this object from a dictionary."""

        sanitised_dict = bhom_dict_to_dict(dictionary)
        sanitised_dict.pop("_t", None)

        return cls(
            identifier=sanitised_dict["identifier"],
            thickness=sanitised_dict["thickness"],
            conductivity=sanitised_dict["conductivity"],
            density=sanitised_dict["density"],
            specific_heat=sanitised_dict["specific_heat"],
            roughness=sanitised_dict["roughness"],
            soil_thermal_absorptance=sanitised_dict["soil_thermal_absorptance"],
            soil_solar_absorptance=sanitised_dict["soil_solar_absorptance"],
            soil_visible_absorptance=sanitised_dict["soil_visible_absorptance"],
            plant_height=sanitised_dict["plant_height"],
            leaf_area_index=sanitised_dict["leaf_area_index"],
            leaf_reflectivity=sanitised_dict["leaf_reflectivity"],
            leaf_emissivity=sanitised_dict["leaf_emissivity"],
            min_stomatal_resist=sanitised_dict["min_stomatal_resist"],
        )

    @classmethod
    def from_json(cls, json_string: str) -> OpaqueVegetationMaterial:
        """Create this object from a JSON string."""

        dictionary = json.loads(json_string)

        return cls.from_dict(dictionary)


def material_from_dict(
    dictionary: Dict[str, Any]
) -> Union[OpaqueMaterial, OpaqueVegetationMaterial]:
    """Attempt to convert a dictionary into a material object."""
    try:
        return OpaqueMaterial.from_dict(dictionary)
    except Exception:  # pylint: disable=[broad-except]
        return OpaqueVegetationMaterial.from_dict(dictionary)


class Materials(Enum):
    """
    Enum of predefined materials for use in External Comfort simulation
    workflow.

    Material properties defined here are from a range of sources.

    If they cannot be found in the locations given below, then the material
    must have been added based on precedent, project requirement and research
    into representative properties of the named material:

    - ASHRAE_2005_HOF_Materials.idf (ASHRAE Handbook of Fundamentals, 2005,
        Chapter 30, Table 19 and Table 22)
    - https://github.com/ladybug-tools/honeybee-energy-standards
    """

    ASPHALT_PAVEMENT: OpaqueMaterial = OpaqueMaterial.from_lbt(
        opaque_material_by_identifier("Asphalt Pavement")
    )
    CONCRETE_PAVEMENT: OpaqueMaterial = OpaqueMaterial.from_lbt(
        opaque_material_by_identifier("Concrete Pavement")
    )
    DRY_DUST: OpaqueMaterial = OpaqueMaterial.from_lbt(
        opaque_material_by_identifier("Dry Dust")
    )
    DRY_SAND: OpaqueMaterial = OpaqueMaterial.from_lbt(
        opaque_material_by_identifier("Dry Sand")
    )
    GRASSY_LAWN: OpaqueVegetationMaterial = OpaqueVegetationMaterial(
        identifier="Grassy Lawn",
        roughness="MediumRough",
        thickness=0.1,
        conductivity=0.35,
        density=1100,
        specific_heat=1200,
        soil_thermal_absorptance=0.9,
        soil_solar_absorptance=0.7,
        soil_visible_absorptance=0.7,
        plant_height=0.2,
        leaf_area_index=1.0,
        leaf_reflectivity=0.22,
        leaf_emissivity=0.95,
        min_stomatal_resist=180,
    )
    METAL: OpaqueMaterial = OpaqueMaterial.from_lbt(
        opaque_material_by_identifier("Metal Surface")
    )
    METAL_REFLECTIVE: OpaqueMaterial = OpaqueMaterial.from_lbt(
        opaque_material_by_identifier("Metal Roof Surface - Highly Reflective")
    )
    MOIST_SOIL: OpaqueMaterial = OpaqueMaterial.from_lbt(
        opaque_material_by_identifier("Moist Soil")
    )
    MUD: OpaqueMaterial = OpaqueMaterial.from_lbt(opaque_material_by_identifier("Mud"))
    SOLID_ROCK: OpaqueMaterial = OpaqueMaterial.from_lbt(
        opaque_material_by_identifier("Solid Rock")
    )
    WOOD_SIDING: OpaqueMaterial = OpaqueMaterial.from_lbt(
        opaque_material_by_identifier("Wood Siding")
    )

    # CUSTOM MATERIALS

    FABRIC: OpaqueMaterial = OpaqueMaterial(
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
    SHRUBS: OpaqueVegetationMaterial = OpaqueVegetationMaterial(
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
    TRAVERTINE: OpaqueMaterial = OpaqueMaterial(
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
