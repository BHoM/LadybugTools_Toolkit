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
    source: str = field(repr=False, compare=True)

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
            source="LadybugTools",
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
            source=sanitised_dict["source"],
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
    source: str = field(repr=False, compare=True)

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
            source="LadybugTools",
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
            source=sanitised_dict["source"],
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
        source="LadybugTools - pre-release",
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
        source="Buro Happold",
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
        source="Buro Happold",
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
        source="Buro Happold",
    )

    # MATERIALS FROM ICE TOOL
    SD01: OpaqueMaterial = OpaqueMaterial(
        identifier="Sand (Quartzite, Beige/brown/black, 0.26 albedo)",
        thickness=0.01,
        conductivity=5.0,
        density=2600,
        specific_heat=690,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Rough",
        thermal_absorptance=0.92,
        solar_absorptance=0.74,
        visible_absorptance=0.74,
    )
    SD02: OpaqueMaterial = OpaqueMaterial(
        identifier="Sand (Quartzite, Beige/brown/black, 0.32 albedo)",
        thickness=0.01,
        conductivity=5.0,
        density=2600,
        specific_heat=690,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Rough",
        thermal_absorptance=0.96,
        solar_absorptance=0.6799999999999999,
        visible_absorptance=0.6799999999999999,
    )
    SD03: OpaqueMaterial = OpaqueMaterial(
        identifier="Sand (Quartzite, Brown, 0.25 albedo)",
        thickness=0.01,
        conductivity=5.0,
        density=2600,
        specific_heat=690,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Rough",
        thermal_absorptance=0.97,
        solar_absorptance=0.75,
        visible_absorptance=0.75,
    )
    ST01: OpaqueMaterial = OpaqueMaterial(
        identifier="Stone (Sandstone, Beige, 0.4 albedo)",
        thickness=0.01,
        conductivity=3.0,
        density=2000,
        specific_heat=745,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Smooth",
        thermal_absorptance=0.9,
        solar_absorptance=0.6,
        visible_absorptance=0.6,
    )
    ST02: OpaqueMaterial = OpaqueMaterial(
        identifier="Stone (Carboniferous Coral Limestone, Grey, 0.2 albedo)",
        thickness=0.01,
        conductivity=2.0,
        density=3250,
        specific_heat=1180,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Smooth",
        thermal_absorptance=0.94,
        solar_absorptance=0.8,
        visible_absorptance=0.8,
    )
    ST03: OpaqueMaterial = OpaqueMaterial(
        identifier="Stone (Sandstone, Yellow, 0.26 albedo)",
        thickness=0.01,
        conductivity=3.0,
        density=2000,
        specific_heat=745,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Smooth",
        thermal_absorptance=0.93,
        solar_absorptance=0.74,
        visible_absorptance=0.74,
    )
    ST04: OpaqueMaterial = OpaqueMaterial(
        identifier="Stone (Limestone, Beige, 0.68 albedo)",
        thickness=0.01,
        conductivity=2.0,
        density=3250,
        specific_heat=1180,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Smooth",
        thermal_absorptance=0.93,
        solar_absorptance=0.31999999999999995,
        visible_absorptance=0.31999999999999995,
    )
    ST05: OpaqueMaterial = OpaqueMaterial(
        identifier="Stone (Sandstone, Light grey, 0.46 albedo)",
        thickness=0.01,
        conductivity=3.0,
        density=2000,
        specific_heat=745,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Smooth",
        thermal_absorptance=0.92,
        solar_absorptance=0.54,
        visible_absorptance=0.54,
    )
    ST06: OpaqueMaterial = OpaqueMaterial(
        identifier="Stone (Granite, White/black, 0.48 albedo)",
        thickness=0.01,
        conductivity=2.7,
        density=2600,
        specific_heat=280,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.92,
        solar_absorptance=0.52,
        visible_absorptance=0.52,
    )
    ST07: OpaqueMaterial = OpaqueMaterial(
        identifier="Stone (Granite With Cement, White/red, 0.34 albedo)",
        thickness=0.01,
        conductivity=2.7,
        density=2600,
        specific_heat=280,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.93,
        solar_absorptance=0.6599999999999999,
        visible_absorptance=0.6599999999999999,
    )
    ST08: OpaqueMaterial = OpaqueMaterial(
        identifier="Stone (Granite With Cement, White/black, 0.41 albedo)",
        thickness=0.01,
        conductivity=2.7,
        density=2600,
        specific_heat=280,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.89,
        solar_absorptance=0.5900000000000001,
        visible_absorptance=0.5900000000000001,
    )
    ST09: OpaqueMaterial = OpaqueMaterial(
        identifier="Stone (Granite, White/red/black, 0.54 albedo)",
        thickness=0.01,
        conductivity=2.7,
        density=2600,
        specific_heat=280,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.93,
        solar_absorptance=0.45999999999999996,
        visible_absorptance=0.45999999999999996,
    )
    ST10: OpaqueMaterial = OpaqueMaterial(
        identifier="Stone (Granite, Red/black, 0.22 albedo)",
        thickness=0.01,
        conductivity=2.7,
        density=2600,
        specific_heat=280,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Smooth",
        thermal_absorptance=0.89,
        solar_absorptance=0.78,
        visible_absorptance=0.78,
    )
    AS01: OpaqueMaterial = OpaqueMaterial(
        identifier="Asphalt (Asphalt With Stone Aggregate, Black/grey, 0.21 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2300,
        specific_heat=1400,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.96,
        solar_absorptance=0.79,
        visible_absorptance=0.79,
    )
    AS02: OpaqueMaterial = OpaqueMaterial(
        identifier="Asphalt (Asphalt With Stone Aggregate, Black/grey, 0.18 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2300,
        specific_heat=1400,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.94,
        solar_absorptance=0.8200000000000001,
        visible_absorptance=0.8200000000000001,
    )
    AS03: OpaqueMaterial = OpaqueMaterial(
        identifier="Asphalt (Asphalt With Stone Aggregate, Black/grey, 0.21 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2300,
        specific_heat=1400,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.94,
        solar_absorptance=0.79,
        visible_absorptance=0.79,
    )
    AS04: OpaqueMaterial = OpaqueMaterial(
        identifier="Asphalt (Asphalt With Stone Aggregate, Black/grey, 0.18 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2300,
        specific_heat=1400,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.94,
        solar_absorptance=0.8200000000000001,
        visible_absorptance=0.8200000000000001,
    )
    AS05: OpaqueMaterial = OpaqueMaterial(
        identifier="Asphalt (Asphalt With Stone Aggregate, Black/grey, 0.19 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2300,
        specific_heat=1400,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.93,
        solar_absorptance=0.81,
        visible_absorptance=0.81,
    )
    AS06: OpaqueMaterial = OpaqueMaterial(
        identifier="Asphalt (Asphalt With Stone Aggregate, Black/grey, 0.12 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2300,
        specific_heat=1400,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.91,
        solar_absorptance=0.88,
        visible_absorptance=0.88,
    )
    TR01: OpaqueMaterial = OpaqueMaterial(
        identifier="Tarmac (Tarmac Roofing Paper, Grey, 0.07 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2300,
        specific_heat=1400,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.93,
        solar_absorptance=0.9299999999999999,
        visible_absorptance=0.9299999999999999,
    )
    TR02: OpaqueMaterial = OpaqueMaterial(
        identifier="Tarmac (Tarmac, Black, 0.13 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2300,
        specific_heat=1400,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.95,
        solar_absorptance=0.87,
        visible_absorptance=0.87,
    )
    TR03: OpaqueMaterial = OpaqueMaterial(
        identifier="Tarmac (Tarmac, Black, 0.08 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2300,
        specific_heat=1400,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.95,
        solar_absorptance=0.92,
        visible_absorptance=0.92,
    )
    TR04: OpaqueMaterial = OpaqueMaterial(
        identifier="Tarmac (Tarmac, Black, 0.1 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2300,
        specific_heat=1400,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.96,
        solar_absorptance=0.9,
        visible_absorptance=0.9,
    )
    CM01: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement, Grey/ochre, 0.29 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.94,
        solar_absorptance=0.71,
        visible_absorptance=0.71,
    )
    CC01: OpaqueMaterial = OpaqueMaterial(
        identifier="Concrete (Light Concrete, Grey/white, 0.21 albedo)",
        thickness=0.01,
        conductivity=0.21,
        density=2800,
        specific_heat=657,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.92,
        solar_absorptance=0.79,
        visible_absorptance=0.79,
    )
    CM02: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement, Grey, 0.23 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.91,
        solar_absorptance=0.77,
        visible_absorptance=0.77,
    )
    CC02: OpaqueMaterial = OpaqueMaterial(
        identifier="Concrete (Concrete, Grey, 0.37 albedo)",
        thickness=0.01,
        conductivity=0.21,
        density=2800,
        specific_heat=657,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.95,
        solar_absorptance=0.63,
        visible_absorptance=0.63,
    )
    CM03: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement, Grey, 0.41 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.95,
        solar_absorptance=0.5900000000000001,
        visible_absorptance=0.5900000000000001,
    )
    CC03: OpaqueMaterial = OpaqueMaterial(
        identifier="Concrete (Concrete, White, 0.42 albedo)",
        thickness=0.01,
        conductivity=0.21,
        density=2800,
        specific_heat=657,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.95,
        solar_absorptance=0.5800000000000001,
        visible_absorptance=0.5800000000000001,
    )
    CC04: OpaqueMaterial = OpaqueMaterial(
        identifier="Concrete (Concrete, Grey, 0.25 albedo)",
        thickness=0.01,
        conductivity=0.21,
        density=2800,
        specific_heat=657,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Rough",
        thermal_absorptance=0.95,
        solar_absorptance=0.75,
        visible_absorptance=0.75,
    )
    CM04: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement Brick, Yellow, 0.3 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.94,
        solar_absorptance=0.7,
        visible_absorptance=0.7,
    )
    CM05: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement Brick, With Sand, Black/light grey, 0.11 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.94,
        solar_absorptance=0.89,
        visible_absorptance=0.89,
    )
    CM06: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement Brick, Black, 0.09 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.95,
        solar_absorptance=0.91,
        visible_absorptance=0.91,
    )
    CL01: OpaqueMaterial = OpaqueMaterial(
        identifier="Clay (Brick Clay, With Cement, Red, 0.31 albedo)",
        thickness=0.01,
        conductivity=1.0,
        density=1300,
        specific_heat=1381,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.91,
        solar_absorptance=0.69,
        visible_absorptance=0.69,
    )
    CM07: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement Brick, Red, 0.17 albedo)",
        thickness=0.01,
        conductivity=0.25,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.94,
        solar_absorptance=0.83,
        visible_absorptance=0.83,
    )
    CM08: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement Brick, With Sand, Black, 0.2 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.89,
        solar_absorptance=0.8,
        visible_absorptance=0.8,
    )
    CM09: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement Brick, Light red, 0.22 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.94,
        solar_absorptance=0.78,
        visible_absorptance=0.78,
    )
    CL02: OpaqueMaterial = OpaqueMaterial(
        identifier="Clay (Brick Clay, Light red, 0.43 albedo)",
        thickness=0.01,
        conductivity=1.0,
        density=1300,
        specific_heat=1381,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.94,
        solar_absorptance=0.5700000000000001,
        visible_absorptance=0.5700000000000001,
    )
    CM10: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement, Red, 0.27 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.95,
        solar_absorptance=0.73,
        visible_absorptance=0.73,
    )
    CL03: OpaqueMaterial = OpaqueMaterial(
        identifier="Clay (Brick Clay, Painted, Red with beige and grey, 0.53 albedo)",
        thickness=0.01,
        conductivity=1.0,
        density=1300,
        specific_heat=1381,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.93,
        solar_absorptance=0.47,
        visible_absorptance=0.47,
    )
    CL04: OpaqueMaterial = OpaqueMaterial(
        identifier="Clay (Brick Clay, With Cement, Red/grey, 0.35 albedo)",
        thickness=0.01,
        conductivity=1.0,
        density=1300,
        specific_heat=1381,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.95,
        solar_absorptance=0.65,
        visible_absorptance=0.65,
    )
    CL05: OpaqueMaterial = OpaqueMaterial(
        identifier="Clay (Brick Clay, Painted, Red with white, 0.56 albedo)",
        thickness=0.01,
        conductivity=1.0,
        density=1300,
        specific_heat=1381,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Rough",
        thermal_absorptance=0.95,
        solar_absorptance=0.43999999999999995,
        visible_absorptance=0.43999999999999995,
    )
    CL06: OpaqueMaterial = OpaqueMaterial(
        identifier="Clay (Brick Clay, Red, 0.32 albedo)",
        thickness=0.01,
        conductivity=1.0,
        density=1300,
        specific_heat=1381,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.95,
        solar_absorptance=0.6799999999999999,
        visible_absorptance=0.6799999999999999,
    )
    CL07: OpaqueMaterial = OpaqueMaterial(
        identifier="Clay (Brick Clay, Yellow/grey, 0.43 albedo)",
        thickness=0.01,
        conductivity=1.0,
        density=2600,
        specific_heat=1381,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.96,
        solar_absorptance=0.5700000000000001,
        visible_absorptance=0.5700000000000001,
    )
    SL01: OpaqueMaterial = OpaqueMaterial(
        identifier="Slate (Slate, Black, 0.09 albedo)",
        thickness=0.01,
        conductivity=2.0,
        density=2771,
        specific_heat=760,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.9,
        solar_absorptance=0.91,
        visible_absorptance=0.91,
    )
    SL02: OpaqueMaterial = OpaqueMaterial(
        identifier="Slate (Slate, Black, 0.14 albedo)",
        thickness=0.01,
        conductivity=2.0,
        density=2771,
        specific_heat=760,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Smooth",
        thermal_absorptance=0.93,
        solar_absorptance=0.86,
        visible_absorptance=0.86,
    )
    FB01: OpaqueMaterial = OpaqueMaterial(
        identifier="Fibre Cement (Fibre Cement, Black, 0.05 albedo)",
        thickness=0.01,
        conductivity=0.25,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.94,
        solar_absorptance=0.95,
        visible_absorptance=0.95,
    )
    FB02: OpaqueMaterial = OpaqueMaterial(
        identifier="Fibre Cement (Fibre Cement, Black, 0.06 albedo)",
        thickness=0.01,
        conductivity=0.25,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.95,
        solar_absorptance=0.94,
        visible_absorptance=0.94,
    )
    CR01: OpaqueMaterial = OpaqueMaterial(
        identifier="Ceramic (Ceramic, Red, 0.31 albedo)",
        thickness=0.01,
        conductivity=5.0,
        density=3800,
        specific_heat=790,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.93,
        solar_absorptance=0.69,
        visible_absorptance=0.69,
    )
    CR02: OpaqueMaterial = OpaqueMaterial(
        identifier="Ceramic (Ceramic, Brown, 0.2 albedo)",
        thickness=0.01,
        conductivity=5.0,
        density=3800,
        specific_heat=790,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.93,
        solar_absorptance=0.8,
        visible_absorptance=0.8,
    )
    CR03: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement, Rustic red, 0.32 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.91,
        solar_absorptance=0.6799999999999999,
        visible_absorptance=0.6799999999999999,
    )
    CR04: OpaqueMaterial = OpaqueMaterial(
        identifier="Ceramic (Ceramic, Burnt red, 0.24 albedo)",
        thickness=0.01,
        conductivity=1.99,
        density=3800,
        specific_heat=908,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.92,
        solar_absorptance=0.76,
        visible_absorptance=0.76,
    )
    CM11: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement, Rustic red with dark shading, 0.17 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Smooth",
        thermal_absorptance=0.96,
        solar_absorptance=0.83,
        visible_absorptance=0.83,
    )
    CM12: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement, Slate grey, 0.12 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.94,
        solar_absorptance=0.88,
        visible_absorptance=0.88,
    )
    CR05: OpaqueMaterial = OpaqueMaterial(
        identifier="Ceramic (Ceramic, Black, 0.16 albedo)",
        thickness=0.01,
        conductivity=5.0,
        density=3800,
        specific_heat=790,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.92,
        solar_absorptance=0.84,
        visible_absorptance=0.84,
    )
    CM13: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement, Rustic red, 0.26 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.95,
        solar_absorptance=0.74,
        visible_absorptance=0.74,
    )
    CM14: OpaqueMaterial = OpaqueMaterial(
        identifier="Cement (Cement, Autumn red, 0.19 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.92,
        solar_absorptance=0.81,
        visible_absorptance=0.81,
    )
    CR06: OpaqueMaterial = OpaqueMaterial(
        identifier="Ceramic (Ceramic, Red, 0.19 albedo)",
        thickness=0.01,
        conductivity=5.0,
        density=3800,
        specific_heat=790,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.95,
        solar_absorptance=0.81,
        visible_absorptance=0.81,
    )
    CR07: OpaqueMaterial = OpaqueMaterial(
        identifier="Ceramic (Ceramic, Red, 0.13 albedo)",
        thickness=0.01,
        conductivity=5.0,
        density=3800,
        specific_heat=790,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.95,
        solar_absorptance=0.87,
        visible_absorptance=0.87,
    )
    CR08: OpaqueMaterial = OpaqueMaterial(
        identifier="Ceramic (Ceramic, Red, 0.12 albedo)",
        thickness=0.01,
        conductivity=5.0,
        density=3800,
        specific_heat=790,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.95,
        solar_absorptance=0.88,
        visible_absorptance=0.88,
    )
    MT01: OpaqueMaterial = OpaqueMaterial(
        identifier="Metal (Aluminium Plus Zinc, Grey, dull, 0.36 albedo)",
        thickness=0.01,
        conductivity=45.0,
        density=7000,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.58,
        solar_absorptance=0.64,
        visible_absorptance=0.64,
    )
    MT02: OpaqueMaterial = OpaqueMaterial(
        identifier="Metal (Aluminium, Stucco, Grey, shiny, 0.25 albedo)",
        thickness=0.01,
        conductivity=45.0,
        density=7000,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.16,
        solar_absorptance=0.75,
        visible_absorptance=0.75,
    )
    MT03: OpaqueMaterial = OpaqueMaterial(
        identifier="Metal (Metal With Paint, Green, 0.11 albedo)",
        thickness=0.01,
        conductivity=45.0,
        density=7000,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.93,
        solar_absorptance=0.89,
        visible_absorptance=0.89,
    )
    MT04: OpaqueMaterial = OpaqueMaterial(
        identifier="Metal (Metal With Paint, Copper patina, 0.45 albedo)",
        thickness=0.01,
        conductivity=0.8,
        density=2100,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.94,
        solar_absorptance=0.55,
        visible_absorptance=0.55,
    )
    MT05: OpaqueMaterial = OpaqueMaterial(
        identifier="Metal (Metal With Paint, Slate grey, 0.12 albedo)",
        thickness=0.01,
        conductivity=1.0,
        density=2200,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.94,
        solar_absorptance=0.88,
        visible_absorptance=0.88,
    )
    MT06: OpaqueMaterial = OpaqueMaterial(
        identifier="Metal (Aluminium, Grey, 0.26 albedo)",
        thickness=0.01,
        conductivity=45.0,
        density=2300,
        specific_heat=1030,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.81,
        solar_absorptance=0.74,
        visible_absorptance=0.74,
    )
    MT07: OpaqueMaterial = OpaqueMaterial(
        identifier="Metal (Lead, Grey, 0.21 albedo)",
        thickness=0.01,
        conductivity=45.0,
        density=2300,
        specific_heat=1400,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.86,
        solar_absorptance=0.79,
        visible_absorptance=0.79,
    )
    MT08: OpaqueMaterial = OpaqueMaterial(
        identifier="Metal (Iron, Black, 0.05 albedo)",
        thickness=0.01,
        conductivity=45.0,
        density=2300,
        specific_heat=2580,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumSmooth",
        thermal_absorptance=0.97,
        solar_absorptance=0.95,
        visible_absorptance=0.95,
    )
    PV01: OpaqueMaterial = OpaqueMaterial(
        identifier="Pvc (Pvc Roofing Material, Lead grey, 0.08 albedo)",
        thickness=0.01,
        conductivity=0.19,
        density=2300,
        specific_heat=1600,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.94,
        solar_absorptance=0.92,
        visible_absorptance=0.92,
    )
    PV02: OpaqueMaterial = OpaqueMaterial(
        identifier="Pvc (Pvc Roofing Material, Light grey, 0.43 albedo)",
        thickness=0.01,
        conductivity=0.19,
        density=2300,
        specific_heat=1600,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.93,
        solar_absorptance=0.5700000000000001,
        visible_absorptance=0.5700000000000001,
    )
    PV03: OpaqueMaterial = OpaqueMaterial(
        identifier="Pvc (Pvc Roofing Material, Copper brown, 0.29 albedo)",
        thickness=0.01,
        conductivity=0.19,
        density=2300,
        specific_heat=1600,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="Rough",
        thermal_absorptance=0.94,
        solar_absorptance=0.71,
        visible_absorptance=0.71,
    )
    PV04: OpaqueMaterial = OpaqueMaterial(
        identifier="Pvc (Pvc Roofing Material, Azure blue, 0.14 albedo)",
        thickness=0.01,
        conductivity=0.19,
        density=2300,
        specific_heat=1600,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.94,
        solar_absorptance=0.86,
        visible_absorptance=0.86,
    )
    PV05: OpaqueMaterial = OpaqueMaterial(
        identifier="Pvc (Pvc Roofing Material, Copper brown, 0.17 albedo)",
        thickness=0.01,
        conductivity=0.19,
        density=2300,
        specific_heat=1600,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.94,
        solar_absorptance=0.83,
        visible_absorptance=0.83,
    )
    PV06: OpaqueMaterial = OpaqueMaterial(
        identifier="Pvc (Pvc Roofing Material, Copper patina, 0.28 albedo)",
        thickness=0.01,
        conductivity=0.19,
        density=2300,
        specific_heat=1600,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.94,
        solar_absorptance=0.72,
        visible_absorptance=0.72,
    )
    VG01: OpaqueMaterial = OpaqueMaterial(
        identifier="Grass (Grass Green Watered, Green, 0.27 albedo)",
        thickness=0.01,
        conductivity=0.28,
        density=1600,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.98,
        solar_absorptance=0.73,
        visible_absorptance=0.73,
    )
    VG02: OpaqueMaterial = OpaqueMaterial(
        identifier="Grass (Grass Dry , Yellow, 0.17 albedo)",
        thickness=0.01,
        conductivity=0.28,
        density=1600,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="MediumRough",
        thermal_absorptance=0.98,
        solar_absorptance=0.83,
        visible_absorptance=0.83,
    )
    VG03: OpaqueMaterial = OpaqueMaterial(
        identifier="Dense Forest (Dense Forest, Green, 0.27 albedo)",
        thickness=0.01,
        conductivity=0.28,
        density=1600,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        roughness="VeryRough",
        thermal_absorptance=0.98,
        solar_absorptance=0.73,
        visible_absorptance=0.73,
    )
    VG04: OpaqueMaterial = OpaqueMaterial(
        identifier="Wood (Tree, Brown, 0.4 albedo)",
        thickness=0.01,
        conductivity=0.11,
        density=545,
        specific_heat=200,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated nan)",
        roughness="Smooth",
        thermal_absorptance=0.9,
        solar_absorptance=0.6,
        visible_absorptance=0.6,
    )
    VS01: OpaqueMaterial = OpaqueMaterial(
        identifier="Soil (Wood, Brown, 0.35 albedo)",
        thickness=0.01,
        conductivity=0.28,
        density=1600,
        specific_heat=800,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated nan)",
        roughness="VeryRough",
        thermal_absorptance=0.95,
        solar_absorptance=0.65,
        visible_absorptance=0.65,
    )
    WT01: OpaqueMaterial = OpaqueMaterial(
        identifier="Water (Water Small, -, 0.07 albedo)",
        thickness=0.01,
        conductivity=0.68,
        density=997,
        specific_heat=4184,
        source="https://github.com/Art-Ev/ICEtool_sources (last updated nan)",
        roughness="MediumRough",
        thermal_absorptance=0.95,
        solar_absorptance=0.9299999999999999,
        visible_absorptance=0.9299999999999999,
    )
