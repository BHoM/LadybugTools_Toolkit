from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Union

from honeybee_energy.material.opaque import EnergyMaterial, EnergyMaterialVegetation

from ..bhomutil.bhom_object import BHoMObject, bhom_dict_to_dict

# NOTE: This is the version of the Honeybee Material Standards that this module referneces. If these standards are updated, this version number should be updated as well.
HONEYBEE_MATERIAL_STANDARDS_VERSION = "2.2.6"


@dataclass(init=True, repr=True, eq=True)
class OpaqueMaterial(BHoMObject):
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
class OpaqueVegetationMaterial(BHoMObject):
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
    # pylint disable=broad-except
    try:
        return OpaqueMaterial.from_dict(dictionary)
    except Exception:
        return OpaqueVegetationMaterial.from_dict(dictionary)
    # pylint enable=broad-except


def create_material(
    obj: Any,
) -> Union[Dict[str, Any], EnergyMaterial, EnergyMaterialVegetation]:
    """Attempt to convert an object into a material object suitable for LBT_Tk External Comfort workflows."""
    if isinstance(obj, dict):
        # pylint disable=broad-except
        try:
            return OpaqueMaterial.from_lbt(EnergyMaterial.from_dict(obj))
        except:
            try:
                return OpaqueVegetationMaterial.from_lbt(
                    EnergyMaterialVegetation.from_dict(obj)
                )
            except:
                try:
                    return OpaqueMaterial.from_dict(obj)
                except:
                    return OpaqueVegetationMaterial.from_dict(obj)
        # pylint enable=broad-except

    if isinstance(obj, EnergyMaterial):
        return OpaqueMaterial.from_lbt(obj)
    if isinstance(obj, EnergyMaterialVegetation):
        return OpaqueVegetationMaterial.from_lbt(obj)

    raise ValueError("input not of known type")


# pylint: disable=C0103
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

    # materials from LBT
    LBT_12InGypsumBoard: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "1/2 in. Gypsum Board",
            "roughness": "Smooth",
            "thickness": 0.0127,
            "conductivity": 0.15989299909405463,
            "density": 800.0018291911765,
            "specific_heat": 1089.2971854559414,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.5,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_12InGypsum: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "1/2IN Gypsum",
            "roughness": "Smooth",
            "thickness": 0.0127,
            "conductivity": 0.15989299909405608,
            "density": 784.9017946651944,
            "specific_heat": 829.4648292921422,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.4,
            "visible_absorptance": 0.4,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_100MmNormalweightConcreteFloor: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "100mm Normalweight concrete floor",
            "roughness": "MediumRough",
            "thickness": 0.1016,
            "conductivity": 2.3089016425031033,
            "density": 2322.006775596,
            "specific_heat": 832.0199665271966,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_12InNormalweightConcreteFloor: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "12 in. Normalweight Concrete Floor",
            "roughness": "MediumRough",
            "thickness": 0.30479999999999996,
            "conductivity": 2.308455174420426,
            "density": 2322.005309227381,
            "specific_heat": 831.4635397241673,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_1InStucco: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "1IN Stucco",
            "roughness": "Smooth",
            "thickness": 0.025299999999999993,
            "conductivity": 0.6913373548329222,
            "density": 1858.004248296516,
            "specific_heat": 836.4603158042426,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.92,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_25MmStucco: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "25mm Stucco",
            "roughness": "Smooth",
            "thickness": 0.0254,
            "conductivity": 0.7195184959232496,
            "density": 1856.0042437235265,
            "specific_heat": 839.4583814522845,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_4InNormalweightConcreteFloor: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "4 in. Normalweight Concrete Floor",
            "roughness": "MediumRough",
            "thickness": 0.1016,
            "conductivity": 2.308455174420426,
            "density": 2322.005309227381,
            "specific_heat": 831.4635397241673,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_4InNormalweightConcreteWall: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "4 in. Normalweight Concrete Wall",
            "roughness": "MediumRough",
            "thickness": 0.1016,
            "conductivity": 2.308455174420426,
            "density": 2322.005309227381,
            "specific_heat": 831.4635397241673,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_58InGypsumBoard: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "5/8 in. Gypsum Board",
            "roughness": "MediumSmooth",
            "thickness": 0.0159,
            "conductivity": 0.15989299909405463,
            "density": 800.0018291911765,
            "specific_heat": 1089.2971854559414,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_58InPlywood: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "5/8 in. Plywood",
            "roughness": "Smooth",
            "thickness": 0.0159,
            "conductivity": 0.11991974932054163,
            "density": 544.0012438499998,
            "specific_heat": 1209.2198113776988,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_6InHeavyweightConcreteRoof: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "6 in. Heavyweight Concrete Roof",
            "roughness": "MediumRough",
            "thickness": 0.15239999999999998,
            "conductivity": 2.308455174420426,
            "density": 2322.005309227381,
            "specific_heat": 831.4635397241673,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_6InNormalweightConcreteFloor: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "6 in. Normalweight Concrete Floor",
            "roughness": "MediumRough",
            "thickness": 0.15239999999999998,
            "conductivity": 2.308455174420426,
            "density": 2322.005309227381,
            "specific_heat": 831.4635397241673,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_8InConcreteBlockBasementWall: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "8 in. Concrete Block Basement Wall",
            "roughness": "MediumRough",
            "thickness": 0.2032,
            "conductivity": 1.325113229991986,
            "density": 1842.0042117126811,
            "specific_heat": 911.4119570053389,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_8InConcreteBlockWall: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "8 in. Concrete Block Wall",
            "roughness": "MediumRough",
            "thickness": 0.2032,
            "conductivity": 0.7195184959232496,
            "density": 800.0018291911765,
            "specific_heat": 831.4635397241673,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_8InNormalweightConcreteFloor: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "8 in. Normalweight Concrete Floor",
            "roughness": "MediumRough",
            "thickness": 0.2032,
            "conductivity": 2.308455174420426,
            "density": 2322.005309227381,
            "specific_heat": 831.4635397241673,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_8InNormalweightConcreteWall: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "8 in. Normalweight Concrete Wall",
            "roughness": "MediumRough",
            "thickness": 0.2032,
            "conductivity": 2.308455174420426,
            "density": 2322.005309227381,
            "specific_heat": 831.4635397241673,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_8InConcreteHwRefbldg: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "8IN CONCRETE HW RefBldg",
            "roughness": "Rough",
            "thickness": 0.2032,
            "conductivity": 1.3101232613269185,
            "density": 2240.0051217352993,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_8InConcreteHw: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "8IN Concrete HW",
            "roughness": "MediumRough",
            "thickness": 0.2033000000000001,
            "conductivity": 1.7284433202067357,
            "density": 2243.0051285947593,
            "specific_heat": 836.4603158042426,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.65,
            "visible_absorptance": 0.65,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_AcousticCeiling: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Acoustic Ceiling",
            "roughness": "MediumSmooth",
            "thickness": 0.0127,
            "conductivity": 0.056961880927257194,
            "density": 288.0006585088233,
            "specific_heat": 1338.1366342435858,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.2,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_AsphaltShingles: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Asphalt Shingles",
            "roughness": "VeryRough",
            "thickness": 0.003199999999999976,
            "conductivity": 0.03997324977351388,
            "density": 1120.002560867648,
            "specific_heat": 1259.1875721784268,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_AtticfloorInsulation: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "AtticFloor Insulation",
            "roughness": "MediumRough",
            "thickness": 0.23789999999999986,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_BuiltUpRoofing: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Built-up Roofing",
            "roughness": "Rough",
            "thickness": 0.009499999999999998,
            "conductivity": 0.15989299909405463,
            "density": 1120.002560867648,
            "specific_heat": 1459.0586153813556,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_BuiltUpRoofingHighlyReflective: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Built-up Roofing - Highly Reflective",
            "roughness": "Rough",
            "thickness": 0.009499999999999998,
            "conductivity": 0.15989299909405463,
            "density": 1120.002560867648,
            "specific_heat": 1459.0586153813556,
            "thermal_absorptance": 0.75,
            "solar_absorptance": 0.45,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_BulkStorageProductsMaterial: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Bulk Storage Products Material",
            "roughness": "Rough",
            "thickness": 2.4384049429641195,
            "conductivity": 1.4410356988618083,
            "density": 200.23111933632342,
            "specific_heat": 836.2604510890168,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_Carpet34InCbes: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Carpet - 3/4 in. CBES",
            "roughness": "Smooth",
            "thickness": 0.019049999999999997,
            "conductivity": 0.05000063634028838,
            "density": 288.33299999999997,
            "specific_heat": 1380.753138075314,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_F08MetalSurface: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "F08 Metal surface",
            "roughness": "Smooth",
            "thickness": 0.0008000000000000003,
            "conductivity": 45.249718743617805,
            "density": 7824.017889489713,
            "specific_heat": 499.6776080073138,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_F16AcousticTile: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "F16 Acoustic tile",
            "roughness": "MediumSmooth",
            "thickness": 0.019100000000000002,
            "conductivity": 0.05995987466027089,
            "density": 368.0008414279416,
            "specific_heat": 589.6195774486317,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.3,
            "visible_absorptance": 0.3,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_G0113MmGypsumBoard: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "G01 13mm gypsum board",
            "roughness": "Smooth",
            "thickness": 0.0127,
            "conductivity": 0.15992392281605958,
            "density": 800.0023344,
            "specific_heat": 1090.0261589958159,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.5,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_G01A19MmGypsumBoard: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "G01a 19mm gypsum board",
            "roughness": "MediumSmooth",
            "thickness": 0.018999999999999996,
            "conductivity": 0.15989299909405608,
            "density": 800.0018291911781,
            "specific_heat": 1089.2971854559455,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.4,
            "visible_absorptance": 0.4,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_G0525MmWood: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "G05 25mm wood",
            "roughness": "MediumSmooth",
            "thickness": 0.0254,
            "conductivity": 0.14989968665067757,
            "density": 608.0013901852949,
            "specific_heat": 1628.949002103841,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.5,
            "visible_absorptance": 0.5,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_Ground_Floor_R11_T2013: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Ground_Floor_R11_T2013",
            "roughness": "Smooth",
            "thickness": 0.044227242,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_Ground_Floor_R17_T2013: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Ground_Floor_R17_T2013",
            "roughness": "Smooth",
            "thickness": 0.06891629599999999,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_Ground_Floor_R22_T2013: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Ground_Floor_R22_T2013",
            "roughness": "Smooth",
            "thickness": 0.08918575,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_GypsumBoard12InCbes: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Gypsum Board - 1/2 in. CBES",
            "roughness": "Smooth",
            "thickness": 0.0127,
            "conductivity": 0.16000030671169507,
            "density": 640.74,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_GypsumOrPlasterBoard38In: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Gypsum Or Plaster Board - 3/8 in.",
            "roughness": "MediumSmooth",
            "thickness": 0.009499999999999998,
            "conductivity": 0.5796121217159504,
            "density": 800.0018291911765,
            "specific_heat": 1089.2971854559414,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_HwConcrete: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "HW CONCRETE",
            "roughness": "Rough",
            "thickness": 0.1016,
            "conductivity": 1.310376644700027,
            "density": 2240.0065363199997,
            "specific_heat": 836.8200836820084,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_HwConcrete8In: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "HW CONCRETE 8 in",
            "roughness": "Rough",
            "thickness": 0.2032,
            "conductivity": 1.310376644700027,
            "density": 2240.0065363199997,
            "specific_heat": 836.8200836820084,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_I0125MmInsulationBoard: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "I01 25mm insulation board",
            "roughness": "MediumRough",
            "thickness": 0.0254,
            "conductivity": 0.029979937330135372,
            "density": 43.000098319025845,
            "specific_heat": 1209.2198113776988,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.6,
            "visible_absorptance": 0.6,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_I0250MmInsulationBoard: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "I02 50mm insulation board",
            "roughness": "MediumRough",
            "thickness": 0.0508,
            "conductivity": 0.029979937330135372,
            "density": 43.000098319025845,
            "specific_heat": 1209.2198113776988,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.6,
            "visible_absorptance": 0.6,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_IeadNonresRoofInsulation176: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "IEAD NonRes Roof Insulation-1.76",
            "roughness": "MediumRough",
            "thickness": 0.07398050297987159,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_IeadRoofInsulationR347Ip: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "IEAD Roof Insulation R-3.47 IP",
            "roughness": "MediumRough",
            "thickness": 0.029942459748632622,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_Insulation1M: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Insulation 1m",
            "roughness": "Smooth",
            "thickness": 0.9999979999999999,
            "conductivity": 0.01999967801037276,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_M01100MmBrick: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "M01 100mm brick",
            "roughness": "MediumRough",
            "thickness": 0.1016,
            "conductivity": 0.8894048074606842,
            "density": 1920.0043900588325,
            "specific_heat": 789.4906206515565,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_M11100MmLightweightConcrete: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "M11 100mm lightweight concrete",
            "roughness": "MediumRough",
            "thickness": 0.1016,
            "conductivity": 0.5296455594990593,
            "density": 1280.0029267058849,
            "specific_heat": 839.4583814522887,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.5,
            "visible_absorptance": 0.5,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_M15200MmHeavyweightConcrete: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "M15 200mm heavyweight concrete",
            "roughness": "MediumRough",
            "thickness": 0.2032,
            "conductivity": 1.9486959264588086,
            "density": 2240.0051217352993,
            "specific_heat": 899.4196944131631,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.5,
            "visible_absorptance": 0.5,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MatCc054HwConcrete: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "MAT-CC05 4 HW CONCRETE",
            "roughness": "Rough",
            "thickness": 0.1016,
            "conductivity": 1.3101232613269185,
            "density": 2240.0051217352993,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.85,
            "visible_absorptance": 0.85,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MatCc058HwConcrete: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "MAT-CC05 8 HW CONCRETE",
            "roughness": "Rough",
            "thickness": 0.2032,
            "conductivity": 1.3101232613269185,
            "density": 2240.0051217352993,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.85,
            "visible_absorptance": 0.85,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MassNonresWallInsulation043: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Mass NonRes Wall Insulation-0.43",
            "roughness": "MediumRough",
            "thickness": 0.0004367814266280437,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MassWallInsulationR423Ip: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Mass Wall Insulation R-4.23 IP",
            "roughness": "MediumRough",
            "thickness": 0.0365371685816,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MetalBuildingSemiCondWallInsulation054: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Metal Building Semi-Cond Wall Insulation-0.54",
            "roughness": "MediumRough",
            "thickness": 0.015345987404915393,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MetalBuildingWallInsulationR414Ip: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Metal Building Wall Insulation R-4.14 IP",
            "roughness": "MediumRough",
            "thickness": 0.0356729086642,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MetalDecking: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Metal Decking",
            "roughness": "MediumSmooth",
            "thickness": 0.0014999999999999994,
            "conductivity": 44.97590198266915,
            "density": 7680.017560235314,
            "specific_heat": 418.1302223805201,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.6,
            "visible_absorptance": 0.6,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MetalRoofInsulationR521Ip: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Metal Roof Insulation R-5.21 IP",
            "roughness": "MediumRough",
            "thickness": 0.044938850457399995,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MetalRoofSurface: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Metal Roof Surface",
            "roughness": "Smooth",
            "thickness": 0.0007999999999999979,
            "conductivity": 45.24971874361766,
            "density": 7824.017889489713,
            "specific_heat": 499.67760800730963,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MetalRoofSurfaceHighlyReflective: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Metal Roof Surface - Highly Reflective",
            "roughness": "Smooth",
            "thickness": 0.0007999999999999979,
            "conductivity": 45.24971874361766,
            "density": 7824.017889489713,
            "specific_heat": 499.67760800730963,
            "thermal_absorptance": 0.75,
            "solar_absorptance": 0.45,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MetalRoofing: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Metal Roofing",
            "roughness": "MediumSmooth",
            "thickness": 0.0014999999999999994,
            "conductivity": 44.97590198266915,
            "density": 7680.017560235314,
            "specific_heat": 418.1302223805201,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.6,
            "visible_absorptance": 0.6,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MetalRoofingHighlyReflective: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Metal Roofing - Highly Reflective",
            "roughness": "MediumSmooth",
            "thickness": 0.0014999999999999994,
            "conductivity": 44.97590198266915,
            "density": 7680.017560235314,
            "specific_heat": 418.1302223805201,
            "thermal_absorptance": 0.75,
            "solar_absorptance": 0.45,
            "visible_absorptance": 0.6,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MetalSemiCondRoofInsulation105: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Metal Semi-Cond Roof Insulation-1.05",
            "roughness": "MediumRough",
            "thickness": 0.04226733019063131,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MetalSiding: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Metal Siding",
            "roughness": "Smooth",
            "thickness": 0.0014999999999999994,
            "conductivity": 44.92993274542956,
            "density": 7688.877580493597,
            "specific_heat": 409.7356385659975,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MetalStandingSeam116InCbes: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Metal Standing Seam - 1/16 in. CBES",
            "roughness": "MediumRough",
            "thickness": 0.001524,
            "conductivity": 0.579999310186949,
            "density": 7820.552070000001,
            "specific_heat": 502.09205020920496,
            "thermal_absorptance": 0.85,
            "solar_absorptance": 0.37,
            "visible_absorptance": 0.85,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MetalSurface: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Metal Surface",
            "roughness": "Smooth",
            "thickness": 0.0007999999999999979,
            "conductivity": 45.24971874361766,
            "density": 7824.017889489713,
            "specific_heat": 499.67760800730963,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_Nacm_Carpet34In: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "NACM_Carpet 3/4in",
            "roughness": "Smooth",
            "thickness": 0.019049999999999997,
            "conductivity": 0.05000063634028838,
            "density": 288.33299999999997,
            "specific_heat": 1380.753138075314,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.9,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_Nacm_Concrete4In_140LbFt3: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "NACM_Concrete 4in_140lb/ft3",
            "roughness": "MediumRough",
            "thickness": 0.1016,
            "conductivity": 1.9500003149271867,
            "density": 2239.0659299999998,
            "specific_heat": 920.5020920502092,
            "thermal_absorptance": 0.75,
            "solar_absorptance": 0.92,
            "visible_absorptance": 0.75,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_Nacm_GypsumBoard58In: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "NACM_Gypsum Board 5/8in",
            "roughness": "MediumRough",
            "thickness": 0.016002,
            "conductivity": 0.16000030671169507,
            "density": 640.74,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.5,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_Plywood58InCbes: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Plywood - 5/8 in. CBES",
            "roughness": "Smooth",
            "thickness": 0.016002,
            "conductivity": 0.11999950937659305,
            "density": 480.555,
            "specific_heat": 1882.8451882845188,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_R1_R1420: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "R1_R14.20",
            "roughness": "Smooth",
            "thickness": 0.05756529,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_R2_R1290: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "R2_R12.90",
            "roughness": "Smooth",
            "thickness": 0.052295298000000004,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_R3_R1774: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "R3_R17.74",
            "roughness": "Smooth",
            "thickness": 0.071916036,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_R4_R2028: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "R4_R20.28",
            "roughness": "Smooth",
            "thickness": 0.082211418,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_R_R30: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "R_R30",
            "roughness": "Smooth",
            "thickness": 0.121616978,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_R_R38: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "R_R38",
            "roughness": "Smooth",
            "thickness": 0.154048206,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_R_T24_2013_2486: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "R_T24_2013_24.86",
            "roughness": "Smooth",
            "thickness": 0.100779834,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_RoofInsulation18: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Roof Insulation [18]",
            "roughness": "MediumRough",
            "thickness": 0.16929999999999995,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_RoofMembrane: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Roof Membrane",
            "roughness": "VeryRough",
            "thickness": 0.009499999999999998,
            "conductivity": 0.15989299909405608,
            "density": 1121.29256381722,
            "specific_heat": 1459.0586153813556,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_RoofMembraneHighlyReflective: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Roof Membrane - Highly Reflective",
            "roughness": "VeryRough",
            "thickness": 0.009499999999999998,
            "conductivity": 0.15989299909405608,
            "density": 1121.29256381722,
            "specific_heat": 1459.0586153813556,
            "thermal_absorptance": 0.75,
            "solar_absorptance": 0.45,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_StdWood6In: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Std Wood 6 in.",
            "roughness": "MediumSmooth",
            "thickness": 0.14999999999999997,
            "conductivity": 0.11991974932054163,
            "density": 540.0012347040437,
            "specific_heat": 1209.2198113776988,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_StdWood6InchFurnishings: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Std Wood 6inch Furnishings",
            "roughness": "MediumSmooth",
            "thickness": 0.150114,
            "conductivity": 0.11999950937659305,
            "density": 539.983635,
            "specific_heat": 1213.389121338912,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_SteelFrameNonresWallInsulation073: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Steel Frame NonRes Wall Insulation-0.73",
            "roughness": "MediumRough",
            "thickness": 0.020278246975300705,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_SteelFrameWallInsulationR102Ip: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Steel Frame Wall Insulation R-1.02 IP",
            "roughness": "MediumRough",
            "thickness": 0.008836428479,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_SteelFrameCavity: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Steel Frame/Cavity",
            "roughness": "MediumRough",
            "thickness": 0.08890018021223342,
            "conductivity": 0.08405993542170573,
            "density": 641.3803214581126,
            "specific_heat": 501.75627065341007,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_Stucco78InCbes: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Stucco - 7/8 in. CBES",
            "roughness": "MediumRough",
            "thickness": 0.022352,
            "conductivity": 0.7000002608778985,
            "density": 1855.102485,
            "specific_heat": 836.8200836820084,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_W1_R860: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "W1_R8.60",
            "roughness": "Smooth",
            "thickness": 0.034862008,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_W2_R1113: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "W2_R11.13",
            "roughness": "Smooth",
            "thickness": 0.045119036,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_W3_R1136: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "W3_R11.36",
            "roughness": "Smooth",
            "thickness": 0.04605147,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_W4_R1262: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "W4_R12.62",
            "roughness": "Smooth",
            "thickness": 0.05115915599999999,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_W_T24_2013_R1399: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "W_T24_2013_R13.99",
            "roughness": "Smooth",
            "thickness": 0.056714136,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_W_M1_R15: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "W_m1_R15",
            "roughness": "Smooth",
            "thickness": 0.060808362,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_W_M2_R19: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "W_m2_R19",
            "roughness": "Smooth",
            "thickness": 0.077023976,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_W_M3_R21: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "W_m3_R21",
            "roughness": "Smooth",
            "thickness": 0.08513190999999999,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_WallInsulation31: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Wall Insulation [31]",
            "roughness": "MediumRough",
            "thickness": 0.03370000000000008,
            "conductivity": 0.043171109755394975,
            "density": 91.0002080704965,
            "specific_heat": 836.4603158042426,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.5,
            "visible_absorptance": 0.5,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_WoodFrameNonresWallInsulation073: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Wood Frame NonRes Wall Insulation-0.73",
            "roughness": "MediumRough",
            "thickness": 0.020278246975300705,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_WoodFrameWallInsulationR161Ip: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Wood Frame Wall Insulation R-1.61 IP",
            "roughness": "MediumRough",
            "thickness": 0.013873826709999999,
            "conductivity": 0.04896723097255453,
            "density": 265.0006059195773,
            "specific_heat": 836.2604447610418,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_WoodSiding: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Wood Siding",
            "roughness": "MediumSmooth",
            "thickness": 0.010000000000000004,
            "conductivity": 0.10992643687716327,
            "density": 544.6212452676245,
            "specific_heat": 1209.2198113776988,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.78,
            "visible_absorptance": 0.78,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_Ceiling_2_Insulation: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "ceiling_2_Insulation",
            "roughness": "Smooth",
            "thickness": 0.100778564,
            "conductivity": 0.02300770107232491,
            "density": 16.0185,
            "specific_heat": 1129.7071129707113,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_DrySand: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Dry Sand",
            "roughness": "Rough",
            "thickness": 0.2,
            "conductivity": 0.33,
            "density": 1555.0,
            "specific_heat": 800.0,
            "thermal_absorptance": 0.85,
            "solar_absorptance": 0.65,
            "visible_absorptance": 0.65,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_DryDust: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Dry Dust",
            "roughness": "Rough",
            "thickness": 0.2,
            "conductivity": 0.5,
            "density": 1600.0,
            "specific_heat": 1026.0,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.7,
            "visible_absorptance": 0.7,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_MoistSoil: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Moist Soil",
            "roughness": "Rough",
            "thickness": 0.2,
            "conductivity": 1.0,
            "density": 1250.0,
            "specific_heat": 1252.0,
            "thermal_absorptance": 0.92,
            "solar_absorptance": 0.75,
            "visible_absorptance": 0.75,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_Mud: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Mud",
            "roughness": "MediumRough",
            "thickness": 0.2,
            "conductivity": 1.4,
            "density": 1840.0,
            "specific_heat": 1480.0,
            "thermal_absorptance": 0.95,
            "solar_absorptance": 0.8,
            "visible_absorptance": 0.8,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_ConcretePavement: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Concrete Pavement",
            "roughness": "MediumRough",
            "thickness": 0.2,
            "conductivity": 1.73,
            "density": 2243.0,
            "specific_heat": 837.0,
            "thermal_absorptance": 0.9,
            "solar_absorptance": 0.65,
            "visible_absorptance": 0.65,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_AsphaltPavement: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Asphalt Pavement",
            "roughness": "MediumRough",
            "thickness": 0.2,
            "conductivity": 0.75,
            "density": 2360.0,
            "specific_heat": 920.0,
            "thermal_absorptance": 0.93,
            "solar_absorptance": 0.87,
            "visible_absorptance": 0.87,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_SolidRock: OpaqueMaterial = create_material(
        {
            "type": "EnergyMaterial",
            "identifier": "Solid Rock",
            "roughness": "MediumRough",
            "thickness": 0.2,
            "conductivity": 3.0,
            "density": 2700.0,
            "specific_heat": 790.0,
            "thermal_absorptance": 0.96,
            "solar_absorptance": 0.55,
            "visible_absorptance": 0.55,
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )
    LBT_GrassyLawn: OpaqueVegetationMaterial = create_material(
        {
            "type": "EnergyMaterialVegetation",
            "identifier": "Grassy Lawn",
            "plant_height": 0.2,
            "leaf_area_index": 1.0,
            "leaf_reflectivity": 0.22,
            "leaf_emissivity": 0.95,
            "min_stomatal_resist": 180.0,
            "roughness": "MediumRough",
            "thickness": 0.1,
            "conductivity": 0.35,
            "density": 1100.0,
            "specific_heat": 1200.0,
            "soil_thermal_absorptance": 0.9,
            "soil_solar_absorptance": 0.7,
            "soil_visible_absorptance": 0.7,
            "sat_vol_moist_cont": 0.3,
            "residual_vol_moist_cont": 0.01,
            "init_vol_moist_cont": 0.1,
            "moist_diff_model": "Simple",
            "source": f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
        }
    )


# pylint: enable=C0103
