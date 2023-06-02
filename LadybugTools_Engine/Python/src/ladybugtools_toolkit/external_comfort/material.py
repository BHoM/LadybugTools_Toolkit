from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from types import FunctionType
from typing import Any, Dict, Union

from honeybee_energy.material.opaque import EnergyMaterial, EnergyMaterialVegetation

from ..bhomutil.bhom_object import BHoMObject

# NOTE: This is the version of the Honeybee Material Standards that this module referneces. If these standards are updated, this version number should be updated as well.
HONEYBEE_MATERIAL_STANDARDS_VERSION = "2.2.6"

# TODO - inheritance here is a nightmare when trying to account for automatic BHoM logging too.


@dataclass(init=True, repr=True, eq=True)
class OpaqueMaterial(BHoMObject):
    """An object representing a material."""

    Identifier: str = field(repr=True, compare=True)
    Source: str = field(repr=False, compare=True)
    Thickness: float = field(repr=False, compare=True)
    Conductivity: float = field(repr=False, compare=True)
    Density: float = field(repr=False, compare=True)
    SpecificHeat: float = field(repr=False, compare=True)
    Roughness: str = field(repr=False, compare=True, default="MediumRough")
    ThermalAbsorptance: float = field(repr=False, compare=True, default=0.9)
    SolarAbsorptance: float = field(repr=False, compare=True, default=0.7)
    VisibleAbsorptance: float = field(repr=False, compare=True, default=0.7)

    _t: str = field(
        init=False,
        repr=False,
        compare=True,
        default="BH.oM.LadybugTools.OpaqueMaterial",
    )

    def __post_init__(self):
        if "," in self.Identifier:
            raise ValueError("commas cannot be used in Identifier values.")
        if len(self.Identifier) > 100:
            raise ValueError("Identifiers must be lass than 100 characters")

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> OpaqueMaterial:
        """Create this object from a dictionary."""

        return cls(
            Identifier=dictionary["Identifier"],
            Thickness=dictionary["Thickness"],
            Conductivity=dictionary["Conductivity"],
            Density=dictionary["Density"],
            SpecificHeat=dictionary["SpecificHeat"],
            Roughness=dictionary["Roughness"],
            ThermalAbsorptance=dictionary["ThermalAbsorptance"],
            SolarAbsorptance=dictionary["SolarAbsorptance"],
            VisibleAbsorptance=dictionary["VisibleAbsorptance"],
            Source=dictionary["Source"],
        )

    @classmethod
    def from_json(cls, json_string: str) -> OpaqueMaterial:
        """Create this object from a JSON string."""

        dictionary = json.loads(json_string)

        return cls.from_dict(dictionary)

    @classmethod
    def from_lbt(cls, lbt_material: EnergyMaterial) -> OpaqueMaterial:
        """Create this object from its LBT equivalent."""
        return cls(
            Identifier=lbt_material.identifier,
            Thickness=lbt_material.thickness,
            Conductivity=lbt_material.conductivity,
            Density=lbt_material.density,
            SpecificHeat=lbt_material.specific_heat,
            Roughness=lbt_material.roughness,
            ThermalAbsorptance=lbt_material.thermal_absorptance,
            SolarAbsorptance=lbt_material.solar_absorptance,
            VisibleAbsorptance=lbt_material.visible_absorptance,
            Source="LadybugTools",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as it's dictionary equivalent."""
        dictionary = {}
        for k, v in self.__dict__.items():
            if isinstance(getattr(self, k), FunctionType):
                continue
            dictionary[k] = v
        dictionary["_t"] = self._t
        return dictionary

    def to_json(self) -> str:
        """Return this object as it's JSON string equivalent."""
        return json.dumps(self.to_dict())

    def to_lbt(self) -> EnergyMaterial:
        """Return this object as its LBT equivalent."""
        return EnergyMaterial(
            identifier=self.Identifier,
            thickness=self.Thickness,
            conductivity=self.Conductivity,
            density=self.Density,
            specific_heat=self.SpecificHeat,
            roughness=self.Roughness,
            thermal_absorptance=self.ThermalAbsorptance,
            solar_absorptance=self.SolarAbsorptance,
            visible_absorptance=self.VisibleAbsorptance,
        )

    @property
    def identifier(self) -> str:
        """Handy accessor using proper Python naming style."""
        return self.Identifier

    @property
    def thickness(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.Thickness

    @property
    def conductivity(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.Conductivity

    @property
    def density(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.Density

    @property
    def specific_heat(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.SpecificHeat

    @property
    def roughness(self) -> str:
        """Handy accessor using proper Python naming style."""
        return self.Roughness

    @property
    def thermal_absorptance(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.ThermalAbsorptance

    @property
    def solar_absorptance(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.SolarAbsorptance

    @property
    def visible_absorptance(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.VisibleAbsorptance

    @property
    def source(self) -> str:
        """Handy accessor using proper Python naming style."""
        return self.Source


@dataclass(init=True, repr=True, eq=True)
class OpaqueVegetationMaterial(BHoMObject):
    """An object representing a vegetation material."""

    Identifier: float = field(repr=True, compare=True)
    Source: str = field(repr=False, compare=True)
    Thickness: float = field(repr=False, compare=True, default=0.1)
    Conductivity: float = field(repr=False, compare=True, default=0.35)
    Density: float = field(repr=False, compare=True, default=1100)
    SpecificHeat: float = field(repr=False, compare=True, default=1200)
    Roughness: float = field(repr=False, compare=True, default="MediumRough")
    SoilThermalAbsorptance: float = field(repr=False, compare=True, default=0.9)
    SoilSolarAbsorptance: float = field(repr=False, compare=True, default=0.7)
    SoilVisibleAbsorptance: float = field(repr=False, compare=True, default=0.7)
    PlantHeight: float = field(repr=False, compare=True, default=0.2)
    LeafAreaIndex: float = field(repr=False, compare=True, default=1)
    LeafReflectivity: float = field(repr=False, compare=True, default=0.22)
    LeafEmissivity: float = field(repr=False, compare=True, default=0.95)
    MinStomatalResist: float = field(repr=False, compare=True, default=180)

    _t: str = field(
        init=False,
        repr=False,
        compare=True,
        default="BH.oM.LadybugTools.OpaqueVegetationMaterial",
    )

    def __post_init__(self):
        if "," in self.Identifier:
            raise ValueError("commas cannot be used in Identifier values.")
        if len(self.Identifier) > 100:
            raise ValueError("Identifiers must be lass than 100 characters")

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> OpaqueVegetationMaterial:
        """Create this object from a dictionary."""

        return cls(
            Identifier=dictionary["Identifier"],
            Thickness=dictionary["Thickness"],
            Conductivity=dictionary["Conductivity"],
            Density=dictionary["Density"],
            SpecificHeat=dictionary["SpecificHeat"],
            Roughness=dictionary["Roughness"],
            SoilThermalAbsorptance=dictionary["SoilThermalAbsorptance"],
            SoilSolarAbsorptance=dictionary["SoilSolarAbsorptance"],
            SoilVisibleAbsorptance=dictionary["SoilVisibleAbsorptance"],
            PlantHeight=dictionary["PlantHeight"],
            LeafAreaIndex=dictionary["LeafAreaIndex"],
            LeafReflectivity=dictionary["LeafReflectivity"],
            LeafEmissivity=dictionary["LeafEmissivity"],
            MinStomatalResist=dictionary["MinStomatalResist"],
            Source=dictionary["Source"],
        )

    @classmethod
    def from_json(cls, json_string: str) -> OpaqueVegetationMaterial:
        """Create this object from a JSON string."""

        dictionary = json.loads(json_string)

        return cls.from_dict(dictionary)

    @classmethod
    def from_lbt(cls, material: EnergyMaterialVegetation):
        """Create this object from its LBT equivalent."""
        return cls(
            Identifier=material.identifier,
            Thickness=material.thickness,
            Conductivity=material.conductivity,
            Density=material.density,
            SpecificHeat=material.specific_heat,
            Roughness=material.roughness,
            SoilThermalAbsorptance=material.soil_thermal_absorptance,
            SoilSolarAbsorptance=material.soil_solar_absorptance,
            SoilVisibleAbsorptance=material.soil_visible_absorptance,
            PlantHeight=material.plant_height,
            LeafAreaIndex=material.leaf_area_index,
            LeafReflectivity=material.leaf_reflectivity,
            LeafEmissivity=material.leaf_emissivity,
            MinStomatalResist=material.min_stomatal_resist,
            Source="LadybugTools",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as it's dictionary equivalent."""
        dictionary = {}
        for k, v in self.__dict__.items():
            if isinstance(getattr(self, k), FunctionType):
                continue
            dictionary[k] = v
        dictionary["_t"] = self._t
        return dictionary

    def to_json(self) -> str:
        """Return this object as it's JSON string equivalent."""
        return json.dumps(self.to_dict())

    def to_lbt(self) -> EnergyMaterialVegetation:
        """Return this object as its LBT equivalent."""
        return EnergyMaterialVegetation(
            identifier=self.Identifier,
            thickness=self.Thickness,
            conductivity=self.Conductivity,
            density=self.Density,
            specific_heat=self.SpecificHeat,
            roughness=self.Roughness,
            soil_thermal_absorptance=self.SoilThermalAbsorptance,
            soil_solar_absorptance=self.SoilSolarAbsorptance,
            soil_visible_absorptance=self.SoilVisibleAbsorptance,
            plant_height=self.PlantHeight,
            leaf_area_index=self.LeafAreaIndex,
            leaf_reflectivity=self.LeafReflectivity,
            leaf_emissivity=self.LeafEmissivity,
            min_stomatal_resist=self.MinStomatalResist,
        )

    @property
    def identifier(self) -> str:
        """Handy accessor using proper Python naming style."""
        return self.Identifier

    @property
    def thickness(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.Thickness

    @property
    def conductivity(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.Conductivity

    @property
    def density(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.Density

    @property
    def specific_heat(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.SpecificHeat

    @property
    def roughness(self) -> str:
        """Handy accessor using proper Python naming style."""
        return self.Roughness

    @property
    def soil_thermal_absorptance(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.SoilThermalAbsorptance

    @property
    def soil_solar_absorptance(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.SoilSolarAbsorptance

    @property
    def soil_visible_absorptance(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.SoilVisibleAbsorptance

    @property
    def plant_height(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.PlantHeight

    @property
    def leaf_area_index(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.LeafAreaIndex

    @property
    def leaf_reflectivity(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.LeafReflectivity

    @property
    def leaf_emissivity(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.LeafEmissivity

    @property
    def min_stomatal_resist(self) -> float:
        """Handy accessor using proper Python naming style."""
        return self.MinStomatalResist

    @property
    def source(self) -> str:
        """Handy accessor using proper Python naming style."""
        return self.Source


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
        Identifier="Fabric",
        Roughness="Smooth",
        Thickness=0.002,
        Conductivity=0.06,
        Density=500.0,
        SpecificHeat=1800.0,
        ThermalAbsorptance=0.89,
        SolarAbsorptance=0.5,
        VisibleAbsorptance=0.5,
        Source="Buro Happold",
    )
    SHRUBS: OpaqueVegetationMaterial = OpaqueVegetationMaterial(
        Identifier="Shrubs",
        Roughness="Rough",
        Thickness=0.1,
        Conductivity=0.35,
        Density=1260,
        SpecificHeat=1100,
        SoilThermalAbsorptance=0.9,
        SoilSolarAbsorptance=0.7,
        SoilVisibleAbsorptance=0.7,
        PlantHeight=0.2,
        LeafAreaIndex=2.08,
        LeafReflectivity=0.21,
        LeafEmissivity=0.95,
        MinStomatalResist=180,
        Source="Buro Happold",
    )
    TRAVERTINE: OpaqueMaterial = OpaqueMaterial(
        Identifier="Travertine",
        Roughness="MediumRough",
        Thickness=0.2,
        Conductivity=3.2,
        Density=2700.0,
        SpecificHeat=790.0,
        ThermalAbsorptance=0.96,
        SolarAbsorptance=0.55,
        VisibleAbsorptance=0.55,
        Source="Buro Happold",
    )

    # MATERIALS FROM ICE TOOL
    SD01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Sand (Quartzite_Beige/brown/black_0.26 albedo)",
        Thickness=0.01,
        Conductivity=5.0,
        Density=2600,
        SpecificHeat=690,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Rough",
        ThermalAbsorptance=0.92,
        SolarAbsorptance=0.74,
        VisibleAbsorptance=0.74,
    )
    SD02: OpaqueMaterial = OpaqueMaterial(
        Identifier="Sand (Quartzite_Beige/brown/black_0.32 albedo)",
        Thickness=0.01,
        Conductivity=5.0,
        Density=2600,
        SpecificHeat=690,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Rough",
        ThermalAbsorptance=0.96,
        SolarAbsorptance=0.6799999999999999,
        VisibleAbsorptance=0.6799999999999999,
    )
    SD03: OpaqueMaterial = OpaqueMaterial(
        Identifier="Sand (Quartzite_Brown_0.25 albedo)",
        Thickness=0.01,
        Conductivity=5.0,
        Density=2600,
        SpecificHeat=690,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Rough",
        ThermalAbsorptance=0.97,
        SolarAbsorptance=0.75,
        VisibleAbsorptance=0.75,
    )
    ST01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Stone (Sandstone_Beige_0.4 albedo)",
        Thickness=0.01,
        Conductivity=3.0,
        Density=2000,
        SpecificHeat=745,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Smooth",
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.6,
        VisibleAbsorptance=0.6,
    )
    ST02: OpaqueMaterial = OpaqueMaterial(
        Identifier="Stone (Carboniferous Coral Limestone_Grey_0.2 albedo)",
        Thickness=0.01,
        Conductivity=2.0,
        Density=3250,
        SpecificHeat=1180,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Smooth",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.8,
        VisibleAbsorptance=0.8,
    )
    ST03: OpaqueMaterial = OpaqueMaterial(
        Identifier="Stone (Sandstone_Yellow_0.26 albedo)",
        Thickness=0.01,
        Conductivity=3.0,
        Density=2000,
        SpecificHeat=745,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Smooth",
        ThermalAbsorptance=0.93,
        SolarAbsorptance=0.74,
        VisibleAbsorptance=0.74,
    )
    ST04: OpaqueMaterial = OpaqueMaterial(
        Identifier="Stone (Limestone_Beige_0.68 albedo)",
        Thickness=0.01,
        Conductivity=2.0,
        Density=3250,
        SpecificHeat=1180,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Smooth",
        ThermalAbsorptance=0.93,
        SolarAbsorptance=0.31999999999999995,
        VisibleAbsorptance=0.31999999999999995,
    )
    ST05: OpaqueMaterial = OpaqueMaterial(
        Identifier="Stone (Sandstone_Light grey_0.46 albedo)",
        Thickness=0.01,
        Conductivity=3.0,
        Density=2000,
        SpecificHeat=745,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Smooth",
        ThermalAbsorptance=0.92,
        SolarAbsorptance=0.54,
        VisibleAbsorptance=0.54,
    )
    ST06: OpaqueMaterial = OpaqueMaterial(
        Identifier="Stone (Granite_White/black_0.48 albedo)",
        Thickness=0.01,
        Conductivity=2.7,
        Density=2600,
        SpecificHeat=280,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.92,
        SolarAbsorptance=0.52,
        VisibleAbsorptance=0.52,
    )
    ST07: OpaqueMaterial = OpaqueMaterial(
        Identifier="Stone (Granite With Cement_White/red_0.34 albedo)",
        Thickness=0.01,
        Conductivity=2.7,
        Density=2600,
        SpecificHeat=280,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.93,
        SolarAbsorptance=0.6599999999999999,
        VisibleAbsorptance=0.6599999999999999,
    )
    ST08: OpaqueMaterial = OpaqueMaterial(
        Identifier="Stone (Granite With Cement_White/black_0.41 albedo)",
        Thickness=0.01,
        Conductivity=2.7,
        Density=2600,
        SpecificHeat=280,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.89,
        SolarAbsorptance=0.5900000000000001,
        VisibleAbsorptance=0.5900000000000001,
    )
    ST09: OpaqueMaterial = OpaqueMaterial(
        Identifier="Stone (Granite_White/red/black_0.54 albedo)",
        Thickness=0.01,
        Conductivity=2.7,
        Density=2600,
        SpecificHeat=280,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.93,
        SolarAbsorptance=0.45999999999999996,
        VisibleAbsorptance=0.45999999999999996,
    )
    ST10: OpaqueMaterial = OpaqueMaterial(
        Identifier="Stone (Granite_Red/black_0.22 albedo)",
        Thickness=0.01,
        Conductivity=2.7,
        Density=2600,
        SpecificHeat=280,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Smooth",
        ThermalAbsorptance=0.89,
        SolarAbsorptance=0.78,
        VisibleAbsorptance=0.78,
    )
    AS01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Asphalt (Asphalt With Stone Aggregate_Black/grey_0.21 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2300,
        SpecificHeat=1400,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.96,
        SolarAbsorptance=0.79,
        VisibleAbsorptance=0.79,
    )
    AS02: OpaqueMaterial = OpaqueMaterial(
        Identifier="Asphalt (Asphalt With Stone Aggregate_Black/grey_0.18 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2300,
        SpecificHeat=1400,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.8200000000000001,
        VisibleAbsorptance=0.8200000000000001,
    )
    AS03: OpaqueMaterial = OpaqueMaterial(
        Identifier="Asphalt (Asphalt With Stone Aggregate_Black/grey_0.21 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2300,
        SpecificHeat=1400,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.79,
        VisibleAbsorptance=0.79,
    )
    AS04: OpaqueMaterial = OpaqueMaterial(
        Identifier="Asphalt (Asphalt With Stone Aggregate_Black/grey_0.18 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2300,
        SpecificHeat=1400,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.8200000000000001,
        VisibleAbsorptance=0.8200000000000001,
    )
    AS05: OpaqueMaterial = OpaqueMaterial(
        Identifier="Asphalt (Asphalt With Stone Aggregate_Black/grey_0.19 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2300,
        SpecificHeat=1400,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.93,
        SolarAbsorptance=0.81,
        VisibleAbsorptance=0.81,
    )
    AS06: OpaqueMaterial = OpaqueMaterial(
        Identifier="Asphalt (Asphalt With Stone Aggregate_Black/grey_0.12 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2300,
        SpecificHeat=1400,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.91,
        SolarAbsorptance=0.88,
        VisibleAbsorptance=0.88,
    )
    TR01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Tarmac (Tarmac Roofing Paper_Grey_0.07 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2300,
        SpecificHeat=1400,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.93,
        SolarAbsorptance=0.9299999999999999,
        VisibleAbsorptance=0.9299999999999999,
    )
    TR02: OpaqueMaterial = OpaqueMaterial(
        Identifier="Tarmac (Tarmac_Black_0.13 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2300,
        SpecificHeat=1400,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.87,
        VisibleAbsorptance=0.87,
    )
    TR03: OpaqueMaterial = OpaqueMaterial(
        Identifier="Tarmac (Tarmac_Black_0.08 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2300,
        SpecificHeat=1400,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.92,
        VisibleAbsorptance=0.92,
    )
    TR04: OpaqueMaterial = OpaqueMaterial(
        Identifier="Tarmac (Tarmac_Black_0.1 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2300,
        SpecificHeat=1400,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.96,
        SolarAbsorptance=0.9,
        VisibleAbsorptance=0.9,
    )
    CM01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement_Grey/ochre_0.29 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.71,
        VisibleAbsorptance=0.71,
    )
    CC01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Concrete (Light Concrete_Grey/white_0.21 albedo)",
        Thickness=0.01,
        Conductivity=0.21,
        Density=2800,
        SpecificHeat=657,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.92,
        SolarAbsorptance=0.79,
        VisibleAbsorptance=0.79,
    )
    CM02: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement_Grey_0.23 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.91,
        SolarAbsorptance=0.77,
        VisibleAbsorptance=0.77,
    )
    CC02: OpaqueMaterial = OpaqueMaterial(
        Identifier="Concrete (Concrete_Grey_0.37 albedo)",
        Thickness=0.01,
        Conductivity=0.21,
        Density=2800,
        SpecificHeat=657,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.63,
        VisibleAbsorptance=0.63,
    )
    CM03: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement_Grey_0.41 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.5900000000000001,
        VisibleAbsorptance=0.5900000000000001,
    )
    CC03: OpaqueMaterial = OpaqueMaterial(
        Identifier="Concrete (Concrete_White_0.42 albedo)",
        Thickness=0.01,
        Conductivity=0.21,
        Density=2800,
        SpecificHeat=657,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.5800000000000001,
        VisibleAbsorptance=0.5800000000000001,
    )
    CC04: OpaqueMaterial = OpaqueMaterial(
        Identifier="Concrete (Concrete_Grey_0.25 albedo)",
        Thickness=0.01,
        Conductivity=0.21,
        Density=2800,
        SpecificHeat=657,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Rough",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.75,
        VisibleAbsorptance=0.75,
    )
    CM04: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement Brick_Yellow_0.3 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
    )
    CM05: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement Brick_With Sand_Black/light grey_0.11 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.89,
        VisibleAbsorptance=0.89,
    )
    CM06: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement Brick_Black_0.09 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.91,
        VisibleAbsorptance=0.91,
    )
    CL01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Clay (Brick Clay_With Cement_Red_0.31 albedo)",
        Thickness=0.01,
        Conductivity=1.0,
        Density=1300,
        SpecificHeat=1381,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.91,
        SolarAbsorptance=0.69,
        VisibleAbsorptance=0.69,
    )
    CM07: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement Brick_Red_0.17 albedo)",
        Thickness=0.01,
        Conductivity=0.25,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.83,
        VisibleAbsorptance=0.83,
    )
    CM08: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement Brick_With Sand_Black_0.2 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.89,
        SolarAbsorptance=0.8,
        VisibleAbsorptance=0.8,
    )
    CM09: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement Brick_Light red_0.22 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.78,
        VisibleAbsorptance=0.78,
    )
    CL02: OpaqueMaterial = OpaqueMaterial(
        Identifier="Clay (Brick Clay_Light red_0.43 albedo)",
        Thickness=0.01,
        Conductivity=1.0,
        Density=1300,
        SpecificHeat=1381,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.5700000000000001,
        VisibleAbsorptance=0.5700000000000001,
    )
    CM10: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement_Red_0.27 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.73,
        VisibleAbsorptance=0.73,
    )
    CL03: OpaqueMaterial = OpaqueMaterial(
        Identifier="Clay (Brick Clay_Painted_Red with beige and grey_0.53 albedo)",
        Thickness=0.01,
        Conductivity=1.0,
        Density=1300,
        SpecificHeat=1381,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.93,
        SolarAbsorptance=0.47,
        VisibleAbsorptance=0.47,
    )
    CL04: OpaqueMaterial = OpaqueMaterial(
        Identifier="Clay (Brick Clay_With Cement_Red/grey_0.35 albedo)",
        Thickness=0.01,
        Conductivity=1.0,
        Density=1300,
        SpecificHeat=1381,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.65,
        VisibleAbsorptance=0.65,
    )
    CL05: OpaqueMaterial = OpaqueMaterial(
        Identifier="Clay (Brick Clay_Painted_Red with white_0.56 albedo)",
        Thickness=0.01,
        Conductivity=1.0,
        Density=1300,
        SpecificHeat=1381,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Rough",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.43999999999999995,
        VisibleAbsorptance=0.43999999999999995,
    )
    CL06: OpaqueMaterial = OpaqueMaterial(
        Identifier="Clay (Brick Clay_Red_0.32 albedo)",
        Thickness=0.01,
        Conductivity=1.0,
        Density=1300,
        SpecificHeat=1381,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.6799999999999999,
        VisibleAbsorptance=0.6799999999999999,
    )
    CL07: OpaqueMaterial = OpaqueMaterial(
        Identifier="Clay (Brick Clay_Yellow/grey_0.43 albedo)",
        Thickness=0.01,
        Conductivity=1.0,
        Density=2600,
        SpecificHeat=1381,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.96,
        SolarAbsorptance=0.5700000000000001,
        VisibleAbsorptance=0.5700000000000001,
    )
    SL01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Slate (Slate_Black_0.09 albedo)",
        Thickness=0.01,
        Conductivity=2.0,
        Density=2771,
        SpecificHeat=760,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.91,
        VisibleAbsorptance=0.91,
    )
    SL02: OpaqueMaterial = OpaqueMaterial(
        Identifier="Slate (Slate_Black_0.14 albedo)",
        Thickness=0.01,
        Conductivity=2.0,
        Density=2771,
        SpecificHeat=760,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Smooth",
        ThermalAbsorptance=0.93,
        SolarAbsorptance=0.86,
        VisibleAbsorptance=0.86,
    )
    FB01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Fibre Cement (Fibre Cement_Black_0.05 albedo)",
        Thickness=0.01,
        Conductivity=0.25,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.95,
        VisibleAbsorptance=0.95,
    )
    FB02: OpaqueMaterial = OpaqueMaterial(
        Identifier="Fibre Cement (Fibre Cement_Black_0.06 albedo)",
        Thickness=0.01,
        Conductivity=0.25,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.94,
        VisibleAbsorptance=0.94,
    )
    CR01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Ceramic (Ceramic_Red_0.31 albedo)",
        Thickness=0.01,
        Conductivity=5.0,
        Density=3800,
        SpecificHeat=790,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.93,
        SolarAbsorptance=0.69,
        VisibleAbsorptance=0.69,
    )
    CR02: OpaqueMaterial = OpaqueMaterial(
        Identifier="Ceramic (Ceramic_Brown_0.2 albedo)",
        Thickness=0.01,
        Conductivity=5.0,
        Density=3800,
        SpecificHeat=790,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.93,
        SolarAbsorptance=0.8,
        VisibleAbsorptance=0.8,
    )
    CR03: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement_Rustic red_0.32 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.91,
        SolarAbsorptance=0.6799999999999999,
        VisibleAbsorptance=0.6799999999999999,
    )
    CR04: OpaqueMaterial = OpaqueMaterial(
        Identifier="Ceramic (Ceramic_Burnt red_0.24 albedo)",
        Thickness=0.01,
        Conductivity=1.99,
        Density=3800,
        SpecificHeat=908,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.92,
        SolarAbsorptance=0.76,
        VisibleAbsorptance=0.76,
    )
    CM11: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement_Rustic red with dark shading_0.17 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Smooth",
        ThermalAbsorptance=0.96,
        SolarAbsorptance=0.83,
        VisibleAbsorptance=0.83,
    )
    CM12: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement_Slate grey_0.12 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.88,
        VisibleAbsorptance=0.88,
    )
    CR05: OpaqueMaterial = OpaqueMaterial(
        Identifier="Ceramic (Ceramic_Black_0.16 albedo)",
        Thickness=0.01,
        Conductivity=5.0,
        Density=3800,
        SpecificHeat=790,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.92,
        SolarAbsorptance=0.84,
        VisibleAbsorptance=0.84,
    )
    CM13: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement_Rustic red_0.26 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.74,
        VisibleAbsorptance=0.74,
    )
    CM14: OpaqueMaterial = OpaqueMaterial(
        Identifier="Cement (Cement_Autumn red_0.19 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.92,
        SolarAbsorptance=0.81,
        VisibleAbsorptance=0.81,
    )
    CR06: OpaqueMaterial = OpaqueMaterial(
        Identifier="Ceramic (Ceramic_Red_0.19 albedo)",
        Thickness=0.01,
        Conductivity=5.0,
        Density=3800,
        SpecificHeat=790,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.81,
        VisibleAbsorptance=0.81,
    )
    CR07: OpaqueMaterial = OpaqueMaterial(
        Identifier="Ceramic (Ceramic_Red_0.13 albedo)",
        Thickness=0.01,
        Conductivity=5.0,
        Density=3800,
        SpecificHeat=790,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.87,
        VisibleAbsorptance=0.87,
    )
    CR08: OpaqueMaterial = OpaqueMaterial(
        Identifier="Ceramic (Ceramic_Red_0.12 albedo)",
        Thickness=0.01,
        Conductivity=5.0,
        Density=3800,
        SpecificHeat=790,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.88,
        VisibleAbsorptance=0.88,
    )
    MT01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal (Aluminium Plus Zinc_Grey_dull_0.36 albedo)",
        Thickness=0.01,
        Conductivity=45.0,
        Density=7000,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.58,
        SolarAbsorptance=0.64,
        VisibleAbsorptance=0.64,
    )
    MT02: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal (Aluminium_Stucco_Grey_shiny_0.25 albedo)",
        Thickness=0.01,
        Conductivity=45.0,
        Density=7000,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.16,
        SolarAbsorptance=0.75,
        VisibleAbsorptance=0.75,
    )
    MT03: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal (Metal With Paint_Green_0.11 albedo)",
        Thickness=0.01,
        Conductivity=45.0,
        Density=7000,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.93,
        SolarAbsorptance=0.89,
        VisibleAbsorptance=0.89,
    )
    MT04: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal (Metal With Paint_Copper patina_0.45 albedo)",
        Thickness=0.01,
        Conductivity=0.8,
        Density=2100,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.55,
        VisibleAbsorptance=0.55,
    )
    MT05: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal (Metal With Paint_Slate grey_0.12 albedo)",
        Thickness=0.01,
        Conductivity=1.0,
        Density=2200,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.88,
        VisibleAbsorptance=0.88,
    )
    MT06: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal (Aluminium_Grey_0.26 albedo)",
        Thickness=0.01,
        Conductivity=45.0,
        Density=2300,
        SpecificHeat=1030,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.81,
        SolarAbsorptance=0.74,
        VisibleAbsorptance=0.74,
    )
    MT07: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal (Lead_Grey_0.21 albedo)",
        Thickness=0.01,
        Conductivity=45.0,
        Density=2300,
        SpecificHeat=1400,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.86,
        SolarAbsorptance=0.79,
        VisibleAbsorptance=0.79,
    )
    MT08: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal (Iron_Black_0.05 albedo)",
        Thickness=0.01,
        Conductivity=45.0,
        Density=2300,
        SpecificHeat=2580,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumSmooth",
        ThermalAbsorptance=0.97,
        SolarAbsorptance=0.95,
        VisibleAbsorptance=0.95,
    )
    PV01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Pvc (Pvc Roofing Material_Lead grey_0.08 albedo)",
        Thickness=0.01,
        Conductivity=0.19,
        Density=2300,
        SpecificHeat=1600,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.92,
        VisibleAbsorptance=0.92,
    )
    PV02: OpaqueMaterial = OpaqueMaterial(
        Identifier="Pvc (Pvc Roofing Material_Light grey_0.43 albedo)",
        Thickness=0.01,
        Conductivity=0.19,
        Density=2300,
        SpecificHeat=1600,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.93,
        SolarAbsorptance=0.5700000000000001,
        VisibleAbsorptance=0.5700000000000001,
    )
    PV03: OpaqueMaterial = OpaqueMaterial(
        Identifier="Pvc (Pvc Roofing Material_Copper brown_0.29 albedo)",
        Thickness=0.01,
        Conductivity=0.19,
        Density=2300,
        SpecificHeat=1600,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="Rough",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.71,
        VisibleAbsorptance=0.71,
    )
    PV04: OpaqueMaterial = OpaqueMaterial(
        Identifier="Pvc (Pvc Roofing Material_Azure blue_0.14 albedo)",
        Thickness=0.01,
        Conductivity=0.19,
        Density=2300,
        SpecificHeat=1600,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.86,
        VisibleAbsorptance=0.86,
    )
    PV05: OpaqueMaterial = OpaqueMaterial(
        Identifier="Pvc (Pvc Roofing Material_Copper brown_0.17 albedo)",
        Thickness=0.01,
        Conductivity=0.19,
        Density=2300,
        SpecificHeat=1600,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.83,
        VisibleAbsorptance=0.83,
    )
    PV06: OpaqueMaterial = OpaqueMaterial(
        Identifier="Pvc (Pvc Roofing Material_Copper patina_0.28 albedo)",
        Thickness=0.01,
        Conductivity=0.19,
        Density=2300,
        SpecificHeat=1600,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.94,
        SolarAbsorptance=0.72,
        VisibleAbsorptance=0.72,
    )
    VG01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Grass (Grass Green Watered_Green_0.27 albedo)",
        Thickness=0.01,
        Conductivity=0.28,
        Density=1600,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.98,
        SolarAbsorptance=0.73,
        VisibleAbsorptance=0.73,
    )
    VG02: OpaqueMaterial = OpaqueMaterial(
        Identifier="Grass (Grass Dry _Yellow_0.17 albedo)",
        Thickness=0.01,
        Conductivity=0.28,
        Density=1600,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.98,
        SolarAbsorptance=0.83,
        VisibleAbsorptance=0.83,
    )
    VG03: OpaqueMaterial = OpaqueMaterial(
        Identifier="Dense Forest (Dense Forest_Green_0.27 albedo)",
        Thickness=0.01,
        Conductivity=0.28,
        Density=1600,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated 25/06/2021)",
        Roughness="VeryRough",
        ThermalAbsorptance=0.98,
        SolarAbsorptance=0.73,
        VisibleAbsorptance=0.73,
    )
    VG04: OpaqueMaterial = OpaqueMaterial(
        Identifier="Wood (Tree_Brown_0.4 albedo)",
        Thickness=0.01,
        Conductivity=0.11,
        Density=545,
        SpecificHeat=200,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated nan)",
        Roughness="Smooth",
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.6,
        VisibleAbsorptance=0.6,
    )
    VS01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Soil (Wood_Brown_0.35 albedo)",
        Thickness=0.01,
        Conductivity=0.28,
        Density=1600,
        SpecificHeat=800,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated nan)",
        Roughness="VeryRough",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.65,
        VisibleAbsorptance=0.65,
    )
    WT01: OpaqueMaterial = OpaqueMaterial(
        Identifier="Water (Water Small_-_0.07 albedo)",
        Thickness=0.01,
        Conductivity=0.68,
        Density=997,
        SpecificHeat=4184,
        Source="https://github.com/Art-Ev/ICEtool_sources (last updated nan)",
        Roughness="MediumRough",
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.9299999999999999,
        VisibleAbsorptance=0.9299999999999999,
    )

    # MATERIALS FROM LBT
    LBT_12InGypsumBoard: OpaqueMaterial = OpaqueMaterial(
        Identifier="1/2 in. Gypsum Board",
        Roughness="Smooth",
        Thickness=0.0127,
        Conductivity=0.15989299909405463,
        Density=800.0018291911765,
        SpecificHeat=1089.2971854559414,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.5,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_12InGypsum: OpaqueMaterial = OpaqueMaterial(
        Identifier="1/2IN Gypsum",
        Roughness="Smooth",
        Thickness=0.0127,
        Conductivity=0.15989299909405608,
        Density=784.9017946651944,
        SpecificHeat=829.4648292921422,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.4,
        VisibleAbsorptance=0.4,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_100MmNormalweightConcreteFloor: OpaqueMaterial = OpaqueMaterial(
        Identifier="100mm Normalweight concrete floor",
        Roughness="MediumRough",
        Thickness=0.1016,
        Conductivity=2.3089016425031033,
        Density=2322.006775596,
        SpecificHeat=832.0199665271966,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_12InNormalweightConcreteFloor: OpaqueMaterial = OpaqueMaterial(
        Identifier="12 in. Normalweight Concrete Floor",
        Roughness="MediumRough",
        Thickness=0.30479999999999996,
        Conductivity=2.308455174420426,
        Density=2322.005309227381,
        SpecificHeat=831.4635397241673,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_1InStucco: OpaqueMaterial = OpaqueMaterial(
        Identifier="1IN Stucco",
        Roughness="Smooth",
        Thickness=0.025299999999999993,
        Conductivity=0.6913373548329222,
        Density=1858.004248296516,
        SpecificHeat=836.4603158042426,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.92,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_25MmStucco: OpaqueMaterial = OpaqueMaterial(
        Identifier="25mm Stucco",
        Roughness="Smooth",
        Thickness=0.0254,
        Conductivity=0.7195184959232496,
        Density=1856.0042437235265,
        SpecificHeat=839.4583814522845,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_4InNormalweightConcreteFloor: OpaqueMaterial = OpaqueMaterial(
        Identifier="4 in. Normalweight Concrete Floor",
        Roughness="MediumRough",
        Thickness=0.1016,
        Conductivity=2.308455174420426,
        Density=2322.005309227381,
        SpecificHeat=831.4635397241673,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_4InNormalweightConcreteWall: OpaqueMaterial = OpaqueMaterial(
        Identifier="4 in. Normalweight Concrete Wall",
        Roughness="MediumRough",
        Thickness=0.1016,
        Conductivity=2.308455174420426,
        Density=2322.005309227381,
        SpecificHeat=831.4635397241673,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_58InGypsumBoard: OpaqueMaterial = OpaqueMaterial(
        Identifier="5/8 in. Gypsum Board",
        Roughness="MediumSmooth",
        Thickness=0.0159,
        Conductivity=0.15989299909405463,
        Density=800.0018291911765,
        SpecificHeat=1089.2971854559414,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_58InPlywood: OpaqueMaterial = OpaqueMaterial(
        Identifier="5/8 in. Plywood",
        Roughness="Smooth",
        Thickness=0.0159,
        Conductivity=0.11991974932054163,
        Density=544.0012438499998,
        SpecificHeat=1209.2198113776988,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_6InHeavyweightConcreteRoof: OpaqueMaterial = OpaqueMaterial(
        Identifier="6 in. Heavyweight Concrete Roof",
        Roughness="MediumRough",
        Thickness=0.15239999999999998,
        Conductivity=2.308455174420426,
        Density=2322.005309227381,
        SpecificHeat=831.4635397241673,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_6InNormalweightConcreteFloor: OpaqueMaterial = OpaqueMaterial(
        Identifier="6 in. Normalweight Concrete Floor",
        Roughness="MediumRough",
        Thickness=0.15239999999999998,
        Conductivity=2.308455174420426,
        Density=2322.005309227381,
        SpecificHeat=831.4635397241673,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_8InConcreteBlockBasementWall: OpaqueMaterial = OpaqueMaterial(
        Identifier="8 in. Concrete Block Basement Wall",
        Roughness="MediumRough",
        Thickness=0.2032,
        Conductivity=1.325113229991986,
        Density=1842.0042117126811,
        SpecificHeat=911.4119570053389,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_8InConcreteBlockWall: OpaqueMaterial = OpaqueMaterial(
        Identifier="8 in. Concrete Block Wall",
        Roughness="MediumRough",
        Thickness=0.2032,
        Conductivity=0.7195184959232496,
        Density=800.0018291911765,
        SpecificHeat=831.4635397241673,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_8InNormalweightConcreteFloor: OpaqueMaterial = OpaqueMaterial(
        Identifier="8 in. Normalweight Concrete Floor",
        Roughness="MediumRough",
        Thickness=0.2032,
        Conductivity=2.308455174420426,
        Density=2322.005309227381,
        SpecificHeat=831.4635397241673,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_8InNormalweightConcreteWall: OpaqueMaterial = OpaqueMaterial(
        Identifier="8 in. Normalweight Concrete Wall",
        Roughness="MediumRough",
        Thickness=0.2032,
        Conductivity=2.308455174420426,
        Density=2322.005309227381,
        SpecificHeat=831.4635397241673,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_8InConcreteHwRefbldg: OpaqueMaterial = OpaqueMaterial(
        Identifier="8IN CONCRETE HW RefBldg",
        Roughness="Rough",
        Thickness=0.2032,
        Conductivity=1.3101232613269185,
        Density=2240.0051217352993,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_8InConcreteHw: OpaqueMaterial = OpaqueMaterial(
        Identifier="8IN Concrete HW",
        Roughness="MediumRough",
        Thickness=0.2033000000000001,
        Conductivity=1.7284433202067357,
        Density=2243.0051285947593,
        SpecificHeat=836.4603158042426,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.65,
        VisibleAbsorptance=0.65,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_AcousticCeiling: OpaqueMaterial = OpaqueMaterial(
        Identifier="Acoustic Ceiling",
        Roughness="MediumSmooth",
        Thickness=0.0127,
        Conductivity=0.056961880927257194,
        Density=288.0006585088233,
        SpecificHeat=1338.1366342435858,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.2,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_AsphaltShingles: OpaqueMaterial = OpaqueMaterial(
        Identifier="Asphalt Shingles",
        Roughness="VeryRough",
        Thickness=0.003199999999999976,
        Conductivity=0.03997324977351388,
        Density=1120.002560867648,
        SpecificHeat=1259.1875721784268,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_AtticfloorInsulation: OpaqueMaterial = OpaqueMaterial(
        Identifier="AtticFloor Insulation",
        Roughness="MediumRough",
        Thickness=0.23789999999999986,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_BuiltUpRoofing: OpaqueMaterial = OpaqueMaterial(
        Identifier="Built-up Roofing",
        Roughness="Rough",
        Thickness=0.009499999999999998,
        Conductivity=0.15989299909405463,
        Density=1120.002560867648,
        SpecificHeat=1459.0586153813556,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_BuiltUpRoofingHighlyReflective: OpaqueMaterial = OpaqueMaterial(
        Identifier="Built-up Roofing - Highly Reflective",
        Roughness="Rough",
        Thickness=0.009499999999999998,
        Conductivity=0.15989299909405463,
        Density=1120.002560867648,
        SpecificHeat=1459.0586153813556,
        ThermalAbsorptance=0.75,
        SolarAbsorptance=0.45,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_BulkStorageProductsMaterial: OpaqueMaterial = OpaqueMaterial(
        Identifier="Bulk Storage Products Material",
        Roughness="Rough",
        Thickness=2.4384049429641195,
        Conductivity=1.4410356988618083,
        Density=200.23111933632342,
        SpecificHeat=836.2604510890168,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_Carpet34InCbes: OpaqueMaterial = OpaqueMaterial(
        Identifier="Carpet - 3/4 in. CBES",
        Roughness="Smooth",
        Thickness=0.019049999999999997,
        Conductivity=0.05000063634028838,
        Density=288.33299999999997,
        SpecificHeat=1380.753138075314,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_F08MetalSurface: OpaqueMaterial = OpaqueMaterial(
        Identifier="F08 Metal surface",
        Roughness="Smooth",
        Thickness=0.0008000000000000003,
        Conductivity=45.249718743617805,
        Density=7824.017889489713,
        SpecificHeat=499.6776080073138,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_F16AcousticTile: OpaqueMaterial = OpaqueMaterial(
        Identifier="F16 Acoustic tile",
        Roughness="MediumSmooth",
        Thickness=0.019100000000000002,
        Conductivity=0.05995987466027089,
        Density=368.0008414279416,
        SpecificHeat=589.6195774486317,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.3,
        VisibleAbsorptance=0.3,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_G0113MmGypsumBoard: OpaqueMaterial = OpaqueMaterial(
        Identifier="G01 13mm gypsum board",
        Roughness="Smooth",
        Thickness=0.0127,
        Conductivity=0.15992392281605958,
        Density=800.0023344,
        SpecificHeat=1090.0261589958159,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.5,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_G01A19MmGypsumBoard: OpaqueMaterial = OpaqueMaterial(
        Identifier="G01a 19mm gypsum board",
        Roughness="MediumSmooth",
        Thickness=0.018999999999999996,
        Conductivity=0.15989299909405608,
        Density=800.0018291911781,
        SpecificHeat=1089.2971854559455,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.4,
        VisibleAbsorptance=0.4,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_G0525MmWood: OpaqueMaterial = OpaqueMaterial(
        Identifier="G05 25mm wood",
        Roughness="MediumSmooth",
        Thickness=0.0254,
        Conductivity=0.14989968665067757,
        Density=608.0013901852949,
        SpecificHeat=1628.949002103841,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.5,
        VisibleAbsorptance=0.5,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_Ground_Floor_R11_T2013: OpaqueMaterial = OpaqueMaterial(
        Identifier="Ground_Floor_R11_T2013",
        Roughness="Smooth",
        Thickness=0.044227242,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_Ground_Floor_R17_T2013: OpaqueMaterial = OpaqueMaterial(
        Identifier="Ground_Floor_R17_T2013",
        Roughness="Smooth",
        Thickness=0.06891629599999999,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_Ground_Floor_R22_T2013: OpaqueMaterial = OpaqueMaterial(
        Identifier="Ground_Floor_R22_T2013",
        Roughness="Smooth",
        Thickness=0.08918575,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_GypsumBoard12InCbes: OpaqueMaterial = OpaqueMaterial(
        Identifier="Gypsum Board - 1/2 in. CBES",
        Roughness="Smooth",
        Thickness=0.0127,
        Conductivity=0.16000030671169507,
        Density=640.74,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_GypsumOrPlasterBoard38In: OpaqueMaterial = OpaqueMaterial(
        Identifier="Gypsum Or Plaster Board - 3/8 in.",
        Roughness="MediumSmooth",
        Thickness=0.009499999999999998,
        Conductivity=0.5796121217159504,
        Density=800.0018291911765,
        SpecificHeat=1089.2971854559414,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_HwConcrete: OpaqueMaterial = OpaqueMaterial(
        Identifier="HW CONCRETE",
        Roughness="Rough",
        Thickness=0.1016,
        Conductivity=1.310376644700027,
        Density=2240.0065363199997,
        SpecificHeat=836.8200836820084,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_HwConcrete8In: OpaqueMaterial = OpaqueMaterial(
        Identifier="HW CONCRETE 8 in",
        Roughness="Rough",
        Thickness=0.2032,
        Conductivity=1.310376644700027,
        Density=2240.0065363199997,
        SpecificHeat=836.8200836820084,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_I0125MmInsulationBoard: OpaqueMaterial = OpaqueMaterial(
        Identifier="I01 25mm insulation board",
        Roughness="MediumRough",
        Thickness=0.0254,
        Conductivity=0.029979937330135372,
        Density=43.000098319025845,
        SpecificHeat=1209.2198113776988,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.6,
        VisibleAbsorptance=0.6,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_I0250MmInsulationBoard: OpaqueMaterial = OpaqueMaterial(
        Identifier="I02 50mm insulation board",
        Roughness="MediumRough",
        Thickness=0.0508,
        Conductivity=0.029979937330135372,
        Density=43.000098319025845,
        SpecificHeat=1209.2198113776988,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.6,
        VisibleAbsorptance=0.6,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_IeadNonresRoofInsulation176: OpaqueMaterial = OpaqueMaterial(
        Identifier="IEAD NonRes Roof Insulation-1.76",
        Roughness="MediumRough",
        Thickness=0.07398050297987159,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_IeadRoofInsulationR347Ip: OpaqueMaterial = OpaqueMaterial(
        Identifier="IEAD Roof Insulation R-3.47 IP",
        Roughness="MediumRough",
        Thickness=0.029942459748632622,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_Insulation1M: OpaqueMaterial = OpaqueMaterial(
        Identifier="Insulation 1m",
        Roughness="Smooth",
        Thickness=0.9999979999999999,
        Conductivity=0.01999967801037276,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_M01100MmBrick: OpaqueMaterial = OpaqueMaterial(
        Identifier="M01 100mm brick",
        Roughness="MediumRough",
        Thickness=0.1016,
        Conductivity=0.8894048074606842,
        Density=1920.0043900588325,
        SpecificHeat=789.4906206515565,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_M11100MmLightweightConcrete: OpaqueMaterial = OpaqueMaterial(
        Identifier="M11 100mm lightweight concrete",
        Roughness="MediumRough",
        Thickness=0.1016,
        Conductivity=0.5296455594990593,
        Density=1280.0029267058849,
        SpecificHeat=839.4583814522887,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.5,
        VisibleAbsorptance=0.5,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_M15200MmHeavyweightConcrete: OpaqueMaterial = OpaqueMaterial(
        Identifier="M15 200mm heavyweight concrete",
        Roughness="MediumRough",
        Thickness=0.2032,
        Conductivity=1.9486959264588086,
        Density=2240.0051217352993,
        SpecificHeat=899.4196944131631,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.5,
        VisibleAbsorptance=0.5,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MatCc054HwConcrete: OpaqueMaterial = OpaqueMaterial(
        Identifier="MAT-CC05 4 HW CONCRETE",
        Roughness="Rough",
        Thickness=0.1016,
        Conductivity=1.3101232613269185,
        Density=2240.0051217352993,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.85,
        VisibleAbsorptance=0.85,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MatCc058HwConcrete: OpaqueMaterial = OpaqueMaterial(
        Identifier="MAT-CC05 8 HW CONCRETE",
        Roughness="Rough",
        Thickness=0.2032,
        Conductivity=1.3101232613269185,
        Density=2240.0051217352993,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.85,
        VisibleAbsorptance=0.85,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MassNonresWallInsulation043: OpaqueMaterial = OpaqueMaterial(
        Identifier="Mass NonRes Wall Insulation-0.43",
        Roughness="MediumRough",
        Thickness=0.0004367814266280437,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MassWallInsulationR423Ip: OpaqueMaterial = OpaqueMaterial(
        Identifier="Mass Wall Insulation R-4.23 IP",
        Roughness="MediumRough",
        Thickness=0.0365371685816,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MetalBuildingSemiCondWallInsulation054: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal Building Semi-Cond Wall Insulation-0.54",
        Roughness="MediumRough",
        Thickness=0.015345987404915393,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MetalBuildingWallInsulationR414Ip: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal Building Wall Insulation R-4.14 IP",
        Roughness="MediumRough",
        Thickness=0.0356729086642,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MetalDecking: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal Decking",
        Roughness="MediumSmooth",
        Thickness=0.0014999999999999994,
        Conductivity=44.97590198266915,
        Density=7680.017560235314,
        SpecificHeat=418.1302223805201,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.6,
        VisibleAbsorptance=0.6,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MetalRoofInsulationR521Ip: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal Roof Insulation R-5.21 IP",
        Roughness="MediumRough",
        Thickness=0.044938850457399995,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MetalRoofSurface: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal Roof Surface",
        Roughness="Smooth",
        Thickness=0.0007999999999999979,
        Conductivity=45.24971874361766,
        Density=7824.017889489713,
        SpecificHeat=499.67760800730963,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MetalRoofSurfaceHighlyReflective: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal Roof Surface - Highly Reflective",
        Roughness="Smooth",
        Thickness=0.0007999999999999979,
        Conductivity=45.24971874361766,
        Density=7824.017889489713,
        SpecificHeat=499.67760800730963,
        ThermalAbsorptance=0.75,
        SolarAbsorptance=0.45,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MetalRoofing: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal Roofing",
        Roughness="MediumSmooth",
        Thickness=0.0014999999999999994,
        Conductivity=44.97590198266915,
        Density=7680.017560235314,
        SpecificHeat=418.1302223805201,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.6,
        VisibleAbsorptance=0.6,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MetalRoofingHighlyReflective: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal Roofing - Highly Reflective",
        Roughness="MediumSmooth",
        Thickness=0.0014999999999999994,
        Conductivity=44.97590198266915,
        Density=7680.017560235314,
        SpecificHeat=418.1302223805201,
        ThermalAbsorptance=0.75,
        SolarAbsorptance=0.45,
        VisibleAbsorptance=0.6,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MetalSemiCondRoofInsulation105: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal Semi-Cond Roof Insulation-1.05",
        Roughness="MediumRough",
        Thickness=0.04226733019063131,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MetalSiding: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal Siding",
        Roughness="Smooth",
        Thickness=0.0014999999999999994,
        Conductivity=44.92993274542956,
        Density=7688.877580493597,
        SpecificHeat=409.7356385659975,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MetalStandingSeam116InCbes: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal Standing Seam - 1/16 in. CBES",
        Roughness="MediumRough",
        Thickness=0.001524,
        Conductivity=0.579999310186949,
        Density=7820.552070000001,
        SpecificHeat=502.09205020920496,
        ThermalAbsorptance=0.85,
        SolarAbsorptance=0.37,
        VisibleAbsorptance=0.85,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MetalSurface: OpaqueMaterial = OpaqueMaterial(
        Identifier="Metal Surface",
        Roughness="Smooth",
        Thickness=0.0007999999999999979,
        Conductivity=45.24971874361766,
        Density=7824.017889489713,
        SpecificHeat=499.67760800730963,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_Nacm_Carpet34In: OpaqueMaterial = OpaqueMaterial(
        Identifier="NACM_Carpet 3/4in",
        Roughness="Smooth",
        Thickness=0.019049999999999997,
        Conductivity=0.05000063634028838,
        Density=288.33299999999997,
        SpecificHeat=1380.753138075314,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.9,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_Nacm_Concrete4In_140LbFt3: OpaqueMaterial = OpaqueMaterial(
        Identifier="NACM_Concrete 4in_140lb/ft3",
        Roughness="MediumRough",
        Thickness=0.1016,
        Conductivity=1.9500003149271867,
        Density=2239.0659299999998,
        SpecificHeat=920.5020920502092,
        ThermalAbsorptance=0.75,
        SolarAbsorptance=0.92,
        VisibleAbsorptance=0.75,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_Nacm_GypsumBoard58In: OpaqueMaterial = OpaqueMaterial(
        Identifier="NACM_Gypsum Board 5/8in",
        Roughness="MediumRough",
        Thickness=0.016002,
        Conductivity=0.16000030671169507,
        Density=640.74,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.5,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_Plywood58InCbes: OpaqueMaterial = OpaqueMaterial(
        Identifier="Plywood - 5/8 in. CBES",
        Roughness="Smooth",
        Thickness=0.016002,
        Conductivity=0.11999950937659305,
        Density=480.555,
        SpecificHeat=1882.8451882845188,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_R1_R1420: OpaqueMaterial = OpaqueMaterial(
        Identifier="R1_R14.20",
        Roughness="Smooth",
        Thickness=0.05756529,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_R2_R1290: OpaqueMaterial = OpaqueMaterial(
        Identifier="R2_R12.90",
        Roughness="Smooth",
        Thickness=0.052295298000000004,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_R3_R1774: OpaqueMaterial = OpaqueMaterial(
        Identifier="R3_R17.74",
        Roughness="Smooth",
        Thickness=0.071916036,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_R4_R2028: OpaqueMaterial = OpaqueMaterial(
        Identifier="R4_R20.28",
        Roughness="Smooth",
        Thickness=0.082211418,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_R_R30: OpaqueMaterial = OpaqueMaterial(
        Identifier="R_R30",
        Roughness="Smooth",
        Thickness=0.121616978,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_R_R38: OpaqueMaterial = OpaqueMaterial(
        Identifier="R_R38",
        Roughness="Smooth",
        Thickness=0.154048206,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_R_T24_2013_2486: OpaqueMaterial = OpaqueMaterial(
        Identifier="R_T24_2013_24.86",
        Roughness="Smooth",
        Thickness=0.100779834,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_RoofInsulation18: OpaqueMaterial = OpaqueMaterial(
        Identifier="Roof Insulation [18]",
        Roughness="MediumRough",
        Thickness=0.16929999999999995,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_RoofMembrane: OpaqueMaterial = OpaqueMaterial(
        Identifier="Roof Membrane",
        Roughness="VeryRough",
        Thickness=0.009499999999999998,
        Conductivity=0.15989299909405608,
        Density=1121.29256381722,
        SpecificHeat=1459.0586153813556,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_RoofMembraneHighlyReflective: OpaqueMaterial = OpaqueMaterial(
        Identifier="Roof Membrane - Highly Reflective",
        Roughness="VeryRough",
        Thickness=0.009499999999999998,
        Conductivity=0.15989299909405608,
        Density=1121.29256381722,
        SpecificHeat=1459.0586153813556,
        ThermalAbsorptance=0.75,
        SolarAbsorptance=0.45,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_StdWood6In: OpaqueMaterial = OpaqueMaterial(
        Identifier="Std Wood 6 in.",
        Roughness="MediumSmooth",
        Thickness=0.14999999999999997,
        Conductivity=0.11991974932054163,
        Density=540.0012347040437,
        SpecificHeat=1209.2198113776988,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_StdWood6InchFurnishings: OpaqueMaterial = OpaqueMaterial(
        Identifier="Std Wood 6inch Furnishings",
        Roughness="MediumSmooth",
        Thickness=0.150114,
        Conductivity=0.11999950937659305,
        Density=539.983635,
        SpecificHeat=1213.389121338912,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_SteelFrameNonresWallInsulation073: OpaqueMaterial = OpaqueMaterial(
        Identifier="Steel Frame NonRes Wall Insulation-0.73",
        Roughness="MediumRough",
        Thickness=0.020278246975300705,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_SteelFrameWallInsulationR102Ip: OpaqueMaterial = OpaqueMaterial(
        Identifier="Steel Frame Wall Insulation R-1.02 IP",
        Roughness="MediumRough",
        Thickness=0.008836428479,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_SteelFrameCavity: OpaqueMaterial = OpaqueMaterial(
        Identifier="Steel Frame/Cavity",
        Roughness="MediumRough",
        Thickness=0.08890018021223342,
        Conductivity=0.08405993542170573,
        Density=641.3803214581126,
        SpecificHeat=501.75627065341007,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_Stucco78InCbes: OpaqueMaterial = OpaqueMaterial(
        Identifier="Stucco - 7/8 in. CBES",
        Roughness="MediumRough",
        Thickness=0.022352,
        Conductivity=0.7000002608778985,
        Density=1855.102485,
        SpecificHeat=836.8200836820084,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_W1_R860: OpaqueMaterial = OpaqueMaterial(
        Identifier="W1_R8.60",
        Roughness="Smooth",
        Thickness=0.034862008,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_W2_R1113: OpaqueMaterial = OpaqueMaterial(
        Identifier="W2_R11.13",
        Roughness="Smooth",
        Thickness=0.045119036,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_W3_R1136: OpaqueMaterial = OpaqueMaterial(
        Identifier="W3_R11.36",
        Roughness="Smooth",
        Thickness=0.04605147,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_W4_R1262: OpaqueMaterial = OpaqueMaterial(
        Identifier="W4_R12.62",
        Roughness="Smooth",
        Thickness=0.05115915599999999,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_W_T24_2013_R1399: OpaqueMaterial = OpaqueMaterial(
        Identifier="W_T24_2013_R13.99",
        Roughness="Smooth",
        Thickness=0.056714136,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_W_M1_R15: OpaqueMaterial = OpaqueMaterial(
        Identifier="W_m1_R15",
        Roughness="Smooth",
        Thickness=0.060808362,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_W_M2_R19: OpaqueMaterial = OpaqueMaterial(
        Identifier="W_m2_R19",
        Roughness="Smooth",
        Thickness=0.077023976,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_W_M3_R21: OpaqueMaterial = OpaqueMaterial(
        Identifier="W_m3_R21",
        Roughness="Smooth",
        Thickness=0.08513190999999999,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_WallInsulation31: OpaqueMaterial = OpaqueMaterial(
        Identifier="Wall Insulation [31]",
        Roughness="MediumRough",
        Thickness=0.03370000000000008,
        Conductivity=0.043171109755394975,
        Density=91.0002080704965,
        SpecificHeat=836.4603158042426,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.5,
        VisibleAbsorptance=0.5,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_WoodFrameNonresWallInsulation073: OpaqueMaterial = OpaqueMaterial(
        Identifier="Wood Frame NonRes Wall Insulation-0.73",
        Roughness="MediumRough",
        Thickness=0.020278246975300705,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_WoodFrameWallInsulationR161Ip: OpaqueMaterial = OpaqueMaterial(
        Identifier="Wood Frame Wall Insulation R-1.61 IP",
        Roughness="MediumRough",
        Thickness=0.013873826709999999,
        Conductivity=0.04896723097255453,
        Density=265.0006059195773,
        SpecificHeat=836.2604447610418,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_WoodSiding: OpaqueMaterial = OpaqueMaterial(
        Identifier="Wood Siding",
        Roughness="MediumSmooth",
        Thickness=0.010000000000000004,
        Conductivity=0.10992643687716327,
        Density=544.6212452676245,
        SpecificHeat=1209.2198113776988,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.78,
        VisibleAbsorptance=0.78,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_Ceiling_2_Insulation: OpaqueMaterial = OpaqueMaterial(
        Identifier="ceiling_2_Insulation",
        Roughness="Smooth",
        Thickness=0.100778564,
        Conductivity=0.02300770107232491,
        Density=16.0185,
        SpecificHeat=1129.7071129707113,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_DrySand: OpaqueMaterial = OpaqueMaterial(
        Identifier="Dry Sand",
        Roughness="Rough",
        Thickness=0.2,
        Conductivity=0.33,
        Density=1555.0,
        SpecificHeat=800.0,
        ThermalAbsorptance=0.85,
        SolarAbsorptance=0.65,
        VisibleAbsorptance=0.65,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_DryDust: OpaqueMaterial = OpaqueMaterial(
        Identifier="Dry Dust",
        Roughness="Rough",
        Thickness=0.2,
        Conductivity=0.5,
        Density=1600.0,
        SpecificHeat=1026.0,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.7,
        VisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_MoistSoil: OpaqueMaterial = OpaqueMaterial(
        Identifier="Moist Soil",
        Roughness="Rough",
        Thickness=0.2,
        Conductivity=1.0,
        Density=1250.0,
        SpecificHeat=1252.0,
        ThermalAbsorptance=0.92,
        SolarAbsorptance=0.75,
        VisibleAbsorptance=0.75,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_Mud: OpaqueMaterial = OpaqueMaterial(
        Identifier="Mud",
        Roughness="MediumRough",
        Thickness=0.2,
        Conductivity=1.4,
        Density=1840.0,
        SpecificHeat=1480.0,
        ThermalAbsorptance=0.95,
        SolarAbsorptance=0.8,
        VisibleAbsorptance=0.8,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_ConcretePavement: OpaqueMaterial = OpaqueMaterial(
        Identifier="Concrete Pavement",
        Roughness="MediumRough",
        Thickness=0.2,
        Conductivity=1.73,
        Density=2243.0,
        SpecificHeat=837.0,
        ThermalAbsorptance=0.9,
        SolarAbsorptance=0.65,
        VisibleAbsorptance=0.65,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_AsphaltPavement: OpaqueMaterial = OpaqueMaterial(
        Identifier="Asphalt Pavement",
        Roughness="MediumRough",
        Thickness=0.2,
        Conductivity=0.75,
        Density=2360.0,
        SpecificHeat=920.0,
        ThermalAbsorptance=0.93,
        SolarAbsorptance=0.87,
        VisibleAbsorptance=0.87,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_SolidRock: OpaqueMaterial = OpaqueMaterial(
        Identifier="Solid Rock",
        Roughness="MediumRough",
        Thickness=0.2,
        Conductivity=3.0,
        Density=2700.0,
        SpecificHeat=790.0,
        ThermalAbsorptance=0.96,
        SolarAbsorptance=0.55,
        VisibleAbsorptance=0.55,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
    LBT_GrassyLawn: OpaqueVegetationMaterial = OpaqueVegetationMaterial(
        Identifier="Grassy Lawn",
        PlantHeight=0.2,
        LeafAreaIndex=1.0,
        LeafReflectivity=0.22,
        LeafEmissivity=0.95,
        MinStomatalResist=180.0,
        Roughness="MediumRough",
        Thickness=0.1,
        Conductivity=0.35,
        Density=1100.0,
        SpecificHeat=1200.0,
        SoilThermalAbsorptance=0.9,
        SoilSolarAbsorptance=0.7,
        SoilVisibleAbsorptance=0.7,
        Source=f"LadybugTools honeybee_energy_standards v{HONEYBEE_MATERIAL_STANDARDS_VERSION}",
    )
