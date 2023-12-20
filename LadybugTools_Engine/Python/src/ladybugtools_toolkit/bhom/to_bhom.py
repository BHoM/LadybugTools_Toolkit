"""Convert methods for objects generic to Ladybug."""

from honeybee_energy.material.opaque import EnergyMaterial, EnergyMaterialVegetation
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW, Location
from ladybug.header import DataTypeBase, Header
from ladybug_geometry.geometry3d.pointvector import Point3D


def material_to_bhom(obj: EnergyMaterial | EnergyMaterialVegetation) -> dict:
    """Convert this object into a BHOM deserialisable dictionary."""

    if isinstance(obj, EnergyMaterial):
        return energymaterial_to_bhom(obj)
    elif isinstance(obj, EnergyMaterialVegetation):
        return energymaterialvegetation_to_bhom(obj)
    else:
        raise ValueError(
            f"Unexpected type {type(obj)} for material_to_bhom. "
            "Expected EnergyMaterial or EnergyMaterialVegetation."
        )


def energymaterial_to_bhom(obj: EnergyMaterial) -> dict:
    """Convert this object into a BHOM deserialisable dictionary."""
    return {
        "_t": "BH.oM.LadybugTools.EnergyMaterial",
        "Type": "EnergyMaterial",
        "Identifier": obj.identifier,
        "Roughness": obj.roughness,
        "Thickness": obj.thickness,
        "Conductivity": obj.conductivity,
        "Density": obj.density,
        "SpecificHeat": obj.specific_heat,
        "ThermalAbsorptance": obj.thermal_absorptance,
        "SolarAbsorptance": obj.solar_absorptance,
        "VisibleAbsorptance": obj.visible_absorptance,
    }


def energymaterialvegetation_to_bhom(obj: EnergyMaterialVegetation) -> dict:
    """Convert this object into a BHOM deserialisable dictionary."""
    return {
        "_t": "BH.oM.LadybugTools.EnergyMaterialVegetation",
        "Type": "EnergyMaterialVegetation",
        "Identifier": obj.identifier,
        "Thickness": obj.thickness,
        "Conductivity": obj.conductivity,
        "Density": obj.density,
        "SpecificHeat": obj.specific_heat,
        "Roughness": obj.roughness,
        "SoilThermalAbsorptance": obj.soil_thermal_absorptance,
        "SoilSolarAbsorptance": obj.soil_solar_absorptance,
        "SoilVisibleAbsorptance": obj.soil_visible_absorptance,
        "PlantHeight": obj.plant_height,
        "LeafAreaIndex": obj.leaf_area_index,
        "LeafReflectivity": obj.leaf_reflectivity,
        "LeafEmissivity": obj.leaf_emissivity,
        "MinStomatalResist": obj.min_stomatal_resist,
    }


def point3d_to_bhom(obj: Point3D) -> dict:
    """Convert this object into a BHOM deserialisable dictionary."""
    return {
        "_t": "BH.oM.Geometry.Point",
        "X": obj.x,
        "Y": obj.y,
        "Z": obj.z,
    }


def analysisperiod_to_bhom(obj: AnalysisPeriod) -> dict:
    """Convert this object into a BHOM deserialisable dictionary."""
    return {
        "_t": "BH.oM.LadybugTools.AnalysisPeriod",
        "Type": "AnalysisPeriod",
        "StHour": obj.st_hour,
        "EndHour": obj.end_hour,
        "StDay": obj.st_day,
        "EndDay": obj.end_day,
        "StMonth": obj.st_month,
        "EndMonth": obj.end_month,
        "IsLeapYear": obj.is_leap_year,
        "Timestep": obj.timestep,
    }


def datatype_to_bhom(obj: DataTypeBase) -> dict:
    """Convert this object into a BHOM deserialisable dictionary."""
    return {
        "_t": "BH.oM.LadybugTools.DataType",
        "Type": "DataType",
        "Name": obj.to_dict()["name"],
        "Data_Type": obj.to_dict()["data_type"],
    }


def header_to_bhom(obj: Header) -> dict:
    """Convert this object into a BHOM deserialisable dictionary."""
    return {
        "_t": "BH.oM.LadybugTools.Header",
        "Type": "Header",
        "DataType": datatype_to_bhom(obj.data_type),
        "Unit": obj.unit,
        "AnalysisPeriod": analysisperiod_to_bhom(obj.analysis_period),
        "Metadata": obj.metadata,
    }


def hourlycontinuouscollection_to_bhom(obj: HourlyContinuousCollection) -> dict:
    """Convert this object into a BHOM deserialisable dictionary."""
    return {
        "_t": "BH.oM.LadybugTools.HourlyContinuousCollection",
        "Type": "HourlyContinuous",
        "Header": header_to_bhom(obj.header),
        "Values": obj.values,
    }


def location_to_bhom(obj: Location) -> dict:
    """Convert this object into a BHOM deserialisable dictionary."""
    return {
        "_t": "BH.oM.LadybugTools.Location",
        "Type": "Location",
        "City": obj.city,
        "State": obj.state,
        "Country": obj.country,
        "Latitude": obj.latitude,
        "Longitude": obj.longitude,
        "TimeZone": obj.time_zone,
        "Elevation": obj.elevation,
        "StationID": obj.station_id,
        "Source": obj.source,
    }


def epw_to_bhom(obj: EPW) -> dict:
    """Convert this object into a BHOM deserialisable dictionary."""

    # call to property required to populate slots
    obj.dry_bulb_temperature  # pylint: disable=pointless-statement

    collections = []
    for collection in obj._data:  # pylint: disable=protected-access
        if isinstance(collection[0], (int, float)):
            collections.append(hourlycontinuouscollection_to_bhom(collection))
    return {
        "_t": "BH.oM.LadybugTools.EPW",
        "Type": "EPW",
        "Location": location_to_bhom(obj.location),
        "Metadata": obj.metadata,
        "DataCollections": collections,
    }
