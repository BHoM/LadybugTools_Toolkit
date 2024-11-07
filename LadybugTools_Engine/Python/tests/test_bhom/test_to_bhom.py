from honeybee_energy.material.opaque import (EnergyMaterial,
                                             EnergyMaterialVegetation)
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import Temperature
from ladybug.header import DataTypeBase, Header
from ladybug_geometry.geometry3d.pointvector import Point3D
from ladybugtools_toolkit.bhom.to_bhom import (
    analysisperiod_to_bhom, datatype_to_bhom, energymaterial_to_bhom,
    energymaterialvegetation_to_bhom, epw_to_bhom, header_to_bhom,
    hourlycontinuouscollection_to_bhom, location_to_bhom, material_to_bhom,
    point3d_to_bhom)

from .. import EPW_OBJ

ENERGY_MATERIAL_VEGETATION = EnergyMaterialVegetation(
    identifier="test",
    thickness=0.1,
    conductivity=0.2,
    density=0.3,
    specific_heat=1000,
    roughness="Smooth",
    soil_thermal_absorptance=0.5,
    soil_solar_absorptance=0.6,
    soil_visible_absorptance=0.7,
    plant_height=0.8,
    leaf_area_index=0.9,
    leaf_reflectivity=0.1,
    leaf_emissivity=0.8,
    min_stomatal_resist=100,
)

ENERGY_MATERIAL = EnergyMaterial(
    identifier="test",
    roughness="Smooth",
    thickness=1.0,
    conductivity=0.5,
    density=2.0,
    specific_heat=1000.0,
    thermal_absorptance=0.6,
    solar_absorptance=0.7,
    visible_absorptance=0.8,
)


def test_energymaterialvegetation_to_bhom():
    """_"""

    result = energymaterialvegetation_to_bhom(ENERGY_MATERIAL_VEGETATION)

    assert result["_t"] == "BH.oM.LadybugTools.EnergyMaterialVegetation"
    assert result["Type"] == "EnergyMaterialVegetation"
    assert result["Identifier"] == "test"
    assert result["Thickness"] == 0.1
    assert result["Conductivity"] == 0.2
    assert result["Density"] == 0.3
    assert result["SpecificHeat"] == 1000
    assert result["Roughness"] == "Smooth"
    assert result["SoilThermalAbsorptance"] == 0.5
    assert result["SoilSolarAbsorptance"] == 0.6
    assert result["SoilVisibleAbsorptance"] == 0.7
    assert result["PlantHeight"] == 0.8
    assert result["LeafAreaIndex"] == 0.9
    assert result["LeafReflectivity"] == 0.1
    assert result["LeafEmissivity"] == 0.8
    assert result["MinStomatalResist"] == 100


def test_energymaterial_to_bhom():
    """_"""

    result = energymaterial_to_bhom(ENERGY_MATERIAL)

    assert result["_t"] == "BH.oM.LadybugTools.EnergyMaterial"
    assert result["Type"] == "EnergyMaterial"
    assert result["Identifier"] == "test"
    assert result["Roughness"] == "Smooth"
    assert result["Thickness"] == 1.0
    assert result["Conductivity"] == 0.5
    assert result["Density"] == 2.0
    assert result["SpecificHeat"] == 1000.0
    assert result["ThermalAbsorptance"] == 0.6
    assert result["SolarAbsorptance"] == 0.7
    assert result["VisibleAbsorptance"] == 0.8


def test_material_to_bhom():
    """_"""

    assert material_to_bhom(ENERGY_MATERIAL)["Roughness"] == "Smooth"
    assert material_to_bhom(ENERGY_MATERIAL_VEGETATION)[
        "MinStomatalResist"] == 100


def test_point3d_to_bhom():
    """_"""
    point3d = Point3D(1.0, 2.0, 3.0)

    result = point3d_to_bhom(point3d)

    assert result["_t"] == "BH.oM.Geometry.Point"
    assert result["X"] == 1.0
    assert result["Y"] == 2.0
    assert result["Z"] == 3.0


def test_analysisperiod_to_bhom():
    """Test analysisperiod_to_bhom function."""
    obj = AnalysisPeriod(
        st_month=1,
        st_day=1,
        end_month=12,
        end_day=31,
        st_hour=0,
        end_hour=23,
        timestep=1,
        is_leap_year=False,
    )

    result = analysisperiod_to_bhom(obj)

    assert result["_t"] == "BH.oM.LadybugTools.AnalysisPeriod"
    assert result["Type"] == "AnalysisPeriod"
    assert result["StHour"] == 0
    assert result["EndHour"] == 23
    assert result["StDay"] == 1
    assert result["EndDay"] == 31
    assert result["StMonth"] == 1
    assert result["EndMonth"] == 12
    assert result["IsLeapYear"] is False
    assert result["Timestep"] == 1


def test_datatype_to_bhom():
    """Test datatype_to_bhom function."""
    obj = Temperature()
    result = datatype_to_bhom(obj)

    assert result["_t"] == "BH.oM.LadybugTools.DataType"
    assert result["Type"] == "DataType"
    assert result["Name"] == "Temperature"
    assert result["Data_Type"] == "Temperature"


def test_header_to_bhom():
    """Test header_to_bhom function."""
    obj = Header(
        data_type=Temperature(),
        unit="C",
        analysis_period=AnalysisPeriod(),
        metadata={},
    )

    result = header_to_bhom(obj)

    assert result["_t"] == "BH.oM.LadybugTools.Header"
    assert result["Type"] == "Header"
    assert result["Unit"] == "C"
    assert result["Metadata"] == {}


def test_hourlycontinuouscollection_to_bhom():
    """Test conversion of HourlyContinuousCollection to BHOM dictionary."""
    header = Header(
        analysis_period=AnalysisPeriod(),
        data_type=Temperature(),
        unit="C",
    )
    values = [0.0] * 8760
    obj = HourlyContinuousCollection(header=header, values=values)

    result = hourlycontinuouscollection_to_bhom(obj)

    assert result["_t"] == "BH.oM.LadybugTools.HourlyContinuousCollection"
    assert result["Type"] == "HourlyContinuous"
    assert sum(result["Values"]) == 0


def test_location_to_bhom():
    """_"""
    result = location_to_bhom(EPW_OBJ.location)

    assert len(result) == 11
    assert result["_t"] == "BH.oM.LadybugTools.Location"
    assert result["Type"] == "Location"
    assert result["City"] == "LONDON GATWICK"


def test_epw_to_bhom():
    """_"""
    result = epw_to_bhom(EPW_OBJ)

    assert len(result) == 5
    assert result["_t"] == "BH.oM.LadybugTools.EPW"
    assert result["Type"] == "EPW"
    assert len(result["DataCollections"]) == 34
