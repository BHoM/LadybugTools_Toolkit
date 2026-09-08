from honeybee_energy.material.opaque import EnergyMaterial, EnergyMaterialVegetation
from ladybug.analysisperiod import AnalysisPeriod as AnalysisPeriodBase
from ladybug.datacollection import HourlyContinuousCollection as HC
from ladybug.epw import EPW, Location
from ladybug.header import DataTypeBase, Header as HeaderBase
from ladybug_geometry.geometry3d.pointvector import Point3D
from python_toolkit.bhom.bhom_object import BHoMObject, IObject, BHoMJSONDecoder

#make custom classes for converting to ladybug objects from bhom objects (where type names and some other differences occur)
class Point():
    @classmethod
    def from_dict(cls, d) -> Point3D:
        d["type"] = "Point3D"

        return Point3D.from_dict(d)

class DataType():
    @classmethod
    def from_dict(cls, d) -> dict:
        d["type"] = "DataTypeBase"
        d["data_type"] = d["data__type"]
        return DataTypeBase.from_dict(d)

class AnalysisPeriod():
    @classmethod
    def from_dict(cls, d) -> dict:
        d["st_hour"] = d["start_hour"]
        d["st_day"] = d["start_day"]
        d["st_month"] = d["start_month"]
        return AnalysisPeriodBase.from_dict(d)

class HourlyContinuousCollection():
    @classmethod
    def from_dict(cls, d) -> dict:
        d["type"] = "HourlyContinuous"
        #see comment in Header() below for the reason the header is converted to a dict.
        d["header"] = d["header"].to_dict()
        return HC.from_dict(d)

class Header():
    @classmethod
    def from_dict(cls, d) -> dict:
        #convert parts of header from class to dictionary so that HeaderBase.from_dict() still works (for some reason ladybug hasn't used a JSONDecoder for json decoding...)
        #this works because the python json decoder works depth first.
        d["data_type"] = d["data_type"].to_dict()
        d["analysis_period"] = d["analysis_period"].to_dict()
        return HeaderBase.from_dict(d)

_TYPES: list[type] = [
    EnergyMaterial,
    EnergyMaterialVegetation,
    AnalysisPeriod,
    HourlyContinuousCollection,
    Location,
    DataType,
    Header,
    Point
]

class LBTBHoMJSONDecoder(BHoMJSONDecoder):
    def deserialise_unknown(self, obj:BHoMObject | IObject | dict):
        """custom object-hook method for BHoMJSONDecoder"""
        if isinstance(obj, BHoMObject) or isinstance(obj, IObject):
            _type = obj._t.split(".")[-1]

            klass = [t for t in _TYPES if t.__name__ == _type]

            if len(klass) == 1:
                setattr(obj, "type", _type)
                return klass[0].from_dict(obj.to_dict())
    
        #default to returning a bhom object if tha above did not work
        return obj
