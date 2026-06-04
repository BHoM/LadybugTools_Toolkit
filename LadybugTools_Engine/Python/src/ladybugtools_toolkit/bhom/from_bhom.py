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
        return d #due to Header.from_dict() not handling already deserialised objects, this should just return the correct dictionary instead of the DataType object.

class AnalysisPeriod():
    @classmethod
    def from_dict(cls, d) -> dict:
        d["st_hour"] = d["start_hour"]
        d["st_day"] = d["start_day"]
        d["st_month"] = d["start_month"]
        return d

class HourlyContinuousCollection():
    @classmethod
    def from_dict(cls, d) -> dict:
        d["type"] = "HourlyContinuous"
        return HC.from_dict(d)

class Header():
    @classmethod
    def from_dict(cls, d) -> dict:
        return d

_TYPES: list[type] = [EnergyMaterial, EnergyMaterialVegetation, AnalysisPeriod, HourlyContinuousCollection, Location, DataType, Header, Point]

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
