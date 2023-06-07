import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from caseconverter import pascalcase, snakecase
from honeybee._base import _Base as HB_Base
from honeybee_energy.material._base import _EnergyMaterialBase
from ladybug._datacollectionbase import BaseCollection
from ladybug.epw import EPW, AnalysisPeriod, Header, HourlyContinuousCollection
from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D


def inf_dtype_to_inf_str(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Convert any values in a dict that are "Infinite" or "-Infinite"
    into a string equivalent."""

    def to_inf_str(v: Any):
        if isinstance(v, dict):
            return inf_dtype_to_inf_str(v)
        if isinstance(v, (list, tuple, np.ndarray)):
            return [to_inf_str(item) for item in v]
        if v in (math.inf, -math.inf):
            return json.dumps(v)
        return v

    return {k: to_inf_str(v) for k, v in dictionary.items()}


def inf_str_to_inf_dtype(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Convert any values in a dict that are "inf" or "-inf" into a numeric
    equivalent."""
    if not isinstance(dictionary, dict):
        return dictionary

    def to_inf_dtype(v: Any):
        if isinstance(v, dict):
            return inf_str_to_inf_dtype(v)
        if isinstance(v, (list, tuple, np.ndarray)):
            return [to_inf_dtype(item) for item in v]
        if v in ("inf", "-inf"):
            return json.loads(v)
        return v

    return {k: to_inf_dtype(v) for k, v in dictionary.items()}


def keys_to_pascalcase(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all keys in a dictionary into pascalcase."""
    return {
        pascalcase(k)
        if k != "_t"
        else k: keys_to_pascalcase(v)
        if isinstance(v, dict)
        else v
        for k, v in dictionary.items()
    }


def keys_to_snakecase(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all keys in a dictionary into snakecase."""
    return {
        snakecase(k)
        if k != "_t"
        else k: keys_to_snakecase(v)
        if isinstance(v, dict)
        else v
        for k, v in dictionary.items()
    }


def fix_bhom_jsondict(dictionary: Dict[str, Any]) -> List[Any]:
    """Convert any values in a dict that in the form x: {"_t": "...", "_v": [...]} into x: [...]"""
    if not isinstance(dictionary, dict):
        return dictionary
    if dictionary.keys() == {"_t", "_v"}:
        dictionary = dictionary["_v"]
        return dictionary
    if dictionary.keys() == {"BHoM_Guid"}:
        dictionary = None
        return dictionary
    return dictionary


class BHoMEncoder(json.JSONEncoder):
    """A custom encoder for converting objects from this toolkit into serialisable JSON with _t flag where necessary for automatic conversion BHoM-side.

    Args:
        o (_type_):
            The object for conversion.
    """

    # pylint : disable=super-with-arguments;too-many-return-statements
    def default(self, dct):
        """_"""
        # Path encoding
        if isinstance(dct, Path):
            return dct.as_posix()

        # NumPy encoding
        if isinstance(dct, (np.number, np.inexact, np.floating, np.complexfloating)):
            return float(dct)
        if isinstance(dct, (np.integer, np.signedinteger, np.unsignedinteger)):
            return int(dct)
        if isinstance(dct, (np.character)):
            return str(dct)
        if isinstance(dct, np.ndarray):
            return dct.tolist()

        # Pandas encoding
        if isinstance(dct, (pd.DataFrame, pd.Series, pd.DatetimeIndex)):
            return dct.to_dict()

        # Ladybug encoding
        if isinstance(dct, EPW):
            return dct.to_dict()
        if isinstance(dct, BaseCollection):
            return inf_dtype_to_inf_str(dct.to_dict())

        # Honeybee encoding
        if isinstance(dct, HB_Base):
            return dct.to_dict()
        if isinstance(dct, _EnergyMaterialBase):
            return dct.to_dict()

        # Ladybug geometry objects
        if isinstance(dct, Point3D):
            return {"_t": "BH.oM.Geometry.Point", "X": dct.x, "Y": dct.y, "Z": dct.z}
        if isinstance(dct, Vector3D):
            return {"_t": "BH.oM.Geometry.Vector", "X": dct.x, "Y": dct.y, "Z": dct.z}
        # TODO - Add more conversions for other geometry types into BHoM geometry types

        # # LBTools_toolkit specific objects
        # if isinstance(dct, OpaqueMaterial):
        #     return {
        #         "_t": "BH.oM.LadybugTools.OpaqueMaterial",
        #         "Identifier": dct.Identifier,
        #         "Source": dct.Source,
        #         "Thickness": dct.Thickness,
        #         "Conductivity": dct.Conductivity,
        #         "Density": dct.Density,
        #         "SpecificHeat": dct.SpecificHeat,
        #         "Roughness": dct.Roughness,
        #         "SolarAbsorptance": dct.SolarAbsorptance,
        #         "VisibleAbsorptance": dct.VisibleAbsorptance,
        #     }
        # if isinstance(dct, OpaqueVegetationMaterial):
        #     return {
        #         "_t": "BH.oM.LadybugTools.OpaqueVegetationMaterial",
        #         "Identifier": dct.Identifier,
        #         "Source": dct.Source,
        #         "Thickness": dct.Thickness,
        #         "Conductivity": dct.Conductivity,
        #         "Density": dct.Density,
        #         "SpecificHeat": dct.SpecificHeat,
        #         "Roughness": dct.Roughness,
        #         "SoilThermalAbsorptance": dct.SoilThermalAbsorptance,
        #         "SoilSolarAbsorptance": dct.SoilSolarAbsorptance,
        #         "SoilVisibleAbsorptance": dct.SoilVisibleAbsorptance,
        #         "PlantHeight": dct.PlantHeight,
        #         "LeafAreaIndex": dct.LeafAreaIndex,
        #         "LeafReflectivity": dct.LeafReflectivity,
        #         "LeafEmissivity": dct.LeafEmissivity,
        #         "MinStomatalResist": dct.MinStomatalResist,
        #     }

        # Catch-all for any other object that has a "to_dict" method
        try:
            return dct.to_dict()
        except AttributeError:
            try:
                return str(dct)
            except Exception:  # pylint: disable=broad-except
                pass

        return super(BHoMEncoder, self).default(dct)

    # pylint : enable=super-with-arguments;too-many-return-statements


# class BHoMDecoder(json.JSONDecoder):
#     """A custom encoder for converting objects from BHoM into native objects within this Python package.

#     Args:
#         o (_type_):
#             The object for conversion.
#     """

#     def default(self, dct):
#         """_"""

#         # handle objectable dicts
#         if isinstance(dct, dict):
#             if "_t" in dct.keys():
#                 # This is a BHoM-style object, and might be convertible into a native object
#                 if dct["_t"] == "BH.oM.Geometry.Point":
#                     return Point3D(dct["X"], dct["Y"], dct["Z"])
#                 if dct["_t"] == "BH.oM.Geometry.Vector":
#                     return Vector3D(dct["X"], dct["Y"], dct["Z"])
#                 # if dct["_t"] == "BH.oM.LadybugTools.OpaqueMaterial":
#                 #     return OpaqueMaterial(
#                 #         Identifier=dct["Identifier"],
#                 #         Source=dct["Source"],
#                 #         Thickness=dct["Thickness"],
#                 #         Conductivity=dct["Conductivity"],
#                 #         Density=dct["Density"],
#                 #         SpecificHeat=dct["SpecificHeat"],
#                 #         Roughness=dct["Roughness"],
#                 #         SolarAbsorptance=dct["SolarAbsorptance"],
#                 #         VisibleAbsorptance=dct["VisibleAbsorptance"],
#                 #     )
#                 # if dct["_t"] == "BH.oM.LadybugTools.OpaqueVegetationMaterial":
#                 #     return OpaqueVegetationMaterial(
#                 #         Identifier=dct["Identifier"],
#                 #         Source=dct["Source"],
#                 #         Thickness=dct["Thickness"],
#                 #         Conductivity=dct["Conductivity"],
#                 #         Density=dct["Density"],
#                 #         SpecificHeat=dct["SpecificHeat"],
#                 #         Roughness=dct["Roughness"],
#                 #         SoilThermalAbsorptance=dct["SoilThermalAbsorptance"],
#                 #         SoilSolarAbsorptance=dct["SoilSolarAbsorptance"],
#                 #         SoilVisibleAbsorptance=dct["SoilVisibleAbsorptance"],
#                 #         PlantHeight=dct["PlantHeight"],
#                 #         LeafAreaIndex=dct["LeafAreaIndex"],
#                 #         LeafReflectivity=dct["LeafReflectivity"],
#                 #         LeafEmissivity=dct["LeafEmissivity"],
#                 #         MinStomatalResist=dct["MinStomatalResist"],
#                 #     )

#             if "type" in dct.keys():
#                 # This is a Ladybug-style object, and might be convertible into a native object
#                 if dct["type"] == "Header":
#                     return Header.from_dict(dct)
#                 if dct["type"] == "AnalysisPeriod":
#                     return AnalysisPeriod.from_dict(dct)
#                 if dct["type"] == "HourlyContinuous":
#                     return HourlyContinuousCollection.from_dict(dct)

#         return super(BHoMDecoder, self).default(dct)


# """
# """Encoding methods enabling data to be passed from Python-style LB/HB objects to BHoM objects and vice versa.
# """

# # pylint : disable=super-with-arguments;too-many-return-statements

# import json
# import math
# from datetime import datetime
# from pathlib import Path

# import numpy as np
# import pandas as pd
# from honeybee._base import _Base as HB_Base
# from honeybee_energy.material._base import _EnergyMaterialBase
# from ladybug._datacollectionbase import BaseCollection
# from ladybug.epw import EPW, AnalysisPeriod, Header, HourlyContinuousCollection
# from ladybug_geometry.geometry2d import LineSegment2D, Point2D, Polyline2D, Vector2D
# from ladybug_geometry.geometry3d import (
#     Face3D,
#     LineSegment3D,
#     Plane,
#     Point3D,
#     Polyline3D,
#     Vector3D,
# )

# from .helpers import fix_bhom_jsondict, inf_dtype_to_inf_str, inf_str_to_inf_dtype


# class CompositeJsonEncoder:
#     """
#     Combine multiple JSON encoders
#     """

#     def __init__(self, *encoders):
#         self.encoders = encoders
#         self.args = ()
#         self.kwargs = {}

#     def default(self, obj):
#         """_"""
#         for encoder in self.encoders:
#             try:
#                 return encoder(*self.args, **self.kwargs).default(obj)
#             except TypeError:
#                 pass
#         raise TypeError(
#             f"Object of type {obj.__class__.__name__} is not JSON serializable"
#         )

#     def __call__(self, *args, **kwargs):
#         self.args = args
#         self.kwargs = kwargs
#         enc = json.JSONEncoder(*args, **kwargs)
#         enc.default = self.default
#         return enc


# class CompositeJsonDecoder:
#     """
#     Combine multiple JSON decoders
#     """

#     def __init__(self, *decoders):
#         self.decoders = decoders
#         self.args = ()
#         self.kwargs = {}

#         raise NotImplementedError()

#     def object_hook(self, obj):
#         """_"""
#         for decoder in self.decoders:
#             try:
#                 return decoder(*self.args, **self.kwargs).object_hook(obj)
#             except TypeError:
#                 pass
#         return obj

#     def __call__(self, *args, **kwargs):
#         self.args = args
#         self.kwargs = kwargs
#         dec = json.JSONDecoder(*args, **kwargs)
#         dec.object_hook = self.object_hook
#         return dec


# class PathEncoder(json.JSONEncoder):
#     """A custom encoder for converting pathlib.Path objects into serialisable JSON."""

#     def default(self, o):
#         """_"""
#         if isinstance(o, Path):
#             return o.resolve().as_posix()
#         return super().default(o)  # pylint: disable=E1101


# class PathDecoder(json.JSONDecoder):
#     """A custom decoder for converting path-like strings into a Path object."""

#     def __init__(self, *args, **kwargs):
#         json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

#     def object_hook(self, obj):
#         return obj


# class DateEncoder(json.JSONEncoder):
#     """A custom encoder for converting datetime objects into serialisable JSON."""

#     def default(self, o):
#         """_"""
#         if isinstance(o, datetime):
#             return o.isoformat()
#         return super().default(o)  # pylint: disable=E1101


# class DateDecoder(json.JSONDecoder):
#     """A custom decoder for converting datetime-like strings into a datetime object."""

#     def __init__(self, *args, **kwargs):
#         json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

#     def object_hook(self, obj):
#         return datetime.fromisoformat(obj)


# class NumpyEncoder(json.JSONEncoder):
#     """A custom encoder for converting numpy objects into serialisable JSON."""

#     def default(self, o):
#         """_"""
#         if isinstance(o, (np.number, np.inexact, np.floating, np.complexfloating)):
#             return float(o)
#         if isinstance(o, (np.integer, np.signedinteger, np.unsignedinteger)):
#             return int(o)
#         if isinstance(o, (np.character)):
#             return str(o)
#         if isinstance(o, np.ndarray):
#             return o.tolist()
#         return super().default(o)  # pylint: disable=E1101


# class NumpyDecoder(json.JSONDecoder):
#     """A custom decoder for converting numpy-like objects into a numpy object."""

#     def __init__(self, *args, **kwargs):
#         json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

#     def object_hook(self, obj):
#         return obj


# class InfinityEncoder(json.JSONEncoder):
#     """A custom encoder for converting infinity objects into serialisable JSON."""

#     def default(self, o):
#         """_"""
#         if o == math.inf:
#             return "Infinity"
#         if o == -math.inf:
#             return "-Infinity"
#         return super().default(o)  # pylint: disable=E1101


# class InfinityDecoder(json.JSONDecoder):
#     """A custom decoder for converting infinity-like strings into a infinity object."""

#     def __init__(self, *args, **kwargs):
#         json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

#     def object_hook(self, obj):
#         """_"""
#         if obj == "Infinity":
#             return math.inf
#         if obj == "-Infinity":
#             return -math.inf
#         return obj


# class BHoMListEncoder(json.JSONEncoder):
#     """A custom encoder for converting BHoM objects into serialisable JSON."""

#     def default(self, o):
#         """_"""
#         return super().default(o)  # pylint: disable=E1101


# class BHOMListDecoder(json.JSONDecoder):
#     """A custom decoder for converting BHoM-like list objects into a proper Python list object."""

#     def __init__(self, *args, **kwargs):
#         json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

#     def object_hook(self, obj):
#         """_"""
#         if isinstance(obj, dict):
#             if obj.keys() == {"_t", "_v"}:
#                 return obj["_v"]
#         return obj


# class PandasEncoder(json.JSONEncoder):
#     """A custom encoder for converting pandas objects into serialisable JSON."""

#     def default(self, o):
#         """_"""
#         if isinstance(o, pd.DataFrame):
#             return {**o.to_dict(), **{"pytype": "pandas.DataFrame"}}
#         if isinstance(o, pd.Series):
#             return {**o.to_dict(), **{"pytype": "pandas.Series"}}
#         return super().default(o)  # pylint: disable=E1101


# class PandasDecoder(json.JSONDecoder):
#     """A custom decoder for converting pandas-like objects into a pandas object."""

#     def __init__(self, *args, **kwargs):
#         json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

#     def object_hook(self, obj):
#         if "pytype" in obj.keys() and obj["pytype"].startswith("pandas."):
#             obj.pop("type")
#             if obj["type"] == "pandas.DataFrame":
#                 return pd.DataFrame.from_dict(obj)
#             if obj["type"] == "pandas.Series":
#                 return pd.Series.from_dict(obj)
#         return obj


# class LadybugEncoder(json.JSONEncoder):
#     """A custom encoder for converting ladybug datacollection-like objects into serialisable JSON."""

#     def default(self, o):
#         """_"""
#         if isinstance(o, EPW):
#             return o.to_dict()
#         if isinstance(o, BaseCollection):
#             return inf_dtype_to_inf_str(o.to_dict())
#         if isinstance(o, Header):
#             return o.to_dict()
#         if isinstance(o, AnalysisPeriod):
#             return o.to_dict()
#         return super().default(o)  # pylint: disable=E1101


# class LadybugDecoder(json.JSONDecoder):
#     """A custom decoder for converting ladybug-like objects into a ladybug object."""

#     def __init__(self, *args, **kwargs):
#         json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

#     def object_hook(self, obj):
#         if "type" in obj.keys():
#             if obj["type"] == "EPW":
#                 # fix any inf values
#                 d = inf_str_to_inf_dtype(obj)
#                 return EPW.from_dict(d)
#             if obj["type"] == "Header":
#                 return Header.from_dict(obj)
#             if obj["type"] == "AnalysisPeriod":
#                 return AnalysisPeriod.from_dict(obj)
#             if obj["type"] == "HourlyContinuous":
#                 # fix any inf values
#                 d = inf_str_to_inf_dtype(obj)
#                 return HourlyContinuousCollection.from_dict(d)
#         return obj


# class LadybugGeometryEncoder(json.JSONEncoder):
#     """A custom encoder for converting ladybug datacollection-like objects into serialisable JSON for BHoM."""

#     def default(self, o):
#         """_"""
#         if isinstance(o, LineSegment2D):
#             return {
#                 "_t": "BH.oM.Geometry.Line",
#                 "Start": o.p1,
#                 "End": o.p2,
#             }
#         if isinstance(o, LineSegment3D):
#             return {
#                 "_t": "BH.oM.Geometry.Line",
#                 "Start": o.p1,
#                 "End": o.p2,
#             }
#         if isinstance(o, Point2D):
#             return {"_t": "BH.oM.Geometry.Point", "X": o.x, "Y": o.y, "Z": 0}
#         if isinstance(o, Polyline2D):
#             return {
#                 "_t": "BH.oM.Geometry.Polyline",
#                 "ControlPoints": o.vertices,
#             }
#         if isinstance(o, Vector2D):
#             return {"_t": "BH.oM.Geometry.Vector", "X": o.x, "Y": o.y}
#         if isinstance(o, Point3D):
#             return {"_t": "BH.oM.Geometry.Point", "X": o.x, "Y": o.y, "Z": o.z}
#         if isinstance(o, Vector3D):
#             return {"_t": "BH.oM.Geometry.Vector", "X": o.x, "Y": o.y, "Z": o.z}
#         if isinstance(o, Plane):
#             return {"_t": "BH.oM.Geometry.Plane", "Origin": o.o, "Normal": o.n}
#         if isinstance(o, Polyline3D):
#             return {
#                 "_t": "BH.oM.Geometry.Polyline",
#                 "ControlPoints": o.vertices,
#             }
#         if isinstance(o, Face3D):
#             _d = {
#                 "_t": "BH.oM.Geometry.PlanarSurface",
#                 "ExternalBoundary": Polyline3D.from_array(
#                     list(o.vertices) + [o.vertices[0]]
#                 ),
#                 "InternalBoundaries": [],
#             }
#             if o.holes:
#                 for hole in o.holes:
#                     _d["InternalBoundaries"].append(
#                         Polyline3D.from_array(list(hole) + [hole[0]])
#                     )
#             return _d


# class LadybugGeometryDecoder(json.JSONDecoder):
#     """A custom decoder for converting ladybug-like objects into a ladybug object."""

#     def __init__(self, *args, **kwargs):
#         json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

#     def object_hook(self, obj):
#         if "_t" in obj.keys():
#             if obj["_t"] == "BH.oM.Geometry.Point":
#                 return Point3D.from_array([obj["X"], obj["Y"], obj["Z"]])
#             if obj["_t"] == "BH.oM.Geometry.Vector":
#                 return Vector3D.from_array([obj["X"], obj["Y"], obj["Z"]])
#             if obj["_t"] == "BH.oM.Geometry.Plane":
#                 return Plane(o=obj["Origin"], n=obj["Normal"])
#             if obj["_t"] == "BH.oM.Geometry.Line":
#                 return LineSegment3D.from_end_points(
#                     p1=obj["Start"],
#                     p2=obj["End"],
#                 )
#             if obj["_t"] == "BH.oM.Geometry.Polyline":
#                 return Polyline3D(vertices=obj["ControlPoints"])
#             if obj["_t"] == "BH.oM.Geometry.PlanarSurface":
#                 holes = []
#                 for hole in obj["InternalBoundaries"]:
#                     holes.append(hole["ControlPoints"])
#                 return Face3D(boundary=obj["ExternalBoundary"], holes=holes)
#         return obj


# BHoMEncoder = CompositeJsonEncoder(
#     json.JSONEncoder,
#     PathEncoder,
#     DateEncoder,
#     NumpyEncoder,
#     PandasEncoder,
#     LadybugEncoder,
#     LadybugGeometryEncoder,
#     InfinityEncoder,
# )

## helpers
# import json
# import math
# import re
# from typing import Any, Dict, List

# import numpy as np
# from caseconverter import pascalcase, snakecase


# def is_decorated(func: callable):
#     """Determine whether a callable function is already decorated."""
#     return hasattr(func, "__wrapped__") or func.__name__ not in globals()


# def is_pascal_case(s: str) -> bool:
#     """Check whether a string is compliant with PascalCase naming conventions."""
#     if len(s) == 1:
#         return s.isupper()
#     return (
#         True
#         if re.match("([A-Z][a-z0-9]+)((\d)|([A-Z0-9][a-z0-9]+))*([A-Z])?", s)
#         and "_" not in s
#         else False
#     )


# def inf_dtype_to_inf_str(dictionary: Dict[str, Any]) -> Dict[str, Any]:
#     """Convert any values in a dict that are "Infinite" or "-Infinite"
#     into a string equivalent."""

#     def to_inf_str(v: Any):
#         if isinstance(v, dict):
#             return inf_dtype_to_inf_str(v)
#         if isinstance(v, (list, tuple, np.ndarray)):
#             return [to_inf_str(item) for item in v]
#         if v in (math.inf, -math.inf):
#             return json.dumps(v)
#         return v

#     return {k: to_inf_str(v) for k, v in dictionary.items()}


# def inf_str_to_inf_dtype(dictionary: Dict[str, Any]) -> Dict[str, Any]:
#     """Convert any values in a dict that are "inf" or "-inf" into a numeric
#     equivalent."""
#     if not isinstance(dictionary, dict):
#         return dictionary

#     def to_inf_dtype(v: Any):
#         if isinstance(v, dict):
#             return inf_str_to_inf_dtype(v)
#         if isinstance(v, (list, tuple, np.ndarray)):
#             return [to_inf_dtype(item) for item in v]
#         if v in ("inf", "-inf"):
#             return json.loads(v)
#         return v

#     return {k: to_inf_dtype(v) for k, v in dictionary.items()}


# def keys_to_pascalcase(dictionary: Dict[str, Any]) -> Dict[str, Any]:
#     """Convert all keys in a dictionary into pascalcase."""
#     return {
#         pascalcase(k)
#         if k != "_t"
#         else k: keys_to_pascalcase(v)
#         if isinstance(v, dict)
#         else v
#         for k, v in dictionary.items()
#     }


# def keys_to_snakecase(dictionary: Dict[str, Any]) -> Dict[str, Any]:
#     """Convert all keys in a dictionary into snakecase."""
#     return {
#         snakecase(k)
#         if k != "_t"
#         else k: keys_to_snakecase(v)
#         if isinstance(v, dict)
#         else v
#         for k, v in dictionary.items()
#     }


# def fix_bhom_jsondict(dictionary: Dict[str, Any]) -> List[Any]:
#     """Convert any values in a dict in the form x: {"_t": "...", "_v": [...]} into x: [...]"""
#     if not isinstance(dictionary, dict):
#         return dictionary
#     if dictionary.keys() == {"_t", "_v"}:
#         dictionary = dictionary["_v"]
#         return dictionary
#     if dictionary.keys() == {"BHoM_Guid"}:
#         dictionary = None
#         return dictionary
#     return dictionary
