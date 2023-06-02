import json
import math
from pathlib import Path
from typing import Any, Dict

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


class BHoMDecoder(json.JSONDecoder):
    """A custom encoder for converting objects from BHoM into native objects within this Python package.

    Args:
        o (_type_):
            The object for conversion.
    """

    def default(self, dct):
        """_"""

        # handle objectable dicts
        if isinstance(dct, dict):
            if "_t" in dct.keys():
                # This is a BHoM-style object, and might be convertible into a native object
                if dct["_t"] == "BH.oM.Geometry.Point":
                    return Point3D(dct["X"], dct["Y"], dct["Z"])
                if dct["_t"] == "BH.oM.Geometry.Vector":
                    return Vector3D(dct["X"], dct["Y"], dct["Z"])
                # if dct["_t"] == "BH.oM.LadybugTools.OpaqueMaterial":
                #     return OpaqueMaterial(
                #         Identifier=dct["Identifier"],
                #         Source=dct["Source"],
                #         Thickness=dct["Thickness"],
                #         Conductivity=dct["Conductivity"],
                #         Density=dct["Density"],
                #         SpecificHeat=dct["SpecificHeat"],
                #         Roughness=dct["Roughness"],
                #         SolarAbsorptance=dct["SolarAbsorptance"],
                #         VisibleAbsorptance=dct["VisibleAbsorptance"],
                #     )
                # if dct["_t"] == "BH.oM.LadybugTools.OpaqueVegetationMaterial":
                #     return OpaqueVegetationMaterial(
                #         Identifier=dct["Identifier"],
                #         Source=dct["Source"],
                #         Thickness=dct["Thickness"],
                #         Conductivity=dct["Conductivity"],
                #         Density=dct["Density"],
                #         SpecificHeat=dct["SpecificHeat"],
                #         Roughness=dct["Roughness"],
                #         SoilThermalAbsorptance=dct["SoilThermalAbsorptance"],
                #         SoilSolarAbsorptance=dct["SoilSolarAbsorptance"],
                #         SoilVisibleAbsorptance=dct["SoilVisibleAbsorptance"],
                #         PlantHeight=dct["PlantHeight"],
                #         LeafAreaIndex=dct["LeafAreaIndex"],
                #         LeafReflectivity=dct["LeafReflectivity"],
                #         LeafEmissivity=dct["LeafEmissivity"],
                #         MinStomatalResist=dct["MinStomatalResist"],
                #     )

            if "type" in dct.keys():
                # This is a Ladybug-style object, and might be convertible into a native object
                if dct["type"] == "Header":
                    return Header.from_dict(dct)
                if dct["type"] == "AnalysisPeriod":
                    return AnalysisPeriod.from_dict(dct)
                if dct["type"] == "HourlyContinuous":
                    return HourlyContinuousCollection.from_dict(dct)

        return super(BHoMDecoder, self).default(dct)
