from dataclasses import dataclass
from typing import Any

import numpy as np
from ladybug.datatype import TYPESDICT
from ladybug.datatype.angle import Angle, WindDirection
from ladybug.datatype.area import Area
from ladybug.datatype.base import DataTypeBase
from ladybug.datatype.current import Current
from ladybug.datatype.distance import (
    CeilingHeight,
    Distance,
    LiquidPrecipitationDepth,
    PrecipitableWater,
    SnowDepth,
    Visibility,
)
from ladybug.datatype.energy import Energy
from ladybug.datatype.energyflux import (
    DiffuseHorizontalIrradiance,
    DirectHorizontalIrradiance,
    DirectNormalIrradiance,
    EffectiveRadiantField,
    EnergyFlux,
    GlobalHorizontalIrradiance,
    HorizontalInfraredRadiationIntensity,
    Irradiance,
    MetabolicRate,
)
from ladybug.datatype.energyintensity import (
    DiffuseHorizontalRadiation,
    DirectHorizontalRadiation,
    DirectNormalRadiation,
    EnergyIntensity,
    ExtraterrestrialDirectNormalRadiation,
    ExtraterrestrialHorizontalRadiation,
    GlobalHorizontalRadiation,
    Radiation,
)
from ladybug.datatype.fraction import (
    AerosolOpticalDepth,
    Albedo,
    Fraction,
    HumidityRatio,
    LiquidPrecipitationQuantity,
    OpaqueSkyCover,
    PercentagePeopleDissatisfied,
    RelativeHumidity,
    TotalSkyCover,
)
from ladybug.datatype.illuminance import (
    DiffuseHorizontalIlluminance,
    DirectNormalIlluminance,
    GlobalHorizontalIlluminance,
    Illuminance,
)
from ladybug.datatype.luminance import Luminance, ZenithLuminance
from ladybug.datatype.mass import Mass
from ladybug.datatype.massflowrate import MassFlowRate
from ladybug.datatype.power import ActivityLevel, Power
from ladybug.datatype.pressure import AtmosphericStationPressure, Pressure
from ladybug.datatype.rvalue import ClothingInsulation, RValue
from ladybug.datatype.specificenergy import Enthalpy, SpecificEnergy
from ladybug.datatype.speed import AirSpeed, Speed, WindSpeed
from ladybug.datatype.temperature import (
    AirTemperature,
    ClothingTemperature,
    CoreBodyTemperature,
    DewPointTemperature,
    DryBulbTemperature,
    GroundTemperature,
    HeatIndexTemperature,
    MeanRadiantTemperature,
    NeutralTemperature,
    OperativeTemperature,
    PhysiologicalEquivalentTemperature,
    PrevailingOutdoorTemperature,
    RadiantTemperature,
    SkinTemperature,
    SkyTemperature,
    StandardEffectiveTemperature,
    Temperature,
    UniversalThermalClimateIndex,
    WetBulbGlobeTemperature,
    WetBulbTemperature,
    WindChillTemperature,
)
from ladybug.datatype.temperaturedelta import (
    AirTemperatureDelta,
    OperativeTemperatureDelta,
    RadiantTemperatureDelta,
    TemperatureDelta,
)
from ladybug.datatype.temperaturetime import CoolingDegreeTime, HeatingDegreeTime, TemperatureTime
from ladybug.datatype.thermalcondition import (
    CoreTemperatureCategory,
    DiscomfortReason,
    PredictedMeanVote,
    ThermalComfort,
    ThermalCondition,
    ThermalConditionElevenPoint,
    ThermalConditionFivePoint,
    ThermalConditionNinePoint,
    ThermalConditionSevenPoint,
    UTCICategory,
)
from ladybug.datatype.time import Time
from ladybug.datatype.uvalue import ConvectionCoefficient, RadiantCoefficient, UValue
from ladybug.datatype.voltage import Voltage
from ladybug.datatype.volume import Volume
from ladybug.datatype.volumeflowrate import VolumeFlowRate
from ladybug.datatype.volumeflowrateintensity import VolumeFlowRateIntensity
from matplotlib.colors import Colormap, to_hex, to_rgba
from pydantic import BaseModel


def color_to_format(
    color: Any, return_type: str, include_alpha: bool = False
) -> str | list[int] | list[float]:
    """Convert a color-like object to a target format.

    Args:
        color (Any):
            The color-like object to convert.
        return_type (str):
            The type of color to return. Options are:
             - hex
             - rgb_int
             - rgb_float
             - plotly
        include_alpha (bool, optional):
            Include the alpha channel in the output. Defaults to False.

    Returns:
        Any:
            The color in the target format.
    """

    # convert color to hex
    c_rgba = to_rgba(color)

    match return_type:
        case "hex":
            temp = to_hex(c_rgba, keep_alpha=True)
            if include_alpha:
                return temp
            return temp[:-2]
        case "rgb_int":
            temp = (np.array(c_rgba) * 255).round(0).astype(int)
            if include_alpha:
                return temp
            return temp[:-1]
        case "rgb_float":
            temp = np.array([float(i) for i in c_rgba])
            if include_alpha:
                return temp
            return temp[:-1]
        case "plotly":
            r, g, b = (np.array(c_rgba)[:3] * 255).round(0).astype(int)
            if include_alpha:
                a = float(c_rgba[3])
                return f"rgba({r},{g},{b},{a})"
            return f"rgb({r},{g},{b})"
        case _:
            raise ValueError("return_type must be either 'hex', 'rgb_int', 'rgb_float or 'plotly'.")

def datatype_to_string(datatype: DataTypeBase, unit: str) -> str:
    """Convert a ladybug datatype to a string representation.

    Args:
        datatype (DataTypeBase):
            The ladybug datatype to convert.
        unit (str):
            The unit of the datatype.

    Returns:
        str:
            The string representation of the datatype.
    """

    return f"{datatype} ({unit})"

class ColorConfig(BaseModel):
    """A configuration class for color settings."""

    identifier: str = "Default"

    class Temperature(BaseModel):
        



# def _create_case_options(datatype: DataTypeBase) -> tuple:
#     """Return the default unit intervals for the given datatype.

#     Args:
#         datatype: A ladybug datatype class.

#     Returns:
#         tuple: A tuple of options that can be used to infer the datatype.

#     Note:
#         For example, DryBulbTemperature datatype can be inferrred from the following - based on
#         varying representations of the given LB datatype:
#             - ladybug.datatype.temperature.DryBulbTemperature
#             - 'DBT'
#             - 'Dry Bulb Temperature (C)'
#             - 'DBT (C)'
#             - 'Dry Bulb Temperature (F)'
#             - 'DBT (F)'
#             - 'Dry Bulb Temperature (K)'
#             - 'DBT (K)'
#             - 'dry_bulb_temperature',
#     """

#     if not isinstance(datatype, DataTypeBase):
#         raise ValueError(
#             f"'{datatype}' is not known datatype. Has it been called, or do you need to define the properties associated with this datatype?"
#         )

#     options = [datatype, datatype.abbreviation]
#     for unit in datatype.units:
#         options.append(f"{datatype} ({unit})")
#         options.append(f"{datatype.abbreviation} ({unit})")
#     options.append(str(datatype).lower().replace(" ", "_"))

#     return options


# def get_default_intervals(datatype: DataTypeBase | str) -> tuple:
#     """Return the default unit intervals for the given datatype. For example, Dry Bulb Temperature"""

#     match datatype:
#         case _ if datatype in _create_case_options(DryBulbTemperature()):
#             intervals = np.arange(-20, 50, 15)
#         case _:
#             raise ValueError(f"No default intervals exist for '{datatype}'.")

#     return intervals


# def get_default_colormap(datatype: DataTypeBase) -> Colormap:
#     match datatype:
#         case _ if datatype in _create_case_options(DryBulbTemperature()):
#             colormap = "Reds"
#         case _:
#             raise ValueError(f"No default colormap exists for '{datatype}'")

#     return colormap


# def get_default_color(datatype: DataTypeBase | str, fmt: str = "hex") -> str:
#     """Return the associated colormap for the given ladybug datatype."""
#     match datatype:
#         case _ if datatype in _create_case_options(DryBulbTemperature()):
#             color = color_to_format("red", return_type=fmt, include_alpha=True)
#         case _:
#             raise ValueError(f"No default color exists for '{datatype}'")

#     return color


# if __name__ == "__main__":
#     for k, v in TYPESDICT.items():
#         print(_create_case_options(v()))
#         try:
#             get_default_intervals(v())
#         except ValueError as e:
#             print(k, e)
#         try:
#             get_default_color(v())
#         except ValueError as e:
#             print(k, e)
#         try:
#             get_default_colormap(v())
#         except ValueError as e:
#             print(k, e)
#         print()
