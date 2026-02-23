"""
Methods for converting various objects, data-types and names into appropriate 
colours for use in visualisations.
"""

import difflib
import re
from typing import Any, Optional

from ladybug.color import Color
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
from ladybug.datatype.generic import GenericType
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
from ladybug.datatype.temperaturetime import (
    CoolingDegreeTime,
    HeatingDegreeTime,
    TemperatureTime,
)
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
from matplotlib.colors import to_hex, to_rgba

from ..ladybug_extension.datatype import (
    ActualSensationVote,
    ApparentTemperature,
    AreaPerPerson,
    DailyLightIntegral,
    DiscomfortIndex,
    Humidex,
    NeutralTemperature,
    People,
    PeoplePerArea,
    RadiationBenefit,
    Season,
    ShadeBenefit,
    SpecificHeatCapacity,
    ThermalSensation,
    Walkability,
    WindBenefit,
)
from ..bhom.logger import CONSOLE_LOGGER

DEFAULT_COLOURS = {
    # Ladybug datatypes
    Angle: "#585253FF",
    WindDirection: "#585253FF",
    Area: "#585253FF",
    Current: "#EB671CFF",
    CeilingHeight: "#585253FF",
    Distance: "#585253FF",
    LiquidPrecipitationDepth: "#8DB9CAFF",
    PrecipitableWater: "#8DB9CAFF",
    SnowDepth: "#E3E0D6FF",
    Visibility: "#958B82FF",
    Energy: "#00A499FF",
    DiffuseHorizontalIrradiance: "#D06A13FF",
    DirectHorizontalIrradiance: "#D06A13FF",
    DirectNormalIrradiance: "#D06A13FF",
    EffectiveRadiantField: "#D06A13FF",
    EnergyFlux: "#D06A13FF",
    GlobalHorizontalIrradiance: "#D06A13FF",
    HorizontalInfraredRadiationIntensity: "#D06A13FF",
    Irradiance: "#D06A13FF",
    MetabolicRate: "#D06A13FF",
    SpecificHeatCapacity: "#FFAF6AFF",
    DiffuseHorizontalRadiation: "#D06A13FF",
    DirectHorizontalRadiation: "#D06A13FF",
    DirectNormalRadiation: "#D06A13FF",
    EnergyIntensity: "#D06A13FF",
    ExtraterrestrialDirectNormalRadiation: "#D06A13FF",
    ExtraterrestrialHorizontalRadiation: "#D06A13FF",
    GlobalHorizontalRadiation: "#D06A13FF",
    Radiation: "#D06A13FF",
    AerosolOpticalDepth: "#B62B77FF",
    Albedo: "#B62B77FF",
    Fraction: "#B62B77FF",
    HumidityRatio: "#24135FFF",
    LiquidPrecipitationQuantity: "#006DA8FF",
    OpaqueSkyCover: "#B62B77FF",
    PercentagePeopleDissatisfied: "#B62B77FF",
    RelativeHumidity: "#1C3660FF",
    TotalSkyCover: "#B62B77FF",
    DiffuseHorizontalIlluminance: "#F0AC1BFF",
    DirectNormalIlluminance: "#F0AC1BFF",
    GlobalHorizontalIlluminance: "#F0AC1BFF",
    Illuminance: "#F0AC1BFF",
    Luminance: "#F0AC1B",
    ZenithLuminance: "#F0AC1B",
    Mass: "#585253FF",
    MassFlowRate: "#585253FF",
    ActivityLevel: "#E63187FF",
    Power: "#E63187FF",
    AtmosphericStationPressure: "#24135FFF",
    Pressure: "#24135FFF",
    ClothingInsulation: "#6D104EFF",
    RValue: "#6D104EFF",
    Enthalpy: "#585253FF",
    SpecificEnergy: "#585253FF",
    AirSpeed: "#5D822DFF",
    Speed: "#5D822DFF",
    WindSpeed: "#5D822DFF",
    AirTemperature: "#BC204BFF",
    ClothingTemperature: "#E6484DFF",
    CoreBodyTemperature: "#E63187FF",
    DewPointTemperature: "#AFC1A2FF",
    DryBulbTemperature: "#BC204BFF",
    GroundTemperature: "#6D104EFF",
    HeatIndexTemperature: "#BC204BFF",
    MeanRadiantTemperature: "#6D104EFF",
    NeutralTemperature: "#585253FF",
    OperativeTemperature: "#BC204BFF",
    PhysiologicalEquivalentTemperature: "#E63187FF",
    PrevailingOutdoorTemperature: "#BC204BFF",
    RadiantTemperature: "#6D104EFF",
    SkinTemperature: "#E63187FF",
    SkyTemperature: "#E6484DFF",
    StandardEffectiveTemperature: "#BC204BFF",
    Temperature: "#BC204BFF",
    UniversalThermalClimateIndex: "#BC204BFF",
    WetBulbGlobeTemperature: "#A0D2C9FF",
    WetBulbTemperature: "#A0D2C9FF",
    WindChillTemperature: "#BC204BFF",
    AirTemperatureDelta: "#D50032FF",
    OperativeTemperatureDelta: "#D50032FF",
    RadiantTemperatureDelta: "#D50032FF",
    TemperatureDelta: "#D50032FF",
    CoolingDegreeTime: "#006DA8FF",
    HeatingDegreeTime: "#BC204BFF",
    TemperatureTime: "#585253FF",
    CoreTemperatureCategory: "#585253FF",
    DiscomfortReason: "#585253FF",
    PredictedMeanVote: "#585253FF",
    ThermalComfort: "#585253FF",
    ThermalCondition: "#585253FF",
    ThermalConditionElevenPoint: "#585253FF",
    ThermalConditionFivePoint: "#585253FF",
    ThermalConditionNinePoint: "#585253FF",
    ThermalConditionSevenPoint: "#585253FF",
    UTCICategory: "#585253FF",
    Time: "#585253FF",
    ConvectionCoefficient: "#702F8AFF",
    RadiantCoefficient: "#702F8AFF",
    UValue: "#702F8AFF",
    Voltage: "#F0AC1BFF",
    Volume: "#8F72B0FF",
    VolumeFlowRate: "#8F72B0FF",
    VolumeFlowRateIntensity: "#8F72B0FF",
    GenericType: "#8F72B0FF",
    # Ladybug extended datatypes
    ActualSensationVote: "#D50032FF",
    DailyLightIntegral: "#FFCD38FF",
    ThermalSensation: "#E6AE48FF",
    Humidex: "#8C52D6FF",
    ApparentTemperature: "#E6484DFF",
    DiscomfortIndex: "#E6484DFF",
    RadiationBenefit: "#7C634BFF",
    ShadeBenefit: "#3B2424FF",
    WindBenefit: "#242E3BFF",
    People: "#B62B77FF",
    AreaPerPerson: "#B62B77FF",
    PeoplePerArea: "#B62B77FF",
    Walkability: "#5DBAA7FF",
    Season: "#808080FF",
    # default end use colours
    "heating": "#f1937aff",
    "service water heating": "#f4b183ff",
    "domestic hot water": "#f4b183ff",
    "cooling": "#9cc3e5ff",
    "fans": "#70ad47ff",
    "humidification": "#a8d08dff",
    "pumps": "#c5e0b3ff",
    "lighting interior": "#ffd965ff",
    "lighting exterior": "#fee599ff",
    "street lighting": "#ffc822ff",
    "transportation systems": "#323f4fff",
    "receptacle equipment": "#44546aff",
    "plug load": "#44546aff",
    "specialist equipment": "#8496b0ff",
    "generator": "#adb9caff",
    "all other uses": "#d6dce4ff",
    "ev charging": "#222a35ff",
    # US default colours
    "it": "#b2a7a1ff",
    "interior central fans": "#5daaa2ff",
    "interior local fans": "#aed5d0ff",
    "exhaust fans": "#396c34ff",
    "pump 2": "#ae72afff",
    "heat rejection": "#405aa4ff",
    "heat recovery": "#9180ddff",
    "space cooling": "#00a9e0ff",
    "dx cooling": "#bfe9f7ff",
    "other cooling": "#bfe9f7ff",
    "space heating": "#e6474cff",
    "space heating (electric)": "#e6474cff",
    "space heating (fossil fuel)": "#f9c8cbff",
    "renewable": "#36ee6dff",
    "renewable 2": "#d0edaaff",
    "elevators & escalators": "#221e59ff",
    "process": "#727272ff",
    "data": "#4befd7ff",
    # load balance
    "floor_conduction": "#585253ff",
    "wall_conduction": "#585253ff",
    "roof_conduction": "#585253ff",
    "window_conduction": "#585253ff",
    "electric_equip": "#5576a5ff",
    "gas_equip": "#f9c8cbff",
    "people": "#b62b77ff",
    "solar": "#d06a13ff",
    "service_hot_water": "#f4b183ff",
    "mech_ventilation": "#407020ff",
    "nat_ventilation": "#70ad47ff",
    "infiltration": "#c5e0b3ff",
    # daylight
    "daytime": "#FCE49Dff",
    "daylight": "#FCE49Dff",
    "apparent_daytime": "#dbc892ff",
    "apparent_daylight": "#dbc892ff",
    "astronomical_twilight": "#B9AC86ff",
    "civil_twilight": "#908A7Aff",
    "nautical_twilight": "#817F76ff",
    "nighttime": "#717171ff",
    # time of day
    "morning": "#FFD580ff",
    "afternoon": "#87CEEBff",
    "evening": "#FF6F61ff",
    "night": "#6d6d6dff",
    # random
    "mechanical ventilation": "#407020ff",
    "cooking and catering": "#d38236ff",
    "storage": "#a0a0a0ff",
    "refrigeration": "#a0c1d4ff",
    "water treatment": "#6b8e23ff",
    "waste management": "#8b4513ff",
    "lighting": "#f0ac1bff",
    "hvac": "#d06a13ff",
    "hvac controls": "#d06a13ff",
    "lifts": "#323f4fff",
    "solar pv": "#00a499ff",
    "photovoltaic": "#00a499ff",
    "solar thermal": "#8db9caff",
    "battery storage": "#b62b77ff",
    "electric vehicle charging": "#006da8ff",
    "grid connection": "#24135fff",
    "grid import": "#24135fff",
    "grid export": "#135f42ff",
    "mr blobby": "#d89ba3ff",
    "celeste": "#8bc8b1ff",
    "irrigation": "#18857cff",
    "facade screens": "#ff98ffff",
    "Area": "#585253FF",
    # Materials
    "concrete": "#A9A9A9",  # Medium gray
    "brick": "#B22222",  # Firebrick red
    "wood": "#8B4513",  # Saddle brown
    "glass": "#5F9EA0",  # Cadet blue
    "steel": "#C0C0C0",  # Silver
    "aluminum": "#B0C4DE",  # Light steel blue
    "aluminium": "#B0C4DE",  # Same as aluminum
    "asphalt": "#1C1C1C",  # Very dark gray
    "clay_tile": "#CD5C5C",  # Indian red
    "plaster": "#FFF8DC",  # Cornsilk
    "insulation_foam": "#FFA500",  # Orange
    "drywall": "#FAF0E6",  # Linen
    "granite": "#808080",  # Gray
    "marble": "#F0FFFF",  # Azure
    "copper": "#B87333",  # Copper
    "bamboo": "#DAA520",  # Goldenrod
    "vinyl_siding": "#D8BFD8",  # Thistle
    "cement": "#696969",  # Dim gray
    "stone": "#A0522D",  # Sienna
    "plastic": "#ADD8E6",  # Light blue
    "gypsum": "#FFE4B5",  # Moccasin
    "metal": "#A9A9A9",  # Dark gray
    "insulation": "#F4A460",  # Sandy brown
    # Embodied carbon stages
    "a1toa3": "#D50032",
    "a4": "#E6484D",
    "a5": "#F9C8CB",
    "b1": "#00A9E0",
    "b2": "#2680B4",
    "b3": "#9065C2",
    "b4": "#B62B77",
    "b5": "#C0A7DB",
    "c1": "#D06A13",
    "c2": "#E68B3C",
    "c3": "#F1B37D",
    "c4": "#B48B66",
    "d": "#40884F",
    "stored": "#095236",
    "waste": "#3F3F3F",
    # ... add more as needed
    "eui": "#D06A13FF",
    "energy": "#00A499FF",
}

DEFAULT_COLOURMAPS = {}


def get_default_colour(obj: Any, cutoff: float = 0.5) -> str:
    """Attempt to convert an object to a hex colour string.

    Args:
        obj: 
            The object to convert.
        cutoff: 
            The cutoff for fuzzy string matching (between 0 and 1). Higher 
            values are more strict for finding matches.

    Returns:
        str: 
            A colour string in hex format.

    """
    default_color = "#808080ff"

    # process ladybug datatype
    if isinstance(obj, DataTypeBase):
        try:
            return DEFAULT_COLOURS[type(obj)]
        except KeyError:
            CONSOLE_LOGGER.warning(
                f"{obj} doesn't seem to have a related colour. Defaulting to gray."
            )
            return default_color

    # process Ladybug color objects
    if isinstance(obj, Color):
        return to_hex(
            (
                float(obj.r / 255),
                float(obj.g / 255),
                float(obj.b / 255),
                float(obj.a / 255),
            ),
            keep_alpha=True,
        )

    # process native named colors (or rgb-like objects)
    try:
        return to_hex(obj, keep_alpha=True)
    except (KeyError, TypeError, ValueError):
        pass

    # process strings
    if isinstance(obj, str):
        s_lower = obj.lower()

        # First, try exact matches
        if DEFAULT_COLOURS.get(s_lower, None):
            return DEFAULT_COLOURS[s_lower]
        # then try substring matches
        for key in DEFAULT_COLOURS:
            if isinstance(key, str):
                if s_lower in key.lower():
                    return DEFAULT_COLOURS[key]
        # Then, try fuzzy matches
        matches = difflib.get_close_matches(
            s_lower,
            [k for k in DEFAULT_COLOURS if isinstance(k, str)],
            n=1,
            cutoff=cutoff,
        )
        if matches:
            return DEFAULT_COLOURS[matches[0]]

    CONSOLE_LOGGER.warning(
        f"{obj} doesn't seem to have a related colour. Defaulting to gray."
    )

    return default_color


def to_colour(obj: Any, fmt: str = "hex", alpha: Optional[float] = None) -> Any:
    """Convert a colour-like object to another colour-like object.
    
    Args:
        obj: 
            The object to convert.
        fmt: 
            The desired output format.
        alpha: 
            The desired alpha value for the output colour (between 0 and 1).
            This is optional; if not provided, the alpha value from the input colour
            will be used.
    Returns:
        Any:
            The converted colour-like object.
    """
    
    # convert color into a common format (we use r, g, b, a 0-1 here as intermediary)
    if isinstance(obj, Color):
        r, g, b, a = obj.r / 255, obj.g / 255, obj.b / 255, obj.a / 255
    elif isinstance(obj, (list, tuple)) and len(obj) in [3,4]:
        if all(isinstance(c, int) and 0 <= c <= 255 for c in obj):
            r, g, b = [c / 255 for c in obj[:3]]
            a = obj[3] / 255 if len(obj) == 4 else 1.0
        elif all(isinstance(c, (float, int)) and 0.0 <= c <= 1.0 for c in obj):
            r, g, b = obj[:3]
            a = obj[3] if len(obj) == 4 else 1.0
        else:
            raise ValueError(f"Color tuple/list {obj} must have values between 0-1 or 0-255.")
    elif isinstance(obj, str):
        s = obj.strip()
        # Strict hex pattern: allow #RGB, #RGBA, #RRGGBB, #RRGGBBAA only
        hex_pattern = re.compile(r'^#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{4}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$')
        plotly_pattern = re.compile(
            r'rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s*,\s*(0|0?\.\d+|1(\.0)?))?\s*\)'
        )

        # If it looks like a hex (starts with #), validate strictly and fail if invalid
        if s.startswith("#"):
            if not hex_pattern.match(s):
                CONSOLE_LOGGER.warning(f"Invalid hex color string: {obj!r}")
                raise ValueError(f"Invalid hex color string: {obj!r}")
            r, g, b, a = to_rgba(s)
        else:
            # try a named/matplotlib color first
            try:
                r, g, b, a = to_rgba(s)
            except ValueError:
                # try a plotly-style rgba(...) string
                m = plotly_pattern.match(s)
                if m:
                    r = int(m.group(1)) / 255
                    g = int(m.group(2)) / 255
                    b = int(m.group(3)) / 255
                    a = float(m.group(4)) if m.group(4) is not None else 1.0
                else:
                    CONSOLE_LOGGER.warning(f"Unrecognized colour string: {obj!r}")
                    raise ValueError(f"Cannot parse colour string: {obj!r}")
    else:
        raise NotImplementedError(f"Cannot convert object {obj} ({type(obj)}) to colour.")
    
    if alpha is not None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha value must be between 0 and 1.")
        a = alpha
        
    match fmt:
        case "hex":
            new_color = to_hex((r, g, b, a), keep_alpha=True)
        case "rgb":
            new_color = (r, g, b)
        case "rgba":
            new_color = (r, g, b, a)
        case "rgb255":
            new_color = (int(r * 255), int(g * 255), int(b * 255))
        case "rgba255":
            new_color = (int(r * 255), int(g * 255), int(b * 255), int(a * 255))
        case "plotly":
            new_color = f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {float(a)})"
        case "ladybug":
            new_color = Color(int(r * 255), int(g * 255), int(b * 255), int(a * 255))
        case _:
            raise ValueError(f"Colour format '{fmt}' is not recognized.")
        
    return new_color
