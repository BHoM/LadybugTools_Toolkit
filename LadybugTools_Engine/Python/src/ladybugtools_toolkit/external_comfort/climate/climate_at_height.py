import numpy as np


from ladybugtools_toolkit import analytics
from ladybug.datacollection import HourlyContinuousCollection


@analytics
def temperature_at_height(
    reference_temperature: HourlyContinuousCollection,
    reference_height: float,
    target_height: float,
    lapse_rate: float = 6.5,
) -> HourlyContinuousCollection:  # TODO - check
    return reference_temperature - ((target_height - reference_height) / 1000 * lapse_rate)

def radiation_at_height(
    reference_radiation: HourlyContinuousCollection,
    reference_height: float,
    target_height: float,
    lapse_rate_percentage: float = 0.08,
) -> HourlyContinuousCollection:
    return reference_radiation * (1 + (target_height - reference_height) / 1000 * lapse_rate)

def pressure_at_height(altitude):
    """https://archive.psas.pdx.edu/RocketScience/PressureAltitude_Derived.pdf"""
    return 100 * ((44331.514 - altitude) / 11880.516) ** (1 / 0.1902632)

def relative_humidity_at_height(
    reference_rh: HourlyContinuousCollection,
    reference_height: float,
    target_height: float,
    lapse_rate: float = 4,
) -> HourlyContinuousCollection:  # TODO - check
    return reference_rh - ((target_height - reference_height) / 1000 * lapse_rate)