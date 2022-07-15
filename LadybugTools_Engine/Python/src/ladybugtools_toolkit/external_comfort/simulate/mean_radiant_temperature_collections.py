from typing import Dict

from honeybee.model import Model
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybug_comfort.collection.solarcal import HorizontalSolarCal
from ladybug_comfort.parameter.solarcal import SolarCalParameter

from .longwave_mean_radiant_temperature import longwave_mean_radiant_temperature
from .solar_radiation import solar_radiation
from .surface_temperature import surface_temperature


def mean_radiant_temperature_collections(
    model: Model, epw: EPW
) -> Dict[str, HourlyContinuousCollection]:
    """Run both a surface temperature and solar radiation simulation and return the
        combined results, plus the mean radiant temperature for both shaded and unshaded conditions.

    Args:
        model (Model): A model used as part of the External Comfort workflow.
        epw (EPW): The EPW file in which to simulate the model.

    Returns:
        Dict[str, HourlyContinuousCollection]: A dictionary containing all collections describing
            shaded and unshaded mean-radiant-temperature.
    """

    # Run surface temperature and radiation simulations
    solar_radiation_results = solar_radiation(model, epw)
    surface_temperature_results = surface_temperature(model, epw)

    mrt_collections = {**solar_radiation_results, **surface_temperature_results}

    # Calculate LW MRT from surface temperatures
    mrt_collections[
        "shaded_longwave_mean_radiant_temperature"
    ] = longwave_mean_radiant_temperature(
        [
            mrt_collections["shaded_below_temperature"],
            mrt_collections["shaded_above_temperature"],
        ],
        [0.5, 0.5],
    )

    mrt_collections[
        "unshaded_longwave_mean_radiant_temperature"
    ] = longwave_mean_radiant_temperature(
        [
            mrt_collections["unshaded_below_temperature"],
            mrt_collections["unshaded_above_temperature"],
        ],
        [0.5, 0.5],
    )

    # Calculate the effective Mean Radiant Temperature from inputs
    solar_body_par = SolarCalParameter()
    fract_body_exp = 0
    ground_reflectivity = 0

    mrt_collections["shaded_mean_radiant_temperature"] = HorizontalSolarCal(
        epw.location,
        mrt_collections["shaded_direct_radiation"],
        mrt_collections["shaded_diffuse_radiation"],
        mrt_collections["shaded_longwave_mean_radiant_temperature"],
        fract_body_exp,
        ground_reflectivity,
        solar_body_par,
    ).mean_radiant_temperature

    mrt_collections["unshaded_mean_radiant_temperature"] = HorizontalSolarCal(
        epw.location,
        mrt_collections["unshaded_direct_radiation"],
        mrt_collections["unshaded_diffuse_radiation"],
        mrt_collections["unshaded_longwave_mean_radiant_temperature"],
        fract_body_exp,
        ground_reflectivity,
        solar_body_par,
    ).mean_radiant_temperature

    return mrt_collections
