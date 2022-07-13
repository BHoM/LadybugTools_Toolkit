from typing import Dict

from honeybee.model import Model
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybug_comfort.collection.solarcal import HorizontalSolarCal
from ladybug_comfort.parameter.solarcal import SolarCalParameter

from .mean_radiant_temperature_inputs import mean_radiant_temperature_inputs
from .radiant_temperature_from_surfaces import radiant_temperature_from_surfaces


def mean_radiant_temperature(
    model: Model, epw: EPW
) -> Dict[str, HourlyContinuousCollection]:
    """Run both a surface temperature and solar radiation simulation concurrently and return the
        combined results, plus the mean radiant temperature for both shaded and unshaded conditions.

    Args:
        model (Model): A model used as part of the External Comfort workflow.
        epw (EPW): The EPW file in which to simulate the model.

    Returns:
        Dict[str, HourlyContinuousCollection]: A dictionary containing all collections describing
            shaded and unshaded mean-radiant-temperature.
    """

    mrt_collections = mean_radiant_temperature_inputs(model, epw)

    mrt_collections[
        "shaded_longwave_mean_radiant_temperature"
    ] = radiant_temperature_from_surfaces(
        [
            mrt_collections["shaded_below_temperature"],
            mrt_collections["shaded_above_temperature"],
        ],
        [0.5, 0.5],
    )

    mrt_collections[
        "unshaded_longwave_mean_radiant_temperature"
    ] = radiant_temperature_from_surfaces(
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
