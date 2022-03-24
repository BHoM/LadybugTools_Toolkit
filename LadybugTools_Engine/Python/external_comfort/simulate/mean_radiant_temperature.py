import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import Temperature
from ladybug.epw import EPW
from ladybug_comfort.collection.solarcal import HorizontalSolarCal
from ladybug_comfort.parameter.solarcal import SolarCalParameter


def mean_radiant_temperature(
    epw: EPW,
    surface_temperature: HourlyContinuousCollection,
    direct_radiation: HourlyContinuousCollection,
    diffuse_radiation: HourlyContinuousCollection,
) -> HourlyContinuousCollection:
    """Using the SolarCal method, convert surrounding surface temperature and direct/diffuse radiation into mean radiant temperature.

    Args:
        epw (EPW): A ladybug EPW object.
        surface_temperature (HourlyContinuousCollection): A ladybug surface temperature data collection.
        direct_radiation (HourlyContinuousCollection): A ladybug radiation data collection representing direct solar radiation.
        diffuse_radiation (HourlyContinuousCollection): A ladybug radiation data collection representing diffuse solar radiation.

    Returns:
        HourlyContinuousCollection: A ladybug mean radiant temperature data collection.
    """
    fract_body_exp = 0
    ground_reflectivity = 0

    if not isinstance(surface_temperature.header.data_type, Temperature):
        surface_temperature.header.data_type = Temperature

    solar_body_par = SolarCalParameter()
    solar_mrt_obj = HorizontalSolarCal(
        epw.location,
        direct_radiation,
        diffuse_radiation,
        surface_temperature,
        fract_body_exp,
        ground_reflectivity,
        solar_body_par,
    )

    mrt = solar_mrt_obj.mean_radiant_temperature

    return mrt
