from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.specificenergy import Enthalpy
from ladybug.epw import EPW
from ladybug.psychrometrics import enthalpy_from_db_hr

from .humidity_ratio import humidity_ratio as hr


def enthalpy(
    epw: EPW, humidity_ratio: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate an annual hourly enthalpy for a given EPW.

    Args:
        epw (EPW): _description_
        humidity_ratio (HourlyContinuousCollection, optional): A pre-calculated HourlyContinuousCollection of humidity ratios. Defaults to None.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of enthalpies.
    """

    if not humidity_ratio:
        humidity_ratio = hr(epw)

    return HourlyContinuousCollection.compute_function_aligned(
        enthalpy_from_db_hr,
        [
            epw.dry_bulb_temperature,
            humidity_ratio,
        ],
        Enthalpy(),
        "kJ/kg",
    )