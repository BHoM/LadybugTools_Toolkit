from ladybug.psychrometrics import enthalpy_from_db_hr
from ladybug.epw import EPW
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.specificenergy import Enthalpy

import sys
sys.path.append(r"C:\ProgramData\BHoM\Extensions")
from LadybugTools.humidity_ratio import humidity_ratio as hr


def enthalpy(epw: EPW, humidity_ratio: HourlyContinuousCollection = None) -> HourlyContinuousCollection:
    """Calculate an annual hourly enthalpy collection for a given EPW.

    Args:
        epw (EPW): A ladybug EPW object.
        humidity_ratio (HourlyContinuousCollection): Optional input for pre-calculated humidity ratio.

    Returns:
        HourlyContinuousCollection: A Enthalpy data collection.
    """
    enthalpy = HourlyContinuousCollection.compute_function_aligned(
        enthalpy_from_db_hr,
        [
            epw.dry_bulb_temperature,
            hr(epw) if humidity_ratio is None else humidity_ratio,
        ],
        Enthalpy(),
        "kJ/kg",
    )
    return enthalpy
