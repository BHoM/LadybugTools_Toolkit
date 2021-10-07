from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from typing import Dict

import sys
sys.path.append(r"C:\ProgramData\BHoM\Extensions")
from LadybugTools.enthalpy import enthalpy
from LadybugTools.humidity_ratio import humidity_ratio
from LadybugTools.wet_bulb_temperature import wet_bulb_temperature


def psychrometrics_collections(epw: EPW) -> Dict[str, HourlyContinuousCollection]:
    """Generate a dictionary of calculated psychrometric collections.

    Args:
        epw (EPW): A ladybug EPW object.

    Returns:
        Dict[str, HourlyContinuousCollection]: A set of psychrometric data collections.
    """
    hr = humidity_ratio(epw)
    return {
        "humidity_ratio": humidity_ratio(epw),
        "wet_bulb_temperature": wet_bulb_temperature(epw),
        "enthalpy": enthalpy(epw, hr),
    }
