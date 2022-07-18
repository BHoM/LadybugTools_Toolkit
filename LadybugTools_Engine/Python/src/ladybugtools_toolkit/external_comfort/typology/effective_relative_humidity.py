from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.moisture.evaporative_cooling_effect_collection import (
    evaporative_cooling_effect_collection,
)
from ladybugtools_toolkit.external_comfort.typology import Typology


def effective_relative_humidity(
    typology: Typology, epw: EPW
) -> HourlyContinuousCollection:
    """Get the effective RH for the given EPW file for this Typology.

    Args:
        typology (Typology): A Typology object.
        epw (EPW): A ladybug EPW object.

    Returns:
        HourlyContinuousCollection: The effective RH following application of any evaporative
            cooling effects.
    """
    return evaporative_cooling_effect_collection(
        epw, typology.evaporative_cooling_effect
    )[1]
