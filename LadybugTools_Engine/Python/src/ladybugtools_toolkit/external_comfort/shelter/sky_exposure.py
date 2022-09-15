from typing import List

from ladybugtools_toolkit.external_comfort.shelter.any_shelters_overlap import (
    any_shelters_overlap,
)
from ladybugtools_toolkit.external_comfort.shelter.shelter import Shelter


from python_toolkit.bhom.analytics import analytics


@analytics
def sky_exposure(shelters: List[Shelter]) -> float:
    """Determine the proportion of the sky visible beneath a set of shelters. Includes porosity of
        shelters in the resultant value (e.g. fully enclosed by a single 50% porous shelter would
        mean 50% sky exposure).

    Args:
        shelters (List[Shelter]):
            Shelters that could block the sun.

    Returns:
        float:
            The proportion of sky visible beneath shelters.
    """

    if any_shelters_overlap(shelters):
        raise ValueError(
            "Shelters overlap, so sky-exposure calculation cannot be completed."
        )

    exposure = 1
    for shelter in shelters:
        exposure -= shelter.sky_blocked()
    return exposure
