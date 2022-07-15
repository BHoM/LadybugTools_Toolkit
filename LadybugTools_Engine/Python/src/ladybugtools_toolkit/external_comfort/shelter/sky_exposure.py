from typing import List

from .any_shelters_overlap import any_shelters_overlap
from .shelter import Shelter


def sky_exposure(shelters: List[Shelter]) -> float:
    """Determine the proportion of the sky visible beneath a set of shelters. Includes porosity of
        shelters in the resultant value (e.g. fully enclosed by a single 50% porous shelter would
        mean 50% sky exposure).

    Args:
        shelters (List[Shelter]): Shelters that could block the sun.

    Returns: The proportion of sky visible beneath shelters.
    """

    if any_shelters_overlap(shelters):
        raise ValueError(
            "Shelters overlap, so sky-exposure calculation cannot be completed."
        )

    exposure = 1
    for shelter in shelters:
        print(shelter)
        exposure -= shelter._sky_occlusion
    return exposure
