from typing import List

from .shelter import Shelter
from .shelters_overlap import shelters_overlap


def any_shelters_overlap(shelters: List[Shelter]) -> bool:
    """Check whether any shelter in a list overlaps with any other shelter in the list.

    Args:
        shelters (List[Shelter]): A list of shelter objects.

    Returns:
        bool: True if any shelter in the list overlaps with any other shelter in the list.
    """

    for shelter1 in shelters:
        for shelter2 in shelters:
            if shelter1 != shelter2:
                if shelters_overlap(shelter1, shelter2):
                    return True
    return False