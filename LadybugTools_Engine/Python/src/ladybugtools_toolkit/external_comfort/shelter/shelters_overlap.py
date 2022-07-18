from ladybugtools_toolkit.external_comfort.shelter import Shelter


def shelters_overlap(shelter1: Shelter, shelter2: Shelter) -> bool:
    """Return True if two shelters overlap with each other.

    Args:
        shelter1 (Shelter): The first shelter to compare.
        shelter2 (Shelter): The second shelter to compare.

    Returns:
        bool: True if the two shelters overlap with each other in any way.
    """
    for poly1 in shelter1.polygons():
        for poly2 in shelter2.polygons():
            if any(
                [
                    poly1.crosses(poly2),
                    poly1.contains(poly2),
                    poly1.within(poly2),
                    poly1.covers(poly2),
                    poly1.covered_by(poly2),
                    poly1.overlaps(poly2),
                ]
            ):
                return True
    return False
