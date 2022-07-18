from __future__ import annotations

from typing import Any, Dict, List

from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.shelter import Shelter
from ladybugtools_toolkit.external_comfort.shelter.any_shelters_overlap import (
    any_shelters_overlap,
)
from ladybugtools_toolkit.external_comfort.shelter.sky_exposure import sky_exposure
from ladybugtools_toolkit.external_comfort.shelter.sun_exposure import sun_exposure


class Typology:
    """An external comfort typology, described by shelters and a proportion of evaporative
        cooling.

    Args:
        name (str): The name of the external comfort typology.
        shelters (List[Shelter], optional): A list of shelters modifying exposure to the
            elements. Defaults to None.
        evaporative_cooling_effect (float, optional): An amount of evaporative cooling to add to
            results calculated by this typology. Defaults to 0.


    Returns:
        Typology: An external comfort typology.
    """

    def __init__(
        self,
        name: str,
        shelters: List[Shelter] = None,
        evaporative_cooling_effectiveness: float = 0,
    ) -> Typology:

        self.name = name
        self.shelters = shelters
        self.evaporative_cooling_effect = evaporative_cooling_effectiveness

        if any_shelters_overlap(shelters):
            raise ValueError("Shelters overlap")

        if (
            evaporative_cooling_effectiveness < 0
            or evaporative_cooling_effectiveness > 1
        ):
            raise ValueError("Evaporative cooling effect must be between 0 and 1.")

    def __repr__(self):
        return f"{self.__class__.__name__} - {self.name}"

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        d = {
            "name": self.name,
            "shelters": self.shelters,
            "evaporative_cooling_effect": self.evaporative_cooling_effect,
        }
        return d

    def sky_exposure(self) -> float:
        """Direct access to "sky_exposure" method for this typology object."""
        return sky_exposure(self.shelters)

    def sun_exposure(self, epw: EPW) -> List[float]:
        """Direct access to "sun_exposure" method for this typology object."""
        return sun_exposure(self.shelters, epw)
