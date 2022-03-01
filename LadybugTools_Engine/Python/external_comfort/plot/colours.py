import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from enum import Enum
from typing import Tuple
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
)
import numpy as np


class UTCICategory(Enum):
    extreme_cold_stress = "#0D104B"
    very_strong_cold_stress = "#262972"
    strong_cold_stress = "#3452A4"
    moderate_cold_stress = "#3C65AF"
    slight_cold_stress = "#37BCED"
    no_thermal_stress = "#2EB349"
    moderate_heat_stress = "#F38322"
    strong_heat_stress = "#C31F25"
    very_strong_heat_stress = "#7F1416"
    extreme_heat_stress = "#580002"

    @property
    def hex_colour(self) -> str:
        return self.value

    @property
    def title(self) -> str:
        if self == self.extreme_cold_stress:
            return "Extreme cold stress"
        elif self == self.very_strong_cold_stress:
            return "Very strong cold stress"
        elif self == self.strong_cold_stress:
            return "Strong cold stress"
        elif self == self.moderate_cold_stress:
            return "Moderate cold stress"
        elif self == self.slight_cold_stress:
            return "Slight cold stress"
        elif self == self.no_thermal_stress:
            return "No thermal stress"
        elif self == self.moderate_heat_stress:
            return "Moderate heat stress"
        elif self == self.strong_heat_stress:
            return "Strong heat stress"
        elif self == self.very_strong_heat_stress:
            return "Very strong heat stress"
        elif self == self.extreme_heat_stress:
            return "Extreme heat stress"
        else:
            return "Unknown"

    @property
    def value_range(self) -> Tuple[float, float]:
        if self == self.extreme_cold_stress:
            return (-100, -40)
        elif self == self.very_strong_cold_stress:
            return (-40, -27)
        elif self == self.strong_cold_stress:
            return (-27, -13)
        elif self == self.moderate_cold_stress:
            return (-13, 0)
        elif self == self.slight_cold_stress:
            return (0, 9)
        elif self == self.no_thermal_stress:
            return (9, 26)
        elif self == self.moderate_heat_stress:
            return (26, 32)
        elif self == self.strong_heat_stress:
            return (32, 38)
        elif self == self.very_strong_heat_stress:
            return (38, 46)
        elif self == self.extreme_heat_stress:
            return (46, 100)
        else:
            return (0, 0)


UTCI_COLOURS = [i.hex_colour for i in UTCICategory]
UTCI_COLOURMAP = LinearSegmentedColormap.from_list("UTCI", colors=UTCI_COLOURS, N=10)
UTCI_CATEGORIES = [i.title for i in UTCICategory]
UTCI_BOUNDS = np.unique(np.array([i.value_range for i in UTCICategory]).flatten())
UTCI_COLOURMAP_NORM = BoundaryNorm(boundaries=UTCI_BOUNDS, ncolors=10)
