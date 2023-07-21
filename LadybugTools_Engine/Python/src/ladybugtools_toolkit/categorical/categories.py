from enum import Enum

import numpy as np

from . import Categorical


class ComfortClass(Enum):
    """Thermal comfort categories."""

    TOO_COLD = "Too cold"
    COMFORTABLE = "Comfortable"
    TOO_HOT = "Too hot"


UTCI_SIMPLIFIED_CATEGORIES = Categorical(
    bins=(-np.inf, 9, 26, np.inf),
    bin_names=("Too cold", "Comfortable", "Too hot"),
    colors=("#3C65AF", "#2EB349", "#C31F25"),
    name="UTCI (simplified)",
)
UTCI_SIMPLIFIED_CATEGORIES.comfort_classes = [
    ComfortClass.TOO_COLD,
    ComfortClass.COMFORTABLE,
    ComfortClass.TOO_HOT,
]

UTCI_DEFAULT_CATEGORIES = Categorical(
    bin_names=(
        "Extreme cold stress",
        "Very strong cold stress",
        "Strong cold stress",
        "Moderate cold stress",
        "Slight cold stress",
        "No thermal stress",
        "Moderate heat stress",
        "Strong heat stress",
        "Very strong heat stress",
        "Extreme heat stress",
    ),
    bins=(-np.inf, -40, -27, -13, 0, 9, 26, 32, 38, 46, np.inf),
    colors=(
        "#0D104B",
        "#262972",
        "#3452A4",
        "#3C65AF",
        "#37BCED",
        "#2EB349",
        "#F38322",
        "#C31F25",
        "#7F1416",
        "#580002",
    ),
    name="UTCI",
)
UTCI_DEFAULT_CATEGORIES.comfort_classes = [
    ComfortClass.TOO_COLD,
    ComfortClass.TOO_COLD,
    ComfortClass.TOO_COLD,
    ComfortClass.TOO_COLD,
    ComfortClass.TOO_COLD,
    ComfortClass.COMFORTABLE,
    ComfortClass.TOO_HOT,
    ComfortClass.TOO_HOT,
    ComfortClass.TOO_HOT,
    ComfortClass.TOO_HOT,
]

UTCI_SLIGHTHEATSTRESS_CATEGORIES = Categorical(
    bin_names=(
        "Extreme cold stress",
        "Very strong cold stress",
        "Strong cold stress",
        "Moderate cold stress",
        "Slight cold stress",
        "No thermal stress",
        "Slight heat stress",
        "Moderate heat stress",
        "Strong heat stress",
        "Very strong heat stress",
        "Extreme heat stress",
    ),
    bins=(-np.inf, -40, -27, -13, 0, 9, 26, 28, 32, 38, 46, np.inf),
    colors=(
        "#0F0F4B",
        "#262872",
        "#3354A5",
        "#3D64AF",
        "#38BCED",
        "#2DB247",
        "#D4D317",
        "#F38023",
        "#C41F26",
        "#7E1416",
        "#500003",
    ),
    name="UTCI (inc. slight heat stress)",
)
UTCI_SLIGHTHEATSTRESS_CATEGORIES.comfort_classes = [
    ComfortClass.TOO_COLD,
    ComfortClass.TOO_COLD,
    ComfortClass.TOO_COLD,
    ComfortClass.TOO_COLD,
    ComfortClass.TOO_COLD,
    ComfortClass.COMFORTABLE,
    ComfortClass.TOO_HOT,
    ComfortClass.TOO_HOT,
    ComfortClass.TOO_HOT,
    ComfortClass.TOO_HOT,
    ComfortClass.TOO_HOT,
]

BEAUFORT_CATEGORIES = Categorical(
    bin_names=(
        "Calm",
        "Light Air",
        "Light Breeze",
        "Gentle Breeze",
        "Moderate Breeze",
        "Fresh Breeze",
        "Strong Breeze",
        "Near Gale",
        "Gale",
        "Severe Gale",
        "Storm",
        "Violent Storm",
        "Hurricane",
    ),
    colors=(
        "#FFFFFF",
        "#CCFFFF",
        "#99FFCC",
        "#99FF99",
        "#99FF66",
        "#99FF00",
        "#CCFF00",
        "#FFFF00",
        "#FFCC00",
        "#FF9900",
        "#FF6600",
        "#FF3300",
        "#FF0000",
    ),
    bins=(0, 0.3, 1.5, 3.3, 5.5, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6, np.inf),
    name="Beaufort scale",
)
