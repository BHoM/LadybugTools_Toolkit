"""Predetermined categories for categorical data."""
import numpy as np

from .categorical import Categorical, CategoricalComfort, ComfortClass

UTCI_SIMPLIFIED_CATEGORIES = CategoricalComfort(
    bins=(-np.inf, 9, 26, np.inf),
    bin_names=("Too cold", "Comfortable", "Too hot"),
    colors=("#3C65AF", "#2EB349", "#C31F25"),
    name="UTCI (simplified)",
    comfort_classes=[
        ComfortClass.TOO_COLD,
        ComfortClass.COMFORTABLE,
        ComfortClass.TOO_HOT,
    ],
)

UTCI_DEFAULT_CATEGORIES = CategoricalComfort(
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
    comfort_classes=[
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
    ],
)

UTCI_SLIGHTHEATSTRESS_CATEGORIES = CategoricalComfort(
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
    name="UTCI inc. slight heat stress",
    comfort_classes=[
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
    ],
)

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

WBGT_CATEGORIES = CategoricalComfort(
    name="Wet Bulb Globe Temperature",
    bin_names=["Any activity", "Very heavy activity", "Heavy activity", "Moderate activity", "Light activity", "Resting only"],
    bins=[-np.inf, 23, 25, 28, 30, 33, np.inf],
    colors=["#c1c1c1", "#32cd32", "#ffff00", "#ffa500", "#ff0000", "#000000"],
    comfort_classes=[ComfortClass.COMFORTABLE, ComfortClass.TOO_HOT, ComfortClass.TOO_HOT, ComfortClass.TOO_HOT, ComfortClass.TOO_HOT, ComfortClass.TOO_HOT]
)

HUMIDEX_CATEGORIES = CategoricalComfort(
    name="Humidex",
    bin_names=["Comfort", "Little to no discomfort", "Some discomfort", "Great discomfort; avoid exertion", "Dangerous; heat stroke quite possible"],
    bins=[-np.inf, 20, 29, 39, 45, np.inf],
    colors=["#c1c1c1", "#32cd32", "#ffff00", "#ff8c00", "#ff0000"],
    comfort_classes=[ComfortClass.COMFORTABLE, ComfortClass.TOO_HOT, ComfortClass.TOO_HOT, ComfortClass.TOO_HOT, ComfortClass.TOO_HOT]
)

HEAT_INDEX_CATEGORIES = CategoricalComfort(
    name="Heat Index",
    bin_names=["Comfort", "Caution", "Extreme caution", "Danger", "Extreme danger"],
    bins=[-np.inf, 27, 32, 41, 54, np.inf],
    colors=["#c1c1c1", "#ffff66", "#ffd700", "#ff8c00", "#ff0000"],
    comfort_classes=[ComfortClass.COMFORTABLE, ComfortClass.TOO_HOT, ComfortClass.TOO_HOT, ComfortClass.TOO_HOT, ComfortClass.TOO_HOT]
)