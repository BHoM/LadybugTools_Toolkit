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

# WBGT categories from https://en.wikipedia.org/wiki/Wet-bulb_globe_temperature
WBGT_CATEGORIES = CategoricalComfort(
    name="Wet Bulb Globe Temperature",
    bin_names=[
        "Any activity",
        "Very heavy activity",
        "Heavy activity",
        "Moderate activity",
        "Light activity",
        "Resting only",
    ],
    bins=[-np.inf, 23, 25, 28, 30, 33, np.inf],
    colors=["#c1c1c1", "#32cd32", "#ffff00", "#ffa500", "#ff0000", "#000000"],
    comfort_classes=[
        ComfortClass.COMFORTABLE,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
    ],
)

# Humidex categories from https://en.wikipedia.org/wiki/Humidex
HUMIDEX_CATEGORIES = CategoricalComfort(
    name="Humidex",
    bin_names=[
        "Comfort",
        "Little to no discomfort",
        "Some discomfort",
        "Great discomfort; avoid exertion",
        "Dangerous; heat stroke quite possible",
    ],
    bins=[-np.inf, 20, 29, 39, 45, np.inf],
    colors=["#c1c1c1", "#32cd32", "#ffff00", "#ff8c00", "#ff0000"],
    comfort_classes=[
        ComfortClass.COMFORTABLE,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
    ],
)

# Heat index categories from https://www.weather.gov/ffc/hichart
HEAT_INDEX_CATEGORIES = CategoricalComfort(
    name="Heat Index",
    bin_names=["Comfort", "Caution", "Extreme caution", "Danger", "Extreme danger"],
    bins=[-np.inf, 27, 32, 41, 54, np.inf],
    colors=["#c1c1c1", "#ffff66", "#ffd700", "#ff8c00", "#ff0000"],
    comfort_classes=[
        ComfortClass.COMFORTABLE,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
    ],
)

# Wind chill categories from https://www.noaa.gov/jetstream/synoptic/wind-chill
WIND_CHILL_CATEGORIES = CategoricalComfort(
    name="Wind Chill",
    bin_names=[
        "No frostbite risk",
        ">2 hours to frostbite",
        "≤30 minutes to frostbite",
        "≤10 minutes to frostbite",
        "≤5 minutes to frostbite",
    ][::-1],
    bins=[np.inf, 37.77777778, -8.33333333, -27.22222222, -45.55555556, -np.inf][::-1],
    colors=["#ffffff", "#a6d5ff", "#3a6ca5", "#1d316b", "#4a005a"],
    comfort_classes=[
        ComfortClass.COMFORTABLE,
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
    ],
)

# PET categories from H Mayer. “Human-Biometeorologische Probleme Des Stadtklimas.” Geowissenschaften 14, no. 6 (June 1996): 233–39.
PET_CATEGORIES = CategoricalComfort(
    name="Physiological Equivalent Temperature",
    bin_names=(
        "Extreme cold stress",
        "Strong cold stress",
        "Moderate cold stress",
        "Slight cold stress",
        "No thermal stress",
        "Slight heat stress",
        "Moderate heat stress",
        "Strong heat stress",
        "Extreme heat stress",
    ),
    bins=(-np.inf, 4, 8, 13, 18, 23, 29, 35, 41, np.inf),
    colors=(
        "#5e4fa2",
        "#3f97b7",
        "#89d0a4",
        "#d8ef9b",
        "#fffebe",
        "#fed27f",
        "#f88c51",
        "#dc484c",
        "#9e0142",
    ),
    comfort_classes=[
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

# SET categories from Li, Kunming, Xiao Liu, and Ying Bao. “Evaluating the Performance of Different Thermal Indices on Quantifying Outdoor Thermal Sensation in Humid Subtropical Residential Areas of China.” Frontiers in Environmental Science 10 (December 1, 2022): 1071668. https://doi.org/10.3389/fenvs.2022.1071668.
SET_CATEGORIES = CategoricalComfort(
    name="Standard Effective Temperature",
    bin_names=(
        "Cold",
        "Cool",
        "Slightly Cool",
        "Neutral",
        "Slightly warm",
        "Warm",
        "Hot",
    ),
    bins=(-np.inf, 14, 20, 26, 32, 38, 44, np.inf),
    colors=("#053061", "#3783bb", "#a7d0e4", "#f7f6f6", "#f7b799", "#c94741", "#67001f"),
    comfort_classes=[
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
        ComfortClass.COMFORTABLE,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
    ],
)

# From Li, Kunming, Xiao Liu, and Ying Bao. “Evaluating the Performance of Different Thermal Indices on Quantifying Outdoor Thermal Sensation in Humid Subtropical Residential Areas of China.” Frontiers in Environmental Science 10 (December 1, 2022): 1071668. https://doi.org/10.3389/fenvs.2022.1071668.
THERMAL_SENSATION_CATEGORIES = CategoricalComfort(
    name="Thermal Sensation",
    bin_names=(
        "Cold",
        "Cool",
        "Slightly Cool",
        "Neutral",
        "Slightly warm",
        "Warm",
        "Hot",
    ),
    bins=(-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf),
    colors=("#053061", "#3783bb", "#a7d0e4", "#f7f6f6", "#f7b799", "#c94741", "#67001f"),
    comfort_classes=[
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
        ComfortClass.COMFORTABLE,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
    ],
)

ACTUAL_SENSATION_VOTE_CATEGORIES = CategoricalComfort(
    name="Thermal Sensation",
    bin_names=(
        "Cold",
        "Cool",
        "Slightly Cool",
        "Neutral",
        "Slightly warm",
        "Warm",
        "Hot",
    ),
    bins=(-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf),
    colors=("#053061", "#3783bb", "#a7d0e4", "#f7f6f6", "#f7b799", "#c94741", "#67001f"),
    comfort_classes=[
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
        ComfortClass.COMFORTABLE,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
    ],
)

DISCOMFORT_INDEX_CATEGORIES = CategoricalComfort(
    name="Discomfort Index",
    bin_names=(
        "Hyper-glacial",
        "Glacial",
        "Extremely cold",
        "Very cold",
        "Cold",
        "Cool",
        "Comfortable",
        "Hot",
        "Very hot",
        "Torrid",
    ),
    bins=(-np.inf, -40, -20, -10, -1.8, 13, 15, 20, 26.5, 30, np.inf),
    colors=(
        "#5e4fa2",
        "#3a7eb8",
        "#54aead",
        "#89d0a4",
        "#bfe5a0",
        "#eaf79e",
        "#fffebe",
        "#fee593",
        "#fdbf6f",
        "#f88c51",
    ),
    comfort_classes=[
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
        ComfortClass.TOO_COLD,
        ComfortClass.COMFORTABLE,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
    ],
)

# from R G Steadman. “The Assessment of Sultriness. Part I: A Temperature-Humidity Index Based on Human Physiology and Clothing Science.” Journal of Applied Meteorology 18, no. 7 (July 1979): 861–73.
APPARENT_TEMPERATURE_CATEGORIES = CategoricalComfort(
    name="Apparent Temperature",
    bin_names=(
        "No risk to health",
        "Prolonged exposure may lead to fatigue",
        "Prolonged exposure may lead to heatstroke",
        "Risk to health, heatstroke imminent",
    ),
    bins=(-np.inf, 27, 32, 56, np.inf),
    colors=("#ccffff", "#00ff00", "#ffcc00", "#ff0000"),
    comfort_classes=[
        ComfortClass.COMFORTABLE,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
        ComfortClass.TOO_HOT,
    ],
)

CLO_VALUE_CATEGORIES = Categorical(
    name="CLO Value",
    bin_names=(
        "Naked",
        "Swimwear",
        "Light summer clothing",
        "Normal summer clothing",
        "Light winter clothing",
        "Normal winter clothing",
        "Arctic clothing",
    ),
    bins=(-np.inf, 0.1, 0.35, 0.55, 0.75, 1.5, 2, np.inf),
    colors=("#f7f4f9", "#e1d4e8", "#cda0cd", "#df64af", "#df2179", "#a90649", "#67001f"),
)

METABOLIC_RATE_CATEGORIES = Categorical(
    name="Metabolic Rate",
    bins=(
        -np.inf,
        0.8,
        1,
        1.6,
        1.9,
        3.4,
        4,
        6.2,
        9.5,
        np.inf,
    ),
    bin_names=(
        "Seated relaxed",
        "Standing at rest",
        "Standing, light activity",
        "Walking, 2km/h",
        "Walking, 5km/h",
        "Cycling, 15km/h",
        "Cycling, 20km/h",
        "Running, 15km/h",
        "Maximum effort",
    ),
    colors=[
        "#ffffff",
        "#f4f4dd",
        "#e9e9b5",
        "#ddcca4",
        "#d0ab93",
        "#c2817f",
        "#a06767",
        "#724949",
        "#1e0000",
    ],
)
