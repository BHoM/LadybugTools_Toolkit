from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from honeybee.config import folders as hb_folders
from ladybug.datatype import TYPESDICT
from ladybug.datatype.base import DataTypeBase
from ladybug.epw import EPW
from ladybugtools_toolkit.categorical.categories import (
    ACTUAL_SENSATION_VOTE_CATEGORIES,
    APPARENT_TEMPERATURE_CATEGORIES,
    BEAUFORT_CATEGORIES,
    CLO_VALUE_CATEGORIES,
    DISCOMFORT_INDEX_CATEGORIES,
    HEAT_INDEX_CATEGORIES,
    HUMIDEX_CATEGORIES,
    METABOLIC_RATE_CATEGORIES,
    PET_CATEGORIES,
    SET_CATEGORIES,
    THERMAL_SENSATION_CATEGORIES,
    UTCI_DEFAULT_CATEGORIES,
    WBGT_CATEGORIES,
    WIND_CHILL_CATEGORIES,
    Categorical,
)
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    ListedColormap,
    Normalize,
    hex2color,
    rgb2hex,
    to_hex,
    to_rgb,
)
from matplotlib.legend import Legend

from ..helpers import average_color

DATA_DIR = Path(hb_folders.default_simulation_folder) / "interactive_thermal_comfort_data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

PERSON_HEIGHT = 1.65
PERSON_AGE = 36
PERSON_SEX = 0.5
PERSON_HEIGHT = 1.65
PERSON_MASS = 62
PERSON_POSITION = "standing"

TERRAIN_ROUGHNESS_LENGTH = 0.03
ATMOSPHERIC_PRESSURE = 101325


# def category_to_plotly_colorscale(cat: Categorical) -> list[str]:
#     """Convert a Categorical object to a Plotly colorscale.

#     Args:
#         cat: A Categorical object.

#     Returns:
#         A list of lists where each sublist contains a value between 0 and 1 and a color
#         in the format "rgb(r, g, b)".
#     """

#     # create finite bins if not finite already
#     if np.isinf(cat.bins[0]):
#         low = cat.bins[1] - 10
#     else:
#         low = cat.bins[0] + 0.00000001
#     if np.isinf(cat.bins[-1]):
#         high = cat.bins[-2] + 10
#     else:
#         high = cat.bins[-2]

#     # generate values and colors
#     vals = np.linspace(low, high, 101).tolist()
#     vals_normed = np.linspace(0, 1, 101).tolist()
#     colors = np.vectorize(cat.get_color)(vals).tolist()
#     colors_fmt = [
#         f"rgb{tuple((np.array(hex2color(color)) * 255).astype(int).tolist())}" for color in colors
#     ]

#     return [[i, j] for i, j in list(zip(*[vals_normed, colors_fmt]))]


# @dataclass
# class ParcoordColorscaleBase:
#     """Configuration options for colouring of variables in a Plotly parcoord plot."""

#     low_lim: float
#     high_lim: float
#     colorscale: list[Any] | str

#     @classmethod
#     def from_categorical(cls, cat: Categorical) -> "ParcoordColorscaleBase":
#         if np.isinf(cat.bins[0]):
#             low = cat.bins[1] - 10
#         else:
#             low = cat.bins[0]
#         if np.isinf(cat.bins[-1]):
#             high = cat.bins[-2] + 10
#         else:
#             high = cat.bins[-2]
#         colorscale = category_to_plotly_colorscale(cat)
#         return cls(low_lim=low, high_lim=high, colorscale=colorscale)


# class ParcoordColorscale(Enum):
#     Tair = ParcoordColorscaleBase(low_lim=-10, high_lim=48, colorscale="viridis")
#     UTCI = ParcoordColorscaleBase.from_categorical(UTCI_DEFAULT_CATEGORIES)
#     vair = ParcoordColorscaleBase.from_categorical(BEAUFORT_CATEGORIES)
#     RH = ParcoordColorscaleBase(low_lim=0, high_lim=100, colorscale="Blues")
#     Esolar = ParcoordColorscaleBase(low_lim=0, high_lim=1400, colorscale="Wistia")
#     MRT = ParcoordColorscaleBase(low_lim=-10, high_lim=100, colorscale="YlOrRd")
#     HDX = ParcoordColorscaleBase.from_categorical(HUMIDEX_CATEGORIES)
#     WBGT = ParcoordColorscaleBase.from_categorical(WBGT_CATEGORIES)
#     HIT = ParcoordColorscaleBase.from_categorical(HEAT_INDEX_CATEGORIES)
#     WCT = ParcoordColorscaleBase.from_categorical(WIND_CHILL_CATEGORIES)
#     PET = ParcoordColorscaleBase.from_categorical(PET_CATEGORIES)
#     SET = ParcoordColorscaleBase.from_categorical(SET_CATEGORIES)
#     TS = ParcoordColorscaleBase.from_categorical(THERMAL_SENSATION_CATEGORIES)
#     AT = ParcoordColorscaleBase.from_categorical(APPARENT_TEMPERATURE_CATEGORIES)
#     CLO = ParcoordColorscaleBase.from_categorical(CLO_VALUE_CATEGORIES)
#     MET = ParcoordColorscaleBase.from_categorical(METABOLIC_RATE_CATEGORIES)
#     ASV = ParcoordColorscaleBase.from_categorical(ACTUAL_SENSATION_VOTE_CATEGORIES)
#     DT = ParcoordColorscaleBase.from_categorical(DISCOMFORT_INDEX_CATEGORIES)

# def generate_epw_comfort_parcoord(epw: EPW, variable: )
