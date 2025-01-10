# import numpy as np
# from ladybugtools_toolkit.interactive.data_config import EnvironmentVariable, average_color
# from matplotlib.colors import (
#     BoundaryNorm,
#     Colormap,
#     LinearSegmentedColormap,
#     ListedColormap,
#     Normalize,
#     is_color_like,
#     to_hex,
# )

# CAT_NAME = "CategoricalExample"
# CAT_ABBREVIATION = "CAT"
# CAT_UNIT = "cats"
# CAT_COLORS = ["red", "orange", "yellow", "green"]
# CAT_BOUNDARIES = [-100, -50, 0, 50, 100]
# CAT_MIDPTS = [-75, -25, 25, 75]
# CAT_CATEGORIES = ["A", "B", "C", "D"]
# CAT_CMAP = ListedColormap(colors=CAT_COLORS)
# CAT_NORM = BoundaryNorm(boundaries=CAT_BOUNDARIES, clip=True, ncolors=len(CAT_COLORS))

# CONT_NAME = "ContinuousExample"
# CONT_ABBREVIATION = "CONT"
# CONT_UNIT = "conts"
# CONT_CMAP = LinearSegmentedColormap.from_list(colors=["white", "black"])
# CONT_NORM = Normalize(vmin=-100, vmax=100)


# def test_ev_cont():
#     """_"""

#     ev_cont = EnvironmentVariable(
#         abbreviation=CONT_ABBREVIATION,
#         name=CONT_NAME,
#         colormap=CONT_CMAP,
#         norm=CONT_NORM,
#         unit=CONT_UNIT,
#     )

#     # test properties
#     assert isinstance(ev_cont, EnvironmentVariable)
#     assert ev_cont.abbreviation == CONT_ABBREVIATION
#     assert ev_cont.name == CONT_NAME
#     assert isinstance(ev_cont.colormap, Colormap)
#     assert isinstance(ev_cont.norm, Normalize)
#     assert ev_cont.unit == CONT_UNIT
#     assert ev_cont._is_categorical() == False

#     # test computed properties
#     assert ev_cont.colormap.name == ev_cont.name
#     assert ev_cont._interval_values() == np.linspace(CONT_NORM.vmin, CONT_NORM.vmax, 11).tolist()
#     assert ev_cont._min() == CONT_NORM.vmin
#     assert ev_cont._max() == CONT_NORM.vmax
#     assert np.isinf(ev_cont._possible_intervals())


# def test_ev_cont_get_colors():
#     """_"""
#     ev_cont = EnvironmentVariable(
#         abbreviation=CONT_ABBREVIATION,
#         name=CONT_NAME,
#         colormap=CONT_CMAP,
#         norm=CONT_NORM,
#         unit=CONT_UNIT,
#     )

#     _N = 11

#     colors = ev_cont.get_colors(n_intervals=_N, return_type="hex")
#     assert len(set(colors)) == _N
#     for color in colors:
#         assert is_color_like(color)

#     colors = ev_cont.get_colors(n_intervals=_N, return_type="rgb_int")
#     assert len(set(colors)) == _N
#     for color in colors:
#         assert is_color_like(color)

#     colors = ev_cont.get_colors(n_intervals=_N, return_type="rgb_float")
#     assert len(set(colors)) == _N
#     for color in colors:
#         assert is_color_like(color)


# def test_ev_cat():
#     """_"""

#     ev_cat = EnvironmentVariable(
#         abbreviation=CAT_ABBREVIATION,
#         name=CAT_NAME,
#         categories=CAT_CATEGORIES,
#         colormap=CAT_CMAP,
#         norm=CAT_NORM,
#         unit=CAT_UNIT,
#     )

#     # test properties
#     assert isinstance(ev_cat, EnvironmentVariable)
#     assert ev_cat.abbreviation == CAT_ABBREVIATION
#     assert ev_cat.name == CAT_NAME
#     assert ev_cat.categories == CAT_CATEGORIES
#     assert isinstance(ev_cat.colormap, Colormap)
#     assert isinstance(ev_cat.norm, BoundaryNorm)
#     assert ev_cat.unit == CAT_UNIT
#     assert ev_cat._is_categorical() == True

#     # test computed properties
#     assert ev_cat.colormap.name == ev_cat.name
#     assert ev_cat._interval_values() == CAT_MIDPTS
#     assert ev_cat._min() == CAT_BOUNDARIES[0] == CAT_NORM.vmin
#     assert ev_cat._max() == CAT_BOUNDARIES[-1] == CAT_NORM.vmax
#     assert ev_cat._possible_intervals() == len(CAT_CATEGORIES)


# def test_ev_cat_get_colors():
#     """_"""
#     ev_cat = EnvironmentVariable(
#         abbreviation=CAT_ABBREVIATION,
#         name=CAT_NAME,
#         categories=CAT_CATEGORIES,
#         colormap=CAT_CMAP,
#         norm=CAT_NORM,
#         unit=CAT_UNIT,
#     )

#     assert ev_cat.get_colors(n_intervals=len(CAT_CATEGORIES), return_type="hex") == [
#         to_hex(i) for i in CAT_COLORS
#     ]
#     assert ev_cat.get_colors(n_intervals=len(CAT_CATEGORIES) + 1, return_type="hex") == [
#         to_hex(i) for i in CAT_COLORS
#     ]


# def test_categorical_get_value_color():
#     """_"""
#     ev_cat = EnvironmentVariable(
#         abbreviation=CAT_ABBREVIATION,
#         name=CAT_NAME,
#         categories=CAT_CATEGORIES,
#         colormap=CAT_CMAP,
#         norm=CAT_NORM,
#         unit=CAT_UNIT,
#     )

#     assert ev_cat.get_value_color(
#         value=np.mean(CAT_BOUNDARIES[0:2]), return_type="hex", interpolate_categorical=True
#     ) == average_color(colors=CAT_COLORS[0:2])
#     assert ev_cat.get_value_color(
#         value=np.mean(CAT_BOUNDARIES[0:2]), return_type="hex", interpolate_categorical=False
#     ) == to_hex(CAT_COLORS[0])


# def test_categorical_get_value_category():
#     """_"""
#     ev_cat = EnvironmentVariable(
#         abbreviation=CAT_ABBREVIATION,
#         name=CAT_NAME,
#         categories=CAT_CATEGORIES,
#         colormap=CAT_CMAP,
#         norm=CAT_NORM,
#         unit=CAT_UNIT,
#     )

#     assert ev_cat.get_value_category(value=np.mean(CAT_BOUNDARIES[1:3])) == CAT_CATEGORIES[2]
