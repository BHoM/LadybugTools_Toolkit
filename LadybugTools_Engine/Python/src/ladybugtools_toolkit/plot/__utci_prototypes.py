"""Protoype UTCI plotting methods."""
# pylint: disable=line-too-long
# # pylint: disable=E0401
# import calendar
# import textwrap
# import warnings
# from calendar import month_abbr
# from datetime import timedelta

# # pylint: enable=E0401

# import matplotlib.dates as mdates
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import numpy as np
# import pandas as pd
# from ladybug.datacollection import HourlyContinuousCollection
# from ladybug.datatype.temperature import UniversalThermalClimateIndex
# from ladybug.epw import EPW
# from ladybug.sunpath import Sunpath
# from matplotlib.colorbar import ColorbarBase
# from matplotlib.colors import BoundaryNorm, ListedColormap
# from matplotlib.figure import Figure
# from matplotlib.gridspec import GridSpec
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# from ..categorical.categories import Categorical

# # from ..external_comfort.utci import (
# #     UTCICategories,
# #     feasible_comfort_category,
# #     utci_comfort_categories,
# # )
# from ..external_comfort.utci import shade_benefit_category
# from ..helpers import sunrise_sunset
# from ..ladybug_extension.analysisperiod import describe_analysis_period
# from ..ladybug_extension.datacollection import collection_to_series
# from ._heatmap import heatmap





# # def utci_feasibility(
# #     epw: EPW,
# #     simplified: bool = False,
# #     comfort_limits: tuple = (9, 26),
# #     included_additional_moisture: bool = False,
# #     analysis_periods: Union[AnalysisPeriod, tuple[AnalysisPeriod]] = (AnalysisPeriod()),
# #     met_rate_adjustment: float = None,
# # ) -> Figure:
# #     """Plot the UTCI feasibility for each month of the year.

# #     Args:
# #         epw (EPW):
# #             An EPW object.
# #         simplified (bool, optional):
# #             Default is False.
# #         comfort_limits (tuple, optional):
# #             Default is (9, 26). Only used if simplified is True.
# #         included_additional_moisture (bool, optional):
# #             Default is False. If True, then include evap cooling in this analysis.
# #         analysis_periods (Union[AnalysisPeriod, tuple[AnalysisPeriod]], optional):
# #             An AnalysisPeriod or a tuple of AnalysisPeriods to be used for the analysis.
# #             Defaults to (AnalysisPeriod(),).
# #         met_rate_adjustment (float, optional):
# #             A value to be added to the metabolic rate of the UTCI model. This can be used
# #             to account for changes in metabolic rate due to clothing, exercise, etc.
# #             Defaults to None.
# #     Returns:
# #         Figure:
# #             A matplotlib Figure object.
# #     """

# #     df = feasible_comfort_category(
# #         epw,
# #         simplified=simplified,
# #         comfort_limits=comfort_limits,
# #         include_additional_moisture=included_additional_moisture,
# #         analysis_periods=analysis_periods,
# #         met_rate_adjustment_value=met_rate_adjustment,
# #     )

# #     labels, _ = utci_comfort_categories(
# #         simplified=simplified,
# #         comfort_limits=comfort_limits,
# #         rtype="category",
# #     )
# #     colors, _ = utci_comfort_categories(
# #         simplified=simplified,
# #         comfort_limits=comfort_limits,
# #         rtype="color",
# #     )

# #     fig, axes = plt.subplots(1, 12, figsize=(10, 4), sharey=True, sharex=False)

# #     ypos = range(len(df))
# #     for n, ax in enumerate(axes):
# #         # get values
# #         low = df.iloc[n].filter(regex="lowest")
# #         high = df.iloc[n].filter(regex="highest")
# #         ypos = range(len(low))

# #         ax.barh(
# #             ypos,
# #             width=high.values - low.values,
# #             left=low.values,
# #             color=colors,
# #             zorder=3,
# #             alpha=0.8,
# #         )

# #         for rect in ax.patches:
# #             width = rect.get_width()
# #             height = rect.get_height()
# #             _x = rect.get_x()
# #             _y = rect.get_y()
# #             if width == 0:
# #                 if _x == 1:
# #                     # text saying 100% of hours are in this category
# #                     ax.text(
# #                         0.5,
# #                         _y + (height / 2),
# #                         textwrap.fill("All times", 15),
# #                         ha="center",
# #                         va="center",
# #                         rotation=0,
# #                         fontsize="xx-small",
# #                         zorder=3,
# #                     )
# #                 continue

# #             ax.text(
# #                 _x - 0.03,
# #                 _y + (height),
# #                 f"{_x:0.1%}",
# #                 ha="right",
# #                 va="top",
# #                 rotation=90,
# #                 fontsize="xx-small",
# #                 zorder=3,
# #             )
# #             ax.text(
# #                 _x + width + 0.03,
# #                 _y + (height),
# #                 f"{_x + width:0.1%}",
# #                 ha="left",
# #                 va="top",
# #                 rotation=90,
# #                 fontsize="xx-small",
# #                 zorder=3,
# #             )

# #         if simplified:
# #             for nn, i in enumerate(colors):
# #                 ax.axhspan(ymin=nn - 0.5, ymax=nn + 0.5, fc=i, alpha=0.2, zorder=1)
# #         else:
# #             for nn, i in enumerate(UTCICategories):
# #                 ax.axhspan(
# #                     ymin=nn - 0.5, ymax=nn + 0.5, fc=i.color, alpha=0.2, zorder=1
# #                 )

# #         ax.set_xlim(-0.1, 1.1)
# #         ax.set_ylim(-0.5, len(ypos) - 0.5)
# #         for spine in ["left", "bottom"]:
# #             ax.spines[spine].set_visible(False)
# #         ax.tick_params(labelleft=False, left=False)
# #         ax.set_xticks([-0.1, 0.5, 1.1])
# #         ax.set_xticklabels(["", month_abbr[n + 1], ""])
# #         ax.grid(False)

# #         if n == 5:
# #             handles = []
# #             if simplified:
# #                 for col, lab in list(zip(*[["#3C65AF", "#2EB349", "#C31F25"], labels])):
# #                     handles.append(mpatches.Patch(color=col, label=lab, alpha=0.3))
# #             else:
# #                 for i in UTCICategories:
# #                     handles.append(
# #                         mpatches.Patch(color=i.color, label=i.value, alpha=0.3)
# #                     )

# #             ax.legend(
# #                 handles=handles,
# #                 bbox_to_anchor=(0.5, -0.1),
# #                 loc="upper center",
# #                 ncol=3 if simplified else 4,
# #                 borderaxespad=0,
# #                 frameon=False,
# #             )

# #         ti = f"{location_to_string(epw.location)}\nFeasible ranges of UTCI temperatures ({describe_ap(analysis_periods)})"
# #         if met_rate_adjustment:
# #             ti += f" with MET rate adjustment to {met_rate_adjustment} MET"
# #         plt.suptitle(
# #             textwrap.fill(ti, 90),
# #             x=0.075,
# #             y=0.9,
# #             ha="left",
# #             va="bottom",
# #         )

# #     plt.tight_layout()
# #     return fig


# def utci_distance_to_comfortable(
#     collection: HourlyContinuousCollection,
#     title: str = None,
#     comfort_thresholds: tuple[float] = (9, 26),
#     low_limit: float = 15,
#     high_limit: float = 25,
# ) -> Figure:
#     """Plot the distance (in C) to comfortable for a given Ladybug HourlyContinuousCollection
#         containing UTCI values.

#     Args:
#         collection (HourlyContinuousCollection):
#             A Ladybug Universal Thermal Climate Index HourlyContinuousCollection object.
#         title (str, optional):
#             A title to place at the top of the plot. Defaults to None.
#         comfort_thresholds (list[float], optional):
#             The comfortable band of UTCI temperatures. Defaults to [9, 26].
#         low_limit (float, optional):
#             The distance from the lower edge of the comfort threshold to include in the "too cold"
#             part of the heatmap. Defaults to 15.
#         high_limit (float, optional):
#             The distance from the upper edge of the comfort threshold to include in the "too hot"
#             part of the heatmap. Defaults to 25.
#     Returns:
#         Figure:
#             A matplotlib Figure object.
#     """

#     if not isinstance(collection.header.data_type, UniversalThermalClimateIndex):
#         raise ValueError("This method only works for UTCI data.")

#     if not len(comfort_thresholds) == 2:
#         raise ValueError("comfort_thresholds must be a list of length 2.")

#     # Create matrices containing the above/below/within UTCI distances to comfortable
#     series = collection_to_series(collection)

#     low, high = comfort_thresholds
#     midpoint = np.mean([low, high])

#     distance_above_comfortable = (series[series > high] - high).to_frame()
#     distance_above_comfortable_matrix = (
#         distance_above_comfortable.set_index(
#             [
#                 distance_above_comfortable.index.dayofyear,
#                 distance_above_comfortable.index.hour,
#             ]
#         )["Universal Thermal Climate Index (C)"]
#         .astype(np.float64)
#         .unstack()
#         .T.reindex(range(24), axis=0)
#         .reindex(range(365), axis=1)
#     )

#     distance_below_comfortable = (low - series[series < low]).to_frame()
#     distance_below_comfortable_matrix = (
#         distance_below_comfortable.set_index(
#             [
#                 distance_below_comfortable.index.dayofyear,
#                 distance_below_comfortable.index.hour,
#             ]
#         )["Universal Thermal Climate Index (C)"]
#         .astype(np.float64)
#         .unstack()
#         .T.reindex(range(24), axis=0)
#         .reindex(range(365), axis=1)
#     )

#     distance_below_midpoint = (
#         midpoint - series[(series >= low) & (series <= midpoint)]
#     ).to_frame()
#     distance_below_midpoint_matrix = (
#         distance_below_midpoint.set_index(
#             [
#                 distance_below_midpoint.index.dayofyear,
#                 distance_below_midpoint.index.hour,
#             ]
#         )["Universal Thermal Climate Index (C)"]
#         .astype(np.float64)
#         .unstack()
#         .T.reindex(range(24), axis=0)
#         .reindex(range(365), axis=1)
#     )

#     distance_above_midpoint = (
#         series[(series <= high) & (series > midpoint)] - midpoint
#     ).to_frame()
#     distance_above_midpoint_matrix = (
#         distance_above_midpoint.set_index(
#             [
#                 distance_above_midpoint.index.dayofyear,
#                 distance_above_midpoint.index.hour,
#             ]
#         )["Universal Thermal Climate Index (C)"]
#         .astype(np.float64)
#         .unstack()
#         .T.reindex(range(24), axis=0)
#         .reindex(range(365), axis=1)
#     )

#     distance_above_comfortable_cmap = plt.get_cmap("YlOrRd")  # Reds
#     distance_above_comfortable_lims = [0, high_limit]
#     distance_above_comfortable_norm = BoundaryNorm(
#         np.linspace(
#             distance_above_comfortable_lims[0], distance_above_comfortable_lims[1], 100
#         ),
#         ncolors=distance_above_comfortable_cmap.N,
#         clip=True,
#     )

#     distance_below_comfortable_cmap = plt.get_cmap("YlGnBu")  # Blues
#     distance_below_comfortable_lims = [0, low_limit]
#     distance_below_comfortable_norm = BoundaryNorm(
#         np.linspace(
#             distance_below_comfortable_lims[0], distance_below_comfortable_lims[1], 100
#         ),
#         ncolors=distance_below_comfortable_cmap.N,
#         clip=True,
#     )

#     distance_below_midpoint_cmap = plt.get_cmap("YlGn_r")  # Greens_r
#     distance_below_midpoint_lims = [0, midpoint - low]
#     distance_below_midpoint_norm = BoundaryNorm(
#         np.linspace(
#             distance_below_midpoint_lims[0], distance_below_midpoint_lims[1], 100
#         ),
#         ncolors=distance_below_midpoint_cmap.N,
#         clip=True,
#     )

#     distance_above_midpoint_cmap = plt.get_cmap("YlGn_r")  # Greens_r
#     distance_above_midpoint_lims = [0, high - midpoint]
#     distance_above_midpoint_norm = BoundaryNorm(
#         np.linspace(
#             distance_above_midpoint_lims[0], distance_above_midpoint_lims[1], 100
#         ),
#         ncolors=distance_above_midpoint_cmap.N,
#         clip=True,
#     )

#     extent = [
#         mdates.date2num(series.index.min()),
#         mdates.date2num(series.index.max()),
#         726449,
#         726450,
#     ]

#     fig = plt.figure(constrained_layout=False, figsize=(15, 5))
#     gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[20, 1])
#     hmap_ax = fig.add_subplot(gs[0, :])
#     cb_low_ax = fig.add_subplot(gs[1, 0])
#     cb_mid_ax = fig.add_subplot(gs[1, 1])
#     cb_high_ax = fig.add_subplot(gs[1, 2])

#     hmap_ax.imshow(
#         np.ma.array(
#             distance_below_comfortable_matrix,
#             mask=np.isnan(distance_below_comfortable_matrix),
#         ),
#         extent=extent,
#         aspect="auto",
#         cmap=distance_below_comfortable_cmap,
#         norm=distance_below_comfortable_norm,
#         interpolation="none",
#     )
#     hmap_ax.imshow(
#         np.ma.array(
#             distance_below_midpoint_matrix,
#             mask=np.isnan(distance_below_midpoint_matrix),
#         ),
#         extent=extent,
#         aspect="auto",
#         cmap=distance_below_midpoint_cmap,
#         norm=distance_below_midpoint_norm,
#         interpolation="none",
#     )
#     hmap_ax.imshow(
#         np.ma.array(
#             distance_above_comfortable_matrix,
#             mask=np.isnan(distance_above_comfortable_matrix),
#         ),
#         extent=extent,
#         aspect="auto",
#         cmap=distance_above_comfortable_cmap,
#         norm=distance_above_comfortable_norm,
#         interpolation="none",
#     )
#     hmap_ax.imshow(
#         np.ma.array(
#             distance_above_midpoint_matrix,
#             mask=np.isnan(distance_above_midpoint_matrix),
#         ),
#         extent=extent,
#         aspect="auto",
#         cmap=distance_above_midpoint_cmap,
#         norm=distance_above_midpoint_norm,
#         interpolation="none",
#     )

#     # Axis formatting
#     hmap_ax.invert_yaxis()
#     hmap_ax.xaxis_date()
#     hmap_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
#     hmap_ax.yaxis_date()
#     hmap_ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
#     hmap_ax.tick_params(labelleft=True, labelright=True, labelbottom=True)
#     plt.setp(hmap_ax.get_xticklabels(), ha="left", color="k")
#     plt.setp(hmap_ax.get_yticklabels(), color="k")

#     # Spine formatting
#     for spine in ["top", "bottom", "left", "right"]:
#         hmap_ax.spines[spine].set_visible(False)

#     # Grid formatting
#     hmap_ax.grid(visible=True, which="major", color="white", linestyle=":", alpha=1)

#     # Colorbars
#     low_cb = ColorbarBase(
#         cb_low_ax,
#         cmap=distance_below_comfortable_cmap,
#         orientation="horizontal",
#         norm=distance_below_comfortable_norm,
#         label='Degrees below "comfortable"',
#         extend="max",
#     )
#     low_cb.outline.set_visible(False)
#     cb_low_ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

#     mid_cb = ColorbarBase(
#         cb_mid_ax,
#         cmap=distance_below_midpoint_cmap,
#         orientation="horizontal",
#         norm=distance_below_midpoint_norm,
#         label='Degrees about "comfortable"',
#         extend="neither",
#     )
#     mid_cb.outline.set_visible(False)
#     cb_mid_ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

#     high_cb = ColorbarBase(
#         cb_high_ax,
#         cmap=distance_above_comfortable_cmap,
#         orientation="horizontal",
#         norm=distance_above_comfortable_norm,
#         label='Degrees above "comfortable"',
#         extend="max",
#     )
#     high_cb.outline.set_visible(False)
#     cb_high_ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

#     if title is None:
#         hmap_ax.set_title(
#             'Distance to "comfortable"', color="k", y=1, ha="left", va="bottom", x=0
#         )
#     else:
#         hmap_ax.set_title(
#             f"Distance to comfortable - {title}",
#             color="k",
#             y=1,
#             ha="left",
#             va="bottom",
#             x=0,
#         )

#     # Tidy plot
#     plt.tight_layout()

#     return fig
