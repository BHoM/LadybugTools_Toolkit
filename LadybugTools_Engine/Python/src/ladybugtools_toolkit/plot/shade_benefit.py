"""Protoype shade benefit plot."""
# pylint: disable=line-too-long
# pylint: disable=E0401
import calendar
from pathlib import Path
import warnings

# pylint: enable=E0401

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import Location
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .utilities import contrasting_color


from ..external_comfort.utci import shade_benefit_category
from ..helpers import sunrise_sunset
from ._heatmap import heatmap


def utci_shade_benefit(
    unshaded_utci: HourlyContinuousCollection | pd.Series,
    shaded_utci: HourlyContinuousCollection | pd.Series,
    comfort_limits: tuple[float] = (9, 26),
    location: Location = None,
    color_config: dict[str, str] = None,
    figsize: tuple[float] = (15, 5),
) -> plt.Figure:
    """Plot the shade benefit category.

    Args:
        unshaded_utci (HourlyContinuousCollection | pd.Series):
            A dataset containing unshaded UTCI values.
        shaded_utci (HourlyContinuousCollection | pd.Series):
            A dataset containing shaded UTCI values.
        comfort_limits (tuple[float], optional):
            The range within which "comfort" is achieved. Defaults to (9, 26).
        location (Location, optional):
            A location object used to plot sun up/down times. Defaults to None.
        color_config (dict[str, str], optional):
            A dictionary of colors for each category. Defaults to None.

    Returns:
        plt.Figure:
            A figure object.
    """

    warnings.warn(
        "This method is not fully formed, and needs to be updated to just be better overall!"
    )

    if color_config is None:
        color_config = {
            "Comfortable with shade": "blue",
            "Comfortable without shade": "green",
            "Shade is beneficial": "orange",
            "Shade is detrimental": "red",
            "Undefined": "grey",
            "Sun up": "k",
        }

    utci_shade_benefit_categories = shade_benefit_category(
        unshaded_utci=unshaded_utci,
        shaded_utci=shaded_utci,
        comfort_limits=comfort_limits,
    )
    cat = pd.Series(
        pd.Categorical(utci_shade_benefit_categories),
        index=utci_shade_benefit_categories.index,
    )
    numeric = cat.cat.codes

    colors = [
        color_config[i]
        for i in [
            "Comfortable with shade",
            "Comfortable without shade",
            "Shade is beneficial",
            "Shade is detrimental",
            "Undefined",
        ]
    ]
    cmap = ListedColormap(colors)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(
        ncols=1, nrows=2, width_ratios=[1], height_ratios=[5, 2], hspace=0.0
    )
    heatmap_ax = fig.add_subplot(spec[0, 0])
    histogram_ax = fig.add_subplot(spec[1, 0])

    # Add heatmap
    hmap = heatmap(numeric, ax=heatmap_ax, show_colorbar=False, cmap=cmap)

    # add categorical
    us = cat.groupby(cat.index.month).value_counts().unstack()
    t = us.divide(us.sum(axis=1), axis=0)
    t.plot(
        ax=histogram_ax,
        kind="bar",
        stacked=True,
        color=colors,
        width=1,
        legend=False,
    )
    histogram_ax.set_xlim(-0.5, len(t) - 0.5)
    histogram_ax.set_ylim(0, 1)
    histogram_ax.set_xticklabels(
        [calendar.month_abbr[int(i)] for i in t.index],
        ha="center",
        rotation=0,
    )
    for spine in ["top", "right", "left", "bottom"]:
        histogram_ax.spines[spine].set_visible(False)
    histogram_ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))
    for i, c in enumerate(histogram_ax.containers):
        label_colors = [contrasting_color(i.get_facecolor()) for i in c.patches]
        labels = [f"{v.get_height():0.1%}" if v.get_height() > 0.15 else "" for v in c]
        histogram_ax.bar_label(
            c,
            labels=labels,
            label_type="center",
            color=label_colors[i],
            fontsize="x-small",
        )

    # add sun up indicator lines
    if location is not None:
        ymin = min(hmap.get_ylim())
        sun_rise_set = sunrise_sunset(location=location)
        sunrise = [
            ymin + (((i.time().hour * 60) + (i.time().minute)) / (60 * 24))
            for i in sun_rise_set.sunrise
        ]
        sunset = [
            ymin + (((i.time().hour * 60) + (i.time().minute)) / (60 * 24))
            for i in sun_rise_set.sunset
        ]
        # heatmap_ax.plot(s.index, s.values, zorder=9, c="#F0AC1B", lw=1)
        xx = np.arange(min(heatmap_ax.get_xlim()), max(heatmap_ax.get_xlim()) + 1, 1)
        heatmap_ax.plot(xx, sunrise, zorder=9, c=color_config["Sun up"], lw=1)
        heatmap_ax.plot(xx, sunset, zorder=9, c=color_config["Sun up"], lw=1)

    # add colorbar
    divider = make_axes_locatable(histogram_ax)
    colorbar_ax = divider.append_axes("bottom", size="20%", pad=0.5)
    cb = fig.colorbar(
        mappable=heatmap_ax.get_children()[0],
        cax=colorbar_ax,
        orientation="horizontal",
        drawedges=False,
        extend="both",
    )
    cb.outline.set_visible(False)
    ticks = np.arange(0, 4, 4 / 5 / 2)[1::2]
    cb.set_ticks(
        ticks,
        labels=[
            "Comfortable with shade",
            "Comfortable without shade",
            "Shade is beneficial",
            "Shade is detrimental",
            "Undefined",
        ],
    )

    return fig
