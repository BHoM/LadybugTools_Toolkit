"""Method to plot the seasonality of an EPW file."""

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from ladybug.epw import EPW
from matplotlib.collections import PatchCollection

from ..ladybug_extension.epw import (seasonality_from_day_length,
                                     seasonality_from_month,
                                     seasonality_from_temperature)
from .utilities import contrasting_color


def seasonality_comparison(epw: EPW, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """_"""

    d = {
        "Winter": kwargs.pop("winter_color", "#8DB9CA"),
        "Summer": kwargs.pop("summer_color", "#E6484D"),
        "Autumn": kwargs.pop("autumn_color", "#EE7837"),
        "Spring": kwargs.pop("spring_color", "#AFc1A2"),
    }
    ywidth = 0.9

    if ax is None:
        ax = plt.gca()

    from_day_length = seasonality_from_day_length(epw=epw).rename("From day-length", inplace=False)
    from_month = seasonality_from_month(epw=epw).rename("From month", inplace=False)
    from_temperature = seasonality_from_temperature(epw=epw).rename("From temperature", inplace=False)

    season_df = pd.concat(
        [
            from_month,
            from_day_length,
            from_temperature,
        ],
        axis=1,
    )
    season_df.index = [mdates.date2num(i) for i in season_df.index]

    y = (1 - ywidth) / 2
    patches = []
    for col in season_df.columns:
        for season in ["Winter", "Spring", "Summer", "Autumn"]:
            local = season_df[col][season_df[col] == season]
            if any(local.index.diff().unique() > 1):
                # get the points at which the values change
                shiftpt = local.index.diff().argmax()
                patches.append(
                    mpatches.Rectangle(
                        xy=(local.index[0], y),
                        height=ywidth,
                        width=local.index[shiftpt - 1] - local.index[0],
                        facecolor=d[season],
                        edgecolor="w",
                    )
                )
                patches.append(
                    mpatches.Rectangle(
                        xy=(local.index[shiftpt], y),
                        height=ywidth,
                        width=local.index[-1],
                        facecolor=d[season],
                        edgecolor="w",
                    )
                )
            else:
                patches.append(
                    mpatches.Rectangle(
                        xy=(local.index[0], y),
                        height=ywidth,
                        width=local.index[-1] - local.index[0],
                        facecolor=d[season],
                        edgecolor="w",
                    )
                )
        y += 1
    pc = PatchCollection(patches=patches, match_original=True, zorder=3)
    ax.add_collection(pc)

    # add annotations
    y = (1 - ywidth) / 2
    for n, col in enumerate(season_df.columns):
        for season in ["Winter", "Spring", "Summer", "Autumn"]:
            local = season_df[col][season_df[col] == season]
            if any(local.index.diff().unique() > 1):
                # get the points at which the values change
                shiftpt = local.index.diff().argmax()
                ax.text(
                    local.index[shiftpt] + 1,
                    n + 0.5,
                    f"{mdates.num2date(local.index[shiftpt]):%b %d}",
                    ha="left",
                    va="top",
                    c=contrasting_color(d[season]),
                )
            else:
                ax.text(
                    local.index[0] + 1,
                    n + 0.5,
                    f"{mdates.num2date(local.index[0]):%b %d}",
                    ha="left",
                    va="top",
                    c=contrasting_color(d[season]),
                )
        y += 1

    ax.set_xlim(season_df.index[0], season_df.index[-1])
    ax.set_ylim(0, 3)

    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(season_df.columns, rotation=0, ha="right")

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%B"))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left")

    # create and add legend
    new_handles = []
    for _, color in d.items():
        new_handles.append(mpatches.Patch(color=color, edgecolor=None))

    plt.legend(new_handles, d.keys(), bbox_to_anchor=(0.5, -0.12), loc="upper center", ncol=4)

    return ax
