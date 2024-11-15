"""Method to plot the seasonality of an EPW file."""

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from ladybug.epw import EPW
from matplotlib.collections import PatchCollection

from ..ladybug_extension.epw import (
    seasonality_from_day_length,
    seasonality_from_month,
    seasonality_from_temperature,
)
from .utilities import contrasting_color


def seasonality_comparison(
    epw: EPW, ax: plt.Axes = None, color_config: dict[str, str] = None, **kwargs
) -> plt.Axes:
    """Create a plot which shows where the seasohn threshholds are for the input EPW object.

    Args:
        epw (EPW):
            An EPW object.
        ax (plt.Axes, optional):
            A matplotlib axes object. Default is None which uses the current axes.
        color_config (dict[str, str], optional):
            A dictionary of colors for each season. If None, then default will be used.
        **kwargs:
            title (str):
                The title of the plot. If not provided, then the name of the EPW file is used.
    """

    from_day_length = seasonality_from_day_length(epw=epw).rename("From day-length", inplace=False)
    from_month = seasonality_from_month(epw=epw).rename("From month", inplace=False)
    from_temperature = seasonality_from_temperature(epw=epw).rename(
        "From temperature", inplace=False
    )

    seasons = ["Winter", "Spring", "Summer", "Autumn"]

    if color_config is None:
        color_config = {
            "Winter": "#8DB9CA",
            "Spring": "#AFC1A2",
            "Summer": "#E6484D",
            "Autumn": "#EE7837",
        }
    else:
        if [i not in color_config for i in seasons]:
            raise ValueError(
                f"The color_config dictionary must contain colors for all four seasons {seasons}."
            )

    title_str = kwargs.pop("title", str(epw))

    ywidth = 0.9

    if ax is None:
        ax = plt.gca()

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
        for season in seasons:
            local = season_df[col][season_df[col] == season]
            if any(local.index.diff().unique() > 1):
                # get the points at which the values change
                shiftpt = local.index.diff().argmax()
                patches.append(
                    mpatches.Rectangle(
                        xy=(local.index[0], y),
                        height=ywidth,
                        width=local.index[shiftpt - 1] - local.index[0],
                        facecolor=color_config[season],
                        edgecolor="w",
                    )
                )
                patches.append(
                    mpatches.Rectangle(
                        xy=(local.index[shiftpt], y),
                        height=ywidth,
                        width=local.index[-1],
                        facecolor=color_config[season],
                        edgecolor="w",
                    )
                )
            else:
                patches.append(
                    mpatches.Rectangle(
                        xy=(local.index[0], y),
                        height=ywidth,
                        width=local.index[-1] - local.index[0],
                        facecolor=color_config[season],
                        edgecolor="w",
                    )
                )
        y += 1
    pc = PatchCollection(patches=patches, match_original=True, zorder=3)
    ax.add_collection(pc)

    # add annotations
    y = (1 - ywidth) / 2
    for n, col in enumerate(season_df.columns):
        for season in seasons:
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
                    c=contrasting_color(color_config[season]),
                )
            else:
                ax.text(
                    local.index[0] + 1,
                    n + 0.5,
                    f"{mdates.num2date(local.index[0]):%b %d}",
                    ha="left",
                    va="top",
                    c=contrasting_color(color_config[season]),
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
    for _, color in color_config.items():
        new_handles.append(mpatches.Patch(color=color, edgecolor=None))
    plt.legend(
        new_handles, color_config.keys(), bbox_to_anchor=(0.5, -0.12), loc="upper center", ncol=4
    )

    # add title
    ax.set_title(title_str)

    return ax
