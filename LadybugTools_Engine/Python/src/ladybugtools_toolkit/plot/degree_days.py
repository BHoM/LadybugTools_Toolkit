import calendar
from typing import List, Tuple, Union

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from ladybug.epw import EPW
from matplotlib.figure import Figure

from ..ladybug_extension.epw import degree_time


def _add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing

        # Vertical alignment for positive values
        va = "bottom"

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = "top"

        # Use Y value as label and format number with one decimal place
        label = "{:.0f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(0, space),  # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha="center",  # Horizontally center label
            va=va,
        )  # Vertically align label differently for
        # positive and negative values.


def degree_days(
    epw: EPW,
    heat_base: float = 18,
    cool_base: float = 23,
) -> Figure:
    """Plot the heating/cooling degree days from a given EPW
    object.

    Args:
        epw (EPW):
            An EPW object.
        heat_base (float, optional):
            The temperature at which heating kicks in. Defaults to 18.
        cool_base (float, optional):
            The temperature at which cooling kicks in. Defaults to 23.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if not isinstance(epw, EPW):
        raise ValueError("epw is not an EPW object.")

    temp = degree_time(
        [epw], return_type="days", cool_base=cool_base, heat_base=heat_base
    )

    location_name = temp.columns.get_level_values(0).unique()[0]
    temp = temp.droplevel(0, axis=1).resample("MS").sum()
    temp.index = [calendar.month_abbr[i] for i in temp.index.month]

    clg = temp.filter(regex="Cooling")
    htg = temp.filter(regex="Heating")

    fig, (clg_ax, htg_ax) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # ax.bar(htg.index)
    htg.plot(ax=htg_ax, kind="bar", width=0.9, color="blue", legend=False)
    clg.plot(ax=clg_ax, kind="bar", width=0.9, color="orange", legend=False)

    htg_ax.set_ylabel(htg.columns[0])
    clg_ax.set_ylabel(clg.columns[0])

    for ax in [htg_ax, clg_ax]:
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        ax.grid(
            visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.2
        )
        _add_value_labels(ax)

        ax.text(
            1,
            1,
            f"Annual: {sum([rect.get_height() for rect in ax.patches]):0.0f}",
            transform=ax.transAxes,
            ha="right",
        )
    plt.suptitle(f"{location_name}\nHeating/Cooling degree days", x=0, ha="left")
    plt.tight_layout()

    return fig
