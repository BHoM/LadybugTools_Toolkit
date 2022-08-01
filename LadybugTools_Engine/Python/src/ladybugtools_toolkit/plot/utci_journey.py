from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ladybugtools_toolkit.plot.colormaps import (
    UTCI_BOUNDARYNORM,
    UTCI_COLORMAP,
    UTCI_LABELS,
    UTCI_LEVELS,
)
from ladybugtools_toolkit.plot.lighten_color import lighten_color
from matplotlib.colors import rgb2hex
from matplotlib.figure import Figure
from scipy.interpolate import make_interp_spline


def utci_journey(
    utci_values: List[float],
    names: List[str] = None,
    curve: bool = False,
    ylims: Tuple = None,
    show_legend: bool = False,
    title: str = None,
    show_grid: bool = False,
) -> Figure:
    """Create a figure showing the pseudo-journey between different UTCI conditions at a given time of year

    Args:
        utci_values (float):
            A list of UTCI values.
        names (List[str], optional):
            A list of names to label each value with. Defaults to None.
        curve (bool, optional):
            Whether to plot the pseudo-journey as a spline. Defaults to False.
        ylims (Tuple[float], optional):
            A user specified upper/lower value to apply to the y-axis.
        show_legend (bool, optional):
            Set to True to plot the UTCI comfort band legend also.
        title (str, optional):
            Add a title to the plot.
        show_grid (bool, optional):
            Set to True to include a grid on the plot.

    Returns:
        Figure: A matplotlib figure object.
    """

    if names:
        if len(utci_values) != len(names):
            raise ValueError("Number of values and names must be equal.")
    else:
        names = [str(i) for i in range(len(utci_values))]

    if (ylims is not None) and len(ylims) != 2:
        raise ValueError("ylims must be a list/tuple of size 2.")

    # Convert collections into series and combine
    df_pit = pd.Series(utci_values, index=names)

    # instantiate figure
    fig, ax = plt.subplots(figsize=(10, 2.5))

    def _rolling_window(array: List[Any], window: int):
        """Throwaway function here to roll a window along a list.

        Args:
            array (List[Any]):
                A 1D list of some kind.
            window (int):
                The size of the window to apply to the list.

        Returns:
            List[List[Any]]:
                The resulting, "windowed" list.
        """
        a = np.array(array)
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    # Add UTCI background colors to the canvas
    background_colors = np.array(
        [rgb2hex(UTCI_COLORMAP.get_under())]
        + UTCI_COLORMAP.colors
        + [rgb2hex(UTCI_COLORMAP.get_over())]
    )
    background_ranges = _rolling_window(np.array([-100] + UTCI_LEVELS + [100]), 2)
    for bg_color, (bg_start, bg_end), label in list(
        zip(*[background_colors, background_ranges, UTCI_LABELS])
    ):
        ax.axhspan(
            bg_start, bg_end, facecolor=lighten_color(bg_color, 0.3), label=label
        )

    # add UTCI instance values to canvas
    for n, (idx, val) in enumerate(df_pit.items()):
        ax.scatter(n, val, c="white", s=400, zorder=9)
        ax.text(
            n, val, idx, c="k", zorder=10, ha="center", va="center", fontsize="medium"
        )

    if show_grid:
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(-100, 101, 10)
        minor_ticks = np.arange(-100, 101, 5)

        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which="major", c="w", alpha=0.25, ls="--", axis="y")
        ax.grid(which="minor", c="w", alpha=0.75, ls=":", axis="y")

    # set ylims
    if ylims is None:
        ax.set_ylim(min(df_pit) - 5, max(df_pit) + 5)
    else:
        ax.set_ylim(ylims[0], ylims[1])

    if curve:
        # Smooth values
        if len(utci_values) < 3:
            k = 1
        else:
            k = 2
        x = np.arange(len(utci_values))
        y = df_pit.values
        xnew = np.linspace(min(x), max(x), 300)
        bspl = make_interp_spline(x, y, k=k)
        ynew = bspl(xnew)

        # Plot the smoothed values
        ax.plot(xnew, ynew, c="#B30202", ls="--", zorder=3)

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    plt.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
    )

    ax.set_ylabel("UTCI (°C)")

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(
            handles[::-1],
            labels[::-1],
            bbox_to_anchor=(1, 1),
            loc=2,
            ncol=1,
            borderaxespad=0,
            frameon=False,
            fontsize="small",
        )
        lgd.get_frame().set_facecolor((1, 1, 1, 0))

    if title is not None:
        ax.set_title(
            title,
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )

    plt.tight_layout()

    return fig
