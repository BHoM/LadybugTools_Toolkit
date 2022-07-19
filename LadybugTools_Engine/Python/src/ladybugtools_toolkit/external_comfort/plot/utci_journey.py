from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.interpolate import make_interp_spline


def plot_utci_journey(
    utci_values: List[float],
    names: List[str] = None,
    curve: bool = False,
) -> Figure:
    """Create a figure showing the pseudo-journey between different UTCI conditions at a given time of year

    Args:
        utci_values (float): A list of UTCI values.
        names (List[str], optional): A list of names to label each value with. Defaults to None.
        curve (bool, optional): Whether to plot the pseudo-journey as a spline. Defaults to False.
        show_legend (bool, optional): Set to True to plot the UTCI comfort band legend also.

    Returns:
        Figure: A matplotlib figure object.
    """

    if names:
        if len(utci_values) != len(names):
            raise ValueError("Number of values and names must be equal.")
    else:
        names = [str(i) for i in range(len(utci_values))]

    # Convert collections into series and combine
    df_pit = pd.Series(utci_values, index=names)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    for n, (idx, val) in enumerate(df_pit.items()):
        ax.scatter(n, val, c="white", s=400, zorder=9)
        ax.text(
            n, val, idx, c="k", zorder=10, ha="center", va="center", fontsize="medium"
        )

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
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off

    ax.set_ylabel("UTCI (°C)")
    plt.tight_layout()

    return fig
