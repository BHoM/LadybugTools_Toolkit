from typing import List, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from ..helpers import cardinality, rolling_window
from .annotate_imshow import annotate_imshow


def wind_speed_direction_frequency(
    wind_speed: List[float],
    wind_direction: List[float],
    speed_bins: List[float] = None,
    n_directions: float = 8,
    cmap: Union[Colormap, str] = "Purples",
    title: str = None,
) -> Figure:
    """Create an image containing a matrix of wind speed/direction frequencies.

    Args:
        wind_speed (List[float]):
            A list if wind speeds to process.
        wind_direction (List[float]):
            A list of wind directions to process.
        speed_bins (List[float], optional):
            A set of bins into which wind speed will be sorted. Defaults to None which divides the
            range 0-MAX into 10 bins.
        n_directions (float, optional):
            A number of cardinal directions into which wind_direction will be grouped.
            Default is 8.
        cmap (Union[Colormap, str], optional):
            The colormap to use in this heatmap. Defaults to "Purples".
        title (str, optional):
            An additional title to give this figure. Defaults to None.

    Returns:
        Figure:
            A matplotlib figure object.
    """

    if len(wind_speed) != len(wind_direction):
        raise ValueError("The given wind speeds and directions do not align.")

    if speed_bins is None:
        speed_bins = np.linspace(0, max(wind_speed), 11)

    # create "bins" of direction based on number of directions to bin into
    direction_bin_width = 360 / n_directions
    direction_bins = (
        [0]
        + (np.linspace(0, 360, n_directions + 1) + (direction_bin_width / 2)).tolist()[
            :-1
        ]
        + [360]
    )

    # bin data into table
    binned, speed_bins, direction_bins = np.histogram2d(
        wind_speed, wind_direction, bins=[speed_bins, direction_bins], normed=False
    )
    df = pd.DataFrame(binned)
    # combine first and last column, and drop last column (to get North winds)
    df.iloc[:, 0] = df.iloc[:, 0] + df.iloc[:, -1]
    df = df.iloc[:, :-1]

    speed_index = [(i[0], i[1]) for i in rolling_window(speed_bins, 2)]
    df.index = speed_index

    if n_directions in [4, 8, 16, 32]:
        direction_index = [
            cardinality(i, n_directions)
            for i in [0] + rolling_window(direction_bins[1:-1], 2).mean(axis=1).tolist()
        ]
    else:
        direction_index = [
            (i[0], i[1])
            for i in np.roll(
                rolling_window(direction_bins[1:-1].tolist() + [direction_bins[1]], 2),
                1,
                axis=0,
            )
        ]
    df.columns = direction_index
    df = df / len(wind_speed)

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.set_aspect("auto")
    im = ax.imshow(df.values, cmap=cmap)

    annotate_imshow(im, valfmt="{x:.02%}", fontsize="x-small", exclude_vals=[0])

    ax.xaxis.set_major_locator(mticker.FixedLocator(range(len(df.columns))))

    if isinstance(df.columns[0], str):
        xticklabels = df.columns
    else:
        xticklabels = [f"{i[0]:0.1f}-\n{i[1]:0.1f}" for i in df.columns]
    ax.set_xticklabels(
        xticklabels,
        rotation=0,
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.set_xlabel("Wind direction (degrees)")

    ax.yaxis.set_major_locator(mticker.FixedLocator(range(len(df.index))))
    yticklabels = [f"{i[0]}-{i[1]}" for i in df.index]
    ax.set_yticklabels(yticklabels, rotation=0, ha="right", va="center", fontsize=8)
    ax.set_ylabel("Wind speed (m/s)")

    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(False)

    ti = "Wind speed/direction frequency"
    if title:
        ti += f"\n{title}"
    ax.set_title(ti, x=0, ha="left")

    plt.tight_layout()

    return fig
