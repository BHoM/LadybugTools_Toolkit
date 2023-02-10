import calendar
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from ladybug.epw import EPW
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure

from ..helpers import wind_direction_average
from ..ladybug_extension.datacollection import to_series
from ..ladybug_extension.epw import filename
from .relative_luminance import contrasting_color, relative_luminance


def wind_matrix(
    wind_speeds: List[List[float]],
    wind_directions: List[List[float]],
    cmap: Union[Colormap, str] = "Spectral_r",
    title: str = None,
    show_values: bool = False,
) -> Figure:
    """Generate a wind-speed/direction matrix from a set of 24*12 speed and direction values.

    Args:
        wind_speeds (List[List[float]]):
            A list of 24*12 wind speeds.
        wind_directions (List[List[float]]):
            A list of 24*12 wind directions (in degrees from north clockwise).
        cmap (Colormap, optional):
            A colormap to apply to the binned data. Defaults to Spectral.
        show_values (bool, optional):
            Show the values in the matrix. Defaults to False.
    Returns:
        Figure:
            A matplotlib figure object.
    """

    if cmap is None:
        cmap = plt.get_cmap("Spectral_r")

    wind_speeds = np.array(wind_speeds)
    wind_directions = np.array(wind_directions)

    if any(
        [
            wind_speeds.shape != (24, 12),
            wind_directions.shape != (24, 12),
            wind_directions.shape != wind_speeds.shape,
        ]
    ):
        raise ValueError("The wind_speeds and wind_directions must be 24*12 matrices.")

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    pc = ax.pcolor(wind_speeds, cmap=cmap)
    ax.invert_yaxis()
    ax.get_xlim()

    norm = Normalize(vmin=wind_speeds.min(), vmax=wind_speeds.max(), clip=True)
    mapper = ScalarMappable(norm=norm, cmap=cmap)

    _x = np.sin(np.deg2rad(wind_directions))
    _y = np.cos(np.deg2rad(wind_directions))

    ax.quiver(
        np.arange(1, 13, 1) - 0.5,
        np.arange(0, 24, 1) + 0.5,
        _x * wind_speeds,
        _y * wind_speeds,
        pivot="mid",
        fc="w",
        ec="k",
        lw=0.5,
        alpha=0.5,
    )

    if show_values:
        for _xx, row in enumerate(wind_directions.T):
            for _yy, local_direction in enumerate(row.T):
                local_speed = wind_speeds[_yy, _xx]
                cell_color = mapper.to_rgba(local_speed)
                text_color = contrasting_color(cell_color)
                if _xx == 0 and _yy == 0:
                    print(cell_color, text_color)
                ax.text(
                    _xx,
                    _yy + 1,
                    f"{local_direction:0.0f}°",
                    color=text_color,
                    ha="left",
                    va="bottom",
                    fontsize="xx-small",
                )
                ax.text(
                    _xx + 1,
                    _yy,
                    f"{local_speed:0.1f}m/s",
                    color=text_color,
                    ha="right",
                    va="top",
                    fontsize="xx-small",
                )

    ax.set_xticks(np.arange(1, 13, 1) - 0.5)
    ax.set_xticklabels([calendar.month_abbr[i] for i in np.arange(1, 13, 1)])
    # for label in ax.xaxis.get_ticklabels()[1::2]:
    #     label.set_visible(False)
    ax.set_yticks(np.arange(0, 24, 1) + 0.5)
    ax.set_yticklabels([f"{i:02d}:00" for i in np.arange(0, 24, 1)])
    for label in ax.yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    cb = plt.colorbar(pc, label="Average speed (m/s)")
    cb.outline.set_visible(False)

    if title is None:
        ax.set_title("Wind speed/direction matrix", x=0, ha="left")
    else:
        ax.set_title(f"{title}\nWind speed/direction matrix", x=0, ha="left")

    plt.tight_layout()

    return fig


def wind_matrix_from_epw(
    epw: EPW, cmap: Union[Colormap, str] = None, show_values: bool = False
) -> Figure:
    """Generate a wind-speed/direction matrix from an EPW file.

    Args:
        epw (EPW):
            An EPW file.
        cmap (Colormap, optional):
            A colormap to apply to the binned data. Defaults to Spectral.
        show_values (bool, optional):
            Show the values in the matrix. Defaults to False.
    Returns:
        Figure:
            A matplotlib figure object.
    """

    wd = to_series(epw.wind_direction)
    ws = to_series(epw.wind_speed)
    wind_directions = (
        (
            (
                wd.groupby([wd.index.month, wd.index.hour], axis=0).apply(
                    wind_direction_average
                )
                # + 90
            )
            % 360
        )
        .unstack()
        .T
    )
    wind_speeds = ws.groupby([ws.index.month, ws.index.hour], axis=0).mean().unstack().T

    return wind_matrix(
        wind_speeds,
        wind_directions,
        cmap=cmap,
        title=filename(epw),
        show_values=show_values,
    )
