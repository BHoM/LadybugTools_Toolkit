import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def radiation_tilt_orientation_factor(
    radiation_matrix: pd.DataFrame,
    title: str = None,
) -> Figure:
    """Convert a radiation matrix to a figure showing the radiation tilt and orientation.

    Args:
        radiation_matrix (pd.DataFrame): A matrix with altitude index, azimuth columns, and radiation values in Wh/m2.
        title (str, optional): A title for the figure. Defaults to None.

    Returns:
        Figure: A figure.
    """

    # Construct input values
    x = np.tile(radiation_matrix.index, [len(radiation_matrix.columns), 1]).T
    y = np.tile(radiation_matrix.columns, [len(radiation_matrix.index), 1])
    z = radiation_matrix.values

    z_min = radiation_matrix.min().min()
    z_max = radiation_matrix.max().max()
    z_percent = z / z_max * 100

    # Find location of max value
    ind = np.unravel_index(np.argmax(z, axis=None), z.shape)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    cf = ax.contourf(y, x, z / 1000, cmap="YlOrRd", levels=51)
    cl = ax.contour(
        y,
        x,
        z_percent,
        levels=[50, 60, 70, 80, 90, 95, 99],
        colors=["w"],
        linewidths=[1],
        alpha=0.75,
        linestyles=[":"],
    )
    ax.clabel(cl, fmt="%r %%")

    ax.scatter(y[ind], x[ind], c="k")
    ax.text(
        y[ind] + 2,
        x[ind] - 2,
        f"{z_max / 1000:0.0f}kWh/m${{^2}}$/year",
        ha="left",
        va="top",
        c="w",
    )

    [ax.spines[spine].set_visible(False) for spine in ["top", "right"]]

    ax.xaxis.set_major_locator(mtick.MultipleLocator(base=30))

    ax.grid(b=True, which="major", color="white", linestyle=":", alpha=0.25)

    cb = fig.colorbar(
        cf,
        orientation="vertical",
        drawedges=False,
        fraction=0.05,
        aspect=25,
        pad=0.02,
        label="kWh/m${^2}$/year",
    )
    cb.outline.set_visible(False)
    cb.add_lines(cl)
    cb.locator = mtick.MaxNLocator(nbins=10, prune=None)

    ax.set_xlabel("Panel orientation (clock-wise from North at 0°)")
    ax.set_ylabel("Panel tilt (0° facing the horizon, 90° facing the sky)")

    # Title
    if title is None:
        ax.set_title(f"Annual cumulative radiation", x=0, ha="left", va="bottom")
    else:
        ax.set_title(
            f"{title}\nAnnual cumulative radiation", x=0, ha="left", va="bottom"
        )

    plt.tight_layout()

    return fig
