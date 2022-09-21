from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, Colormap
from matplotlib.figure import Figure
from matplotlib.tri.triangulation import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable


from ladybugtools_toolkit import analytics


@analytics
def spatial_heatmap(
    triangulations: List[Triangulation],
    values: List[List[float]],
    levels: Union[List[float], int] = None,
    contours: List[float] = None,
    cmap: Colormap = "viridis",
    extend: str = "neither",
    norm: BoundaryNorm = None,
    xlims: List[float] = None,
    ylims: List[float] = None,
    colorbar_label: str = "",
    title: str = "",
    highlight_pts: Dict[str, Tuple[int]] = None,
    show_legend_title: bool = True,
) -> Figure:
    """Plot a spatial map of a variable using a triangulation and associated values.

    Args:
        triangulations (List[Triangulation]):
            A list of triangulations to plot.
        values (List[List[float]]):
            A list of values, corresponding with the triangulations and their respective indices.
        levels (Union[List[float], int], optional):
            The number of levels to include in the colorbar. Defaults to None which will use
            10-steps between the min/max for all given values.
        contours (List[float], optional):
            Add contours at the given values to the spatial plot. Defaults to None.
        cmap (Colormap, optional):
            The colormap to use for this plot. Defaults to "viridis".
        extend (str, optional):
            Define how to handle the end-points of the colorbar. Defaults to "neither".
        norm (BoundaryNorm, optional):
            A matploltib BoundaryNorm object containing colormap boundary mapping information.
            Defaults to None.
        xlims (List[float], optional):
            The x-limit for the plot. Defaults to None.
        ylims (List[float], optional):
            The y-limit for the plot. Defaults to None.
        colorbar_label (str, optional):
            A label to be placed next to the colorbar. Defaults to "".
        title (str, optional):
            The title to be placed on the plot. Defaults to "".
        highlight_pts (Dict[str, int], optional):
            A set of points (and their names) to indicate on the spatial plot. Value is the int
            index of the highlighted point.
        show_legend_title (bool, optional):
            A convenient flag to hide the legend and title.

    Returns:
        Figure: A matplotlib Figure object.
    """
    for tri, zs in list(zip(*[triangulations, values])):
        if len(tri.x) != len(zs):
            raise ValueError(
                "The shape of the triangulations and values given do not match."
            )

    if levels is None:
        levels = np.linspace(
            min([np.amin(i) for i in values]), max([np.amax(i) for i in values]), 10
        )

    if xlims is None:
        xlims = [
            min([i.x.min() for i in triangulations]),
            max([i.x.max() for i in triangulations]),
        ]

    if ylims is None:
        ylims = [
            min([i.y.min() for i in triangulations]),
            max([i.y.max() for i in triangulations]),
        ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    for tri, zs in list(zip(*[triangulations, values])):
        tcf = ax.tricontourf(
            tri, zs, extend=extend, cmap=cmap, levels=levels, norm=norm
        )
        if contours is not None:
            tcl = ax.tricontour(tri, zs, levels=contours, colors=["w"])
            ax.clabel(tcl, inline=1, fontsize=10, colors=["k"])

    if highlight_pts is not None:
        if len(triangulations) > 1:
            raise ValueError(
                "Point highlighting is only possible for 1-length triangulations."
            )
        pt_size = (xlims[1] - xlims[0]) / 5
        for k, v in highlight_pts.items():
            ax.scatter(
                triangulations[0].x[v], triangulations[0].y[v], s=pt_size, c="red"
            )
            ax.text(
                triangulations[0].x[v] + (pt_size / 10),
                triangulations[0].y[v],
                k,
                ha="left",
                va="center",
            )

    if show_legend_title:
        # Plot colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)

        cbar = plt.colorbar(
            tcf, cax=cax  # , format=mticker.StrMethodFormatter("{x:04.1f}")
        )
        cbar.outline.set_visible(False)
        cbar.set_label(colorbar_label)

        ax.set_title(title, ha="left", va="bottom", x=0)

    plt.tight_layout()

    return fig
