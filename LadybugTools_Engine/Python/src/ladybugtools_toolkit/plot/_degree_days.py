"""Methods for plotting degree days from EPW files."""
import calendar  # pylint: disable=E0401

import matplotlib.pyplot as plt
from ladybug.epw import EPW
from .utilities import contrasting_color

from ..bhom import decorator_factory
from ..ladybug_extension.epw import EPW, degree_time
from ..ladybug_extension.location import location_to_string


@decorator_factory()
def cooling_degree_days(
    epw: EPW,
    ax: plt.Axes = None,
    cool_base: float = 23,
    show_labels: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot the cooling degree days from a given EPW object.

    Args:
        epw (EPW):
            An EPW object.
        ax (plt.Axes, optional):
            A matplotlib Axes object. Defaults to None.
        cool_base (float, optional):
            The temperature at which cooling kicks in. Defaults to 23.
        show_labels (bool, optional):
            Whether to show the labels on the bars. Defaults to True.
        **kwargs:
            Additional keyword arguments to pass to the matplotlib bar plot.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if not isinstance(epw, EPW):
        raise ValueError("epw is not an EPW object.")

    if ax is None:
        ax = plt.gca()

    temp = degree_time([epw], return_type="days", cool_base=cool_base)

    temp = temp.droplevel(0, axis=1).resample("MS").sum()
    temp.index = [calendar.month_abbr[i] for i in temp.index.month]
    clg = temp.filter(regex="Cooling")

    color = kwargs.pop("color", "#00A9E0")

    clg.plot(ax=ax, kind="bar", color=color, **kwargs)
    ax.set_ylabel(clg.columns[0])
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.grid(visible=True, which="major", axis="both", ls="--", lw=1, alpha=0.2)

    ax.text(
        1,
        1,
        f"Annual: {sum(rect.get_height() for rect in ax.patches):0.0f} days",
        transform=ax.transAxes,
        ha="right",
    )

    if show_labels:
        max_height = max(v.get_height() for v in ax.containers[0])
        for i, c in enumerate(ax.containers):
            label_colors = [contrasting_color(i.get_facecolor()) for i in c.patches]
            labels = [
                f"{v.get_height():0.0f}" if v.get_height() > 0.1 * max_height else ""
                for v in c
            ]
            ax.bar_label(
                c,
                labels=labels,
                label_type="edge",
                color=label_colors[i],
                # fontsize="small",
                fontweight="bold",
                padding=-15,
            )
            labels = [
                f"{v.get_height():0.0f}" if v.get_height() <= 0.1 * max_height else ""
                for v in c
            ]
            ax.bar_label(
                c,
                labels=labels,
                label_type="edge",
                # color=label_colors[i],
                # fontsize="small",
                fontweight="bold",
                # padding=-15,
            )

    plt.tight_layout()
    return ax


@decorator_factory()
def heating_degree_days(
    epw: EPW,
    ax: plt.Axes = None,
    heat_base: float = 18,
    show_labels: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot the heating/cooling degree days from a given EPW
    object.

    Args:
        epw (EPW):
            An EPW object.
        ax (plt.Axes, optional):
            A matplotlib Axes object. Defaults to None.
        heat_base (float, optional):
            The temperature at which heating kicks in. Defaults to 18.
        show_labels (bool, optional):
            Whether to show the labels on the bars. Defaults to True.
        **kwargs:
            Additional keyword arguments to pass to the matplotlib
            bar plot.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if not isinstance(epw, EPW):
        raise ValueError("epw is not an EPW object.")

    if ax is None:
        ax = plt.gca()

    temp = degree_time([epw], return_type="days", heat_base=heat_base)

    temp = temp.droplevel(0, axis=1).resample("MS").sum()
    temp.index = [calendar.month_abbr[i] for i in temp.index.month]
    data = temp.filter(regex="Heating")

    color = kwargs.pop("color", "#D50032")

    data.plot(ax=ax, kind="bar", color=color, **kwargs)
    ax.set_ylabel(data.columns[0])
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.grid(visible=True, which="major", axis="both", ls="--", lw=1, alpha=0.2)

    ax.text(
        1,
        1,
        f"Annual: {sum(rect.get_height() for rect in ax.patches):0.0f} days",
        transform=ax.transAxes,
        ha="right",
    )

    if show_labels:
        max_height = max(v.get_height() for v in ax.containers[0])
        for i, c in enumerate(ax.containers):
            label_colors = [contrasting_color(i.get_facecolor()) for i in c.patches]
            labels = [
                f"{v.get_height():0.0f}" if v.get_height() > 0.1 * max_height else ""
                for v in c
            ]
            ax.bar_label(
                c,
                labels=labels,
                label_type="edge",
                color=label_colors[i],
                # fontsize="small",
                fontweight="bold",
                padding=-15,
            )
            labels = [
                f"{v.get_height():0.0f}" if v.get_height() <= 0.1 * max_height else ""
                for v in c
            ]
            ax.bar_label(
                c,
                labels=labels,
                label_type="edge",
                # color=label_colors[i],
                # fontsize="small",
                fontweight="bold",
                # padding=-15,
            )

    plt.tight_layout()
    return ax


@decorator_factory()
def degree_days(epw: EPW, heat_base: float = 18, cool_base: float = 23, **kwargs):
    """Plot the heating/cooling degree days from a given EPW object.

    Args:
        epw (EPW):
            An EPW object.
        heat_base (float, optional):
            The temperature at which heating kicks in. Defaults to 18.
        cool_base (float, optional):
            The temperature at which cooling kicks in. Defaults to 23.
        **kwargs:
            Additional keyword arguments to pass to the plot. These can include:
            heat_color (str):
                The color of the heating degree days bars.
            cool_color (str):
                The color of the cooling degree days bars.
            figsize (Tuple[float]):
                The size of the figure.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    figsize = kwargs.pop("figsize", (15, 6))
    heat_color = kwargs.pop("heat_color", "#D50032")
    cool_color = kwargs.pop("cool_color", "#00A9E0")

    fig, ax = plt.subplots(nrows=2, figsize=figsize)
    heating_degree_days(epw, ax=ax[0], heat_base=heat_base, color=heat_color, **kwargs)
    cooling_degree_days(epw, ax=ax[1], cool_base=cool_base, color=cool_color, **kwargs)

    ax[0].set_title(f"{location_to_string(epw.location)} degree days", x=0, ha="left")
    return fig
