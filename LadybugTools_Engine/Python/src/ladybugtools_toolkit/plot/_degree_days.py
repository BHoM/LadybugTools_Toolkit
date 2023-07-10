import calendar

import matplotlib.pyplot as plt
from ladybug.epw import EPW

from ..ladybug_extension.epw import EPW, degree_time
from .utilities import add_bar_labels


def cooling_degree_days(
    epw: EPW, ax: plt.Axes = None, cool_base: float = 23, **kwargs
) -> plt.Axes:
    """Plot the cooling degree days from a given EPW object.

    Args:
        epw (EPW):
            An EPW object.
        ax (plt.Axes, optional):
            A matplotlib Axes object. Defaults to None.
        cool_base (float, optional):
            The temperature at which cooling kicks in. Defaults to 23.
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

    location_name = temp.columns.get_level_values(0).unique()[0]
    temp = temp.droplevel(0, axis=1).resample("MS").sum()
    temp.index = [calendar.month_abbr[i] for i in temp.index.month]
    clg = temp.filter(regex="Cooling")

    title = kwargs.pop("title", f"{location_name}\nCooling degree days")
    color = kwargs.pop("color", "blue")

    clg.plot(ax=ax, kind="bar", color=color, **kwargs)
    ax.set_ylabel(clg.columns[0])
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.grid(visible=True, which="major", axis="both", ls="--", lw=1, alpha=0.2)
    add_bar_labels(ax)

    ax.text(
        1,
        1,
        f"Annual: {sum(rect.get_height() for rect in ax.patches):0.0f}",
        transform=ax.transAxes,
        ha="right",
    )
    ax.set_title(title, x=0, ha="left")
    plt.tight_layout()
    return ax


def heating_degree_days(
    epw: EPW, ax: plt.Axes = None, heat_base: float = 18, **kwargs
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

    location_name = temp.columns.get_level_values(0).unique()[0]
    temp = temp.droplevel(0, axis=1).resample("MS").sum()
    temp.index = [calendar.month_abbr[i] for i in temp.index.month]
    data = temp.filter(regex="Heating")

    title = kwargs.pop("title", f"{location_name}\nHeating degree days")
    color = kwargs.pop("color", "orange")

    data.plot(ax=ax, kind="bar", color=color, **kwargs)
    ax.set_ylabel(data.columns[0])
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.grid(visible=True, which="major", axis="both", ls="--", lw=1, alpha=0.2)
    add_bar_labels(ax)

    ax.text(
        1,
        1,
        f"Annual: {sum(rect.get_height() for rect in ax.patches):0.0f}",
        transform=ax.transAxes,
        ha="right",
    )
    ax.set_title(title, x=0, ha="left")
    plt.tight_layout()
    return ax
