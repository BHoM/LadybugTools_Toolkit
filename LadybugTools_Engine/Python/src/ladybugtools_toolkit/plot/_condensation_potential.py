"""Plotting methods for condensation potential."""

import calendar  # pylint: disable=E0401

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from ..bhom import decorator_factory
from .utilities import create_title


@decorator_factory()
def condensation_potential(
    dry_bulb_temperature: pd.Series,
    dew_point_temperature: pd.Series,
    ax: plt.Axes = None,
    dbt_quantile: float = 0.1,
    dpt_quantile: float = 0.9,
    **kwargs,
) -> plt.Axes:
    """Create a plot of the condensation potential for a given set of
    timeseries dry bulb temperature and dew point temperature.

    Args:
        dry_bulb_temperature (pd.Series):
            The dry bulb temperature dataset.
        dew_point_temperature (pd.Series):
            The dew point temperature dataset.
        ax (plt.Axes, optional):
            An optional plt.Axes object to populate. Defaults to None,
            which creates a new plt.Axes object.
        dbt_quantile (float, optional):
            The quantile of the dry bulb temperature to use for the
            condensation potential calculation. Defaults to 0.1.
        dpt_quantile (float, optional):
            The quantile of the dew point temperature to use for the
            condensation potential calculation. Defaults to 0.9.
        **kwargs:
            A set of kwargs to pass to plt.plot.

    Returns:
        plt.Axes: The plt.Axes object populated with the plot.

    """

    # check that the series are the same length and have the same index and are both indexes of pd.DatetimeIndex
    if not len(dry_bulb_temperature) == len(dew_point_temperature):
        raise ValueError(
            "The dry bulb temperature and dew point temperature must have the same length"
        )
    if not (dry_bulb_temperature.index == dew_point_temperature.index).all():
        raise ValueError(
            "The dry bulb temperature and dew point temperature must have the same index"
        )
    if not isinstance(dry_bulb_temperature.index, pd.DatetimeIndex) or not isinstance(
        dew_point_temperature.index, pd.DatetimeIndex
    ):
        raise ValueError(
            "The dry bulb temperature and dew point temperature must be time series"
        )

    if ax is None:
        ax = plt.gca()

    dbt_color = "red" if "dbt_color" not in kwargs else kwargs["dbt_color"]
    kwargs.pop("dbt_color", None)
    dpt_color = "blue" if "dpt_color" not in kwargs else kwargs["dpt_color"]
    kwargs.pop("dpt_color", None)
    potential_color = (
        "orange" if "risk_color" not in kwargs else kwargs["potential_color"]
    )
    kwargs.pop("potential_color", None)

    ax.set_title(
        create_title(
            kwargs.pop("title", None),
            "Condensation potential",
        )
    )

    # prepare data
    df = pd.concat(
        [dry_bulb_temperature, dew_point_temperature], axis=1, keys=["dbt", "dpt"]
    )
    dbt = df.dbt.groupby([df.index.month, df.index.hour], axis=0).quantile(dbt_quantile)
    dpt = df.dpt.groupby([df.index.month, df.index.hour], axis=0).quantile(dpt_quantile)

    # plot values
    for n, i in enumerate(range(len(dbt) + 1)[::24]):
        if n == len(range(len(dbt) + 1)[::24]) - 1:
            continue

        # get local values
        x = np.array(range(len(dbt) + 1)[i : i + 25])
        dbt_y = np.array(
            (dbt.values.tolist() + [dbt.values[0]])[i : i + 24]
            + [(dbt.values.tolist() + [dbt.values[0]])[i : i + 24][0]]
        )
        dpt_y = np.array(
            (dpt.values.tolist() + [dpt.values[0]])[i : i + 24]
            + [(dpt.values.tolist() + [dpt.values[0]])[i : i + 24][0]]
        )

        # dbt line
        ax.plot(
            x,
            dbt_y,
            c=dbt_color,
            label=f"Dry bulb temperature ({dbt_quantile:0.0%}-ile)"
            if n == 0
            else "_nolegend_",
        )
        # dpt line
        ax.plot(
            x,
            dpt_y,
            c=dpt_color,
            label=f"Dew-point temperature ({dpt_quantile:0.0%}-ile)"
            if n == 0
            else "_nolegend_",
        )
        # potential ranges
        ax.fill_between(
            x,
            dbt_y,
            dpt_y,
            where=dbt_y < dpt_y,
            color=potential_color,
            label="Highest condensation potential" if n == 0 else "_nolegend_",
        )

    ax.text(
        1,
        1,
        (
            f"{(dbt.values < dpt.values).sum() / len(dbt):0.1%} of annual hours "
            f"with potential for condensation\n(using {dbt_quantile:0.0%}-ile DBT "
            f"and {dpt_quantile:0.0%}-ile DPT)"
        ),
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    major_ticks = range(len(dbt))[::12]
    minor_ticks = range(len(dbt))[::6]
    major_ticklabels = []
    for i in dbt.index:
        if i[1] == 0:
            major_ticklabels.append(f"{calendar.month_abbr[i[0]]}")
        elif i[1] == 12:
            major_ticklabels.append("")

    ax.set_xlim(0, len(dbt))
    ax.xaxis.set_major_locator(mticker.FixedLocator(major_ticks))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
    ax.set_xticklabels(
        major_ticklabels,
        minor=False,
        ha="left",
    )

    # print
    ax.set_ylabel("Temperature (°C)")

    ax.legend(loc="upper left", bbox_to_anchor=(0, 1))

    return ax
