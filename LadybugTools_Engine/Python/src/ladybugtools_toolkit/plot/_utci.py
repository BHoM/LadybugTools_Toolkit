import calendar
import textwrap
import warnings
from datetime import timedelta
from typing import List, Tuple

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import (
    UniversalThermalClimateIndex as LB_UniversalThermalClimateIndex,
)
from ladybug.epw import EPW
from ladybug.sunpath import Sunpath
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import make_interp_spline

from ..categorical.categories import (
    UTCI_DEFAULT_CATEGORIES,
    CategoriesBase,
    ComfortClass,
)
from ..external_comfort.utci import categorise, utci_comfort_categories
from ..helpers import ZeroPadPercentFormatter, contrasting_color, lighten_color
from ..ladybug_extension.analysis_period import (
    analysis_period_to_boolean,
    describe_analysis_period,
)
from ..ladybug_extension.datacollection import collection_to_series
from ..ladybug_extension.epw import EPW
from ._heatmap import heatmap
from .colormaps import UTCI_DIFFERENCE_COLORMAP
from .utilities import colormap_sequential, create_title


def utci_distance_to_comfortable(
    utci_collection: HourlyContinuousCollection,
    ax: plt.Axes = None,
    comfort_thresholds: Tuple[float] = (9, 26),
    distance_to_comfort_band_centroid: bool = False,
    **kwargs,
) -> plt.Axes:
    """Plot the distance (in C) to comfortable for a given Ladybug HourlyContinuousCollection
        containing UTCI values - using a different, less funky colour-scheme.

    Args:
        utci_collection (HourlyContinuousCollection):
            A Ladybug Universal Thermal Climate Index HourlyContinuousCollection object.
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        comfort_thresholds (List[float], optional):
            The comfortable band of UTCI temperatures. Defaults to [9, 26].
        distance_to_comfort_band_centroid (bool, optional):
            If True, the distance to the centroid of the comfort band is plotted. If False, the
            distance to the edge of the comfort band is plotted. Defaults to False.
        **kwargs (Dict[str, Any], optional):
            Additional keyword arguments to pass to the matplotlib plotting function.
    Returns:
        plt.Axes:
            A matplotlib Axes object.
    """

    if len(comfort_thresholds) != 2:
        raise ValueError("comfort_thresholds must be a list of length 2.")

    if not isinstance(
        utci_collection.header.data_type, LB_UniversalThermalClimateIndex
    ):
        raise ValueError("This method only works for UTCI data.")

    if ax is None:
        ax = plt.gca()

    low_limit = min(comfort_thresholds)
    high_limit = max(comfort_thresholds)
    midpoint = np.mean(comfort_thresholds)

    vals = np.array(utci_collection.values)
    if not distance_to_comfort_band_centroid:
        vals = np.where(
            vals < low_limit,
            vals - low_limit,
            np.where(vals > high_limit, vals - high_limit, 0),
        )
        ti = f'Distance from "comfortable" category (between {low_limit}°C and {high_limit}°C UTCI)'
    else:
        vals = np.where(vals < midpoint, -(midpoint - vals), vals - midpoint)
        ti = f'Distance from "comfortable" category midpoint (at {midpoint:0.1f}°C between {low_limit}°C and {high_limit}°C UTCI)'
    new_collection = utci_collection.get_aligned_collection(vals)

    return heatmap(
        collection_to_series(new_collection),
        cmap=kwargs.get("cmap", colormap_sequential("#00A9E0", "w", "#ba000d")),
        title="\n".join([i for i in [kwargs.pop("title", None), ti] if i is not None]),
        vmin=kwargs.get("vmin", -20),
        vmax=kwargs.get("vmax", 20),
    )


def utci_comfort_band_comparison(
    utci_collections: Tuple[HourlyContinuousCollection],
    ax: plt.Axes = None,
    identifiers: Tuple[str] = None,
    utci_categories: CategoriesBase = UTCI_DEFAULT_CATEGORIES,
    **kwargs,
) -> plt.Axes:
    """Create a proportional bar chart showing how different UTCI collections compare in terms of time within each comfort band.

    Args:
        utci_collections (List[HourlyContinuousCollection]):
            A list of UTCI collections.
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        identifiers (List[str], optional):
            A list of names to give each collection. Defaults to None.
        utci_categories (Categories, optional):
            The UTCI categories to use. Defaults to UTCI_DEFAULT_CATEGORIES.
        **kwargs:
            Additional keyword arguments to pass to the function.

    Returns:
        plt.Axes:
            A matplotlib Axes object.
    """

    for n, col in enumerate(utci_collections):
        if not isinstance(col.header.data_type, LB_UniversalThermalClimateIndex):
            raise ValueError(
                f"Collection {n} data type is not UTCI and cannot be used in this plot."
            )
    if any(len(i) != len(utci_collections[0]) for i in utci_collections):
        raise ValueError("All collections must be the same length.")

    if ax is None:
        ax = plt.gca()

    # set the title
    ax.set_title(kwargs.pop("title", None))

    if identifiers is None:
        identifiers = [f"{n}" for n in range(len(utci_collections))]
    if len(identifiers) != len(utci_collections):
        raise ValueError(
            "The number of identifiers given does not match the number of UTCI collections given!"
        )

    counts = (
        pd.concat(
            [
                utci_categories.categorise(collection_to_series(i)).value_counts()
                for i in utci_collections
            ],
            axis=1,
            keys=identifiers,
        )
        .T[[i.name for i in utci_categories.categories_with_limits()]]
        .T
    )
    (counts / counts.sum()).T.plot(
        ax=ax,
        kind="bar",
        stacked=True,
        color=utci_categories.colors,
        width=0.8,
        legend=False,
    )

    if kwargs.pop("legend", True):
        utci_categories.create_legend(ax, bbox_to_anchor=(1, 0.5), loc="center left")

    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    for c in ax.containers:
        # Optional: if the segment is small or 0, customize the labels
        labels = [f"{v.get_height():0.1%}" if v.get_height() > 0.035 else "" for v in c]

        # remove the labels parameter if it's not needed for customized labels
        ax.bar_label(c, labels=labels, label_type="center", color="w")

    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(False)
    plt.xticks(rotation=0)
    ax.yaxis.set_major_locator(plt.NullLocator())

    return ax


def utci_day_comfort_metrics(
    utci: pd.Series,
    dbt: pd.Series,
    mrt: pd.Series,
    rh: pd.Series,
    ws: pd.Series,
    ax: plt.Axes = None,
    month: int = 6,
    day: int = 21,
    **kwargs,
) -> Figure:
    """Plot a single days UTCI with composite DBT, RH, MRT and WS components shown also.

    Args:
        utci (pd.Series):
            An annual time-indexed series containing UTCI values.
        dbt (pd.Series):
            An annual time-indexed series containing DBT values.
        mrt (pd.Series):
            An annual time-indexed series containing MRT values.
        rh (pd.Series):
            An annual time-indexed series containing RH values.
        ws (pd.Series):
            An annual time-indexed series containing WS values.
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        month (int, optional):
            The month to plot. Default is 6.
        day (int, optional):
             The day to plot. Default is 21.
        kwargs:
            Additional keyword arguments to pass to the matplotlib plot function.


    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if any(all(utci.index != i.index) for i in [dbt, mrt, rh, ws]):
        raise ValueError("All series must have the same index")

    if ax is None:
        ax = plt.gca()

    try:
        dt = f"{utci.index.year[0]}-{month}-{day}"
        date = utci.loc[dt].index[0]
    except KeyError as e:
        raise e

    axes = []
    for i in range(5):
        if i == 0:
            axes.append(ax)
        else:
            temp_ax = ax.twinx()
            rspine = temp_ax.spines["right"]
            rspine.set_position(("axes", 1 + (i / 20)))
            temp_ax.set_frame_on(True)
            temp_ax.patch.set_visible(False)
            axes.append(temp_ax)

    (a,) = axes[0].plot(utci.loc[dt], c="black", label="UTCI")
    axes[0].set_ylabel("UTCI")
    (b,) = axes[1].plot(dbt.loc[dt], c="red", alpha=0.75, label="DBT", ls="--")
    axes[1].set_ylabel("DBT")
    (c,) = axes[2].plot(mrt.loc[dt], c="orange", alpha=0.75, label="MRT", ls="--")
    axes[2].set_ylabel("MRT")
    (d,) = axes[3].plot(rh.loc[dt], c="blue", alpha=0.75, label="RH", ls="--")
    axes[3].set_ylabel("RH")
    (e,) = axes[4].plot(ws.loc[dt], c="green", alpha=0.75, label="WS", ls="--")
    axes[4].set_ylabel("WS")

    axes[0].spines["right"].set_visible(False)

    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[0].set_xlim(utci.loc[dt].index.min(), utci.loc[dt].index.max())

    axes[0].legend(
        handles=[a, b, c, d, e],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=[0.5, -0.15],
        frameon=False,
    )

    # set the title
    title = [kwargs.pop("title", None), f"{date:%B %d}"]
    ax.set_title("\n".join([i for i in title if i is not None]))

    return ax


def utci_comparison_diurnal_day(
    utci_collections: List[HourlyContinuousCollection],
    ax: plt.Axes = None,
    month: int = 6,
    collection_ids: List[str] = None,
    agg: str = "mean",
    utci_categories: CategoriesBase = UTCI_DEFAULT_CATEGORIES,
    show_legend: bool = True,
    categories_in_legend: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot a set of UTCI collections on a single figure for monthly diurnal periods.

    Args:
        utci_collections (List[HourlyContinuousCollection]):
            A list of UTCI collections.
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        month (int, optional):
            The month to get the typical day from. Default is 6.
        collection_ids (List[str], optional):
            A list of descriptions for each of the input collections. Defaults to None.
        agg (str, optional):
            How to generate the "typical" day. Defualt is "mean" which uses the mean for each timestep in that month.
        utci_categories (Categories, optional):
            The UTCI categories to use. Defaults to UTCI_DEFAULT_CATEGORIES.
        show_legend (bool, optional):
            If True, show the legend. Defaults to True.
        categories_in_legend (bool, optional):
            If True, add the UTCI categories to the legend. Defaults to True.
        **kwargs:
            Additional keyword arguments to pass to the matplotlib plot function.
            ylims (List[float], optional):
                The y-axis limits. Defaults to None which just uses the min/ax of the fiven collections.

    Returns:
        Figure:
            A matplotlib figure object.
    """

    # check all input collections are UTCI collections
    for n, col in enumerate(utci_collections):
        if not isinstance(col.header.data_type, LB_UniversalThermalClimateIndex):
            raise ValueError(
                f"Collection {n} data type is not UTCI and cannot be used in this plot."
            )

    if ax is None:
        ax = plt.gca()

    if collection_ids is None:
        collection_ids = [f"{i:02d}" for i in range(len(utci_collections))]
    assert len(utci_collections) == len(
        collection_ids
    ), "The length of collections_ids must match the number of collections."

    # set the title
    title = [
        kwargs.pop("title", None),
        f"{calendar.month_name[month]} typical day ({agg})",
    ]
    ax.set_title("\n".join([i for i in title if i is not None]))

    # combine utcis and add names to columns
    df = pd.concat(
        [collection_to_series(i) for i in utci_collections], axis=1, keys=collection_ids
    )
    ylims = kwargs.pop("ylims", [df.min().min(), df.max().max()])
    df_agg = df.groupby([df.index.month, df.index.hour], axis=0).agg(agg).loc[month]
    df_agg.index = range(24)
    # add a final value to close the day
    df_agg.loc[24] = df_agg.loc[0]

    df_agg.plot(ax=ax, legend=True, zorder=3, **kwargs)

    # Fill between ranges
    for cat in utci_categories.categories_with_limits():
        ax.axhspan(
            max([cat.low_limit, -100]),
            min([cat.high_limit, 100]),
            color=lighten_color(cat.color, 0.2),
            zorder=2,
            label="_nolegend_" if not categories_in_legend else cat.name,
        )

    # Format plots
    ax.set_xlim(0, 24)
    ax.set_ylim(ylims)
    ax.xaxis.set_major_locator(plt.FixedLocator([0, 6, 12, 18]))
    ax.xaxis.set_minor_locator(plt.FixedLocator([3, 9, 15, 21]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00"], minor=False, ha="left")
    ax.set_ylabel("Universal Thermal Climate Index (°C)")
    ax.set_xlabel("Time of day")

    # add grid using a hacky fix
    for i in ax.get_xticks():
        ax.axvline(
            i, color=ax.xaxis.label.get_color(), ls=":", lw=0.5, alpha=0.1, zorder=5
        )
    for i in ax.get_yticks():
        ax.axhline(
            i, color=ax.yaxis.label.get_color(), ls=":", lw=0.5, alpha=0.1, zorder=5
        )

    if show_legend:
        # Construct legend
        ax.legend(
            loc="upper left",
            bbox_to_anchor=[1, 1],
            frameon=False,
            fontsize="small",
            ncol=1,
        )

    return ax


def utci_heatmap_difference(
    utci_collection1: HourlyContinuousCollection,
    utci_collection2: HourlyContinuousCollection,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Create a heatmap showing the annual hourly UTCI difference between collections.

    Args:
        utci_collection1 (HourlyContinuousCollection):
            The first UTCI collection.
        utci_collection2 (HourlyContinuousCollection):
            The second UTCI collection.
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        **kwargs:
            Additional keyword arguments to pass to the heatmap function.

    Returns:
        plt.Axes:
            A matplotlib Axes object.
    """

    if not isinstance(
        utci_collection1.header.data_type, LB_UniversalThermalClimateIndex
    ):
        raise ValueError("Input collection 1 is not a UTCI collection.")
    if not isinstance(
        utci_collection2.header.data_type, LB_UniversalThermalClimateIndex
    ):
        raise ValueError("Input collection 2 is not a UTCI collection.")

    if ax is None:
        ax = plt.gca()

    return heatmap(
        collection_to_series(utci_collection2) - collection_to_series(utci_collection1),
        ax=ax,
        cmap=kwargs.pop("cmap", UTCI_DIFFERENCE_COLORMAP),
        title=kwargs.pop("title", "UTCI difference"),
        **kwargs,
    )


def utci_pie(
    utci_collection: HourlyContinuousCollection,
    ax: plt.Axes = None,
    show_legend: bool = True,
    show_values: bool = False,
    utci_categories: CategoriesBase = UTCI_DEFAULT_CATEGORIES,
    **kwargs,
) -> plt.Axes:
    """Create a figure showing the UTCI proportion for the given analysis period.

    Args:
        utci_collection (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object.
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        show_legend (bool, optional):
            Set to True to plot the legend also. Default is True.
        show_values (bool, optional):
            Set to True to show values. Default is True.
        utci_categories (Categories, optional):
            The UTCI categories to use. Defaults to UTCI_DEFAULT_CATEGORIES.
        **kwargs:
            Additional keyword arguments to pass to the plotting function.

    Returns:
        plt.Axes: A matplotlib Axes object.
    """

    if not isinstance(
        utci_collection.header.data_type, LB_UniversalThermalClimateIndex
    ):
        raise ValueError("Input collection is not a UTCI collection.")

    if ax is None:
        ax = plt.gca()

    title = kwargs.pop("title", None)
    ax.set_title(title)

    series = collection_to_series(utci_collection)

    sizes = utci_categories.categorise(series).value_counts()[
        utci_categories.names
    ] / len(series)

    def func(pct, allvals):
        absolute = int(np.round(pct / 100.0 * np.sum(allvals)))
        if pct <= 0.05:
            return ""
        return f"{pct:.1f}%"

    _, _, autotexts = ax.pie(
        sizes,
        colors=utci_categories.colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "w", "linewidth": 1},
        autopct=lambda pct: func(pct, sizes) if show_values else None,
        pctdistance=0.9,
    )

    if show_values:
        plt.setp(autotexts, weight="bold", color="w")

    centre_circle = plt.Circle((0, 0), 0.60, fc="white")
    ax.add_artist(centre_circle)

    if show_legend:
        utci_categories.create_legend(
            ax=ax, loc="center left", bbox_to_anchor=(1, 0.5), **kwargs
        )

    return ax


def utci_journey(
    utci_values: Tuple[float],
    ax: plt.Axes = None,
    names: Tuple[str] = None,
    curve: bool = False,
    show_legend: bool = False,
    show_grid: bool = False,
    utci_categories: CategoriesBase = UTCI_DEFAULT_CATEGORIES,
    **kwargs,
) -> plt.Axes:
    """Create a figure showing the pseudo-journey between different UTCI conditions at a
        given time of year

    Args:
        utci_values (float):
            A list of UTCI values.
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        names (List[str], optional):
            A list of names to label each value with. Defaults to None.
        curve (bool, optional):
            Whether to plot the pseudo-journey as a spline. Defaults to False.
        show_legend (bool, optional):
            Set to True to plot the UTCI comfort band legend also.
        show_grid (bool, optional):
            Set to True to include a grid on the plot.
        utci_categories (Categories, optional):
            The UTCI categories to use. Defaults to UTCI_DEFAULT_CATEGORIES.
        **kwargs:
            Additional keyword arguments to pass to the plotting function.

    Returns:
        plt.Axes: A matplotlib Axes object.
    """

    # TODO - fix ylims kwargs to enasbal rthem to qwork!!!
    if names:
        if len(utci_values) != len(names):
            raise ValueError("Number of values and names must be equal.")
    else:
        names = [str(i) for i in range(len(utci_values))]

    if ax is None:
        ax = plt.gca()

    ylims = kwargs.get("ylims", None)
    if (ylims is not None) and len(ylims) != 2:
        raise ValueError("ylims must be a list/tuple of size 2.")

    # Convert collections into series and combine
    df_pit = pd.Series(utci_values, index=names)

    # Add UTCI background colors to the canvas
    for cat in utci_categories.categories_with_limits():
        ax.axhspan(
            max([cat.low_limit, -100]),
            min([cat.high_limit, 100]),
            facecolor=lighten_color(cat.color, 0.3),
            label=cat.name,
        )

    # add UTCI instance values to canvas
    for n, (idx, val) in enumerate(df_pit.items()):
        ax.scatter(n, val, c="white", s=400, zorder=9)
        ax.text(n, val, idx, zorder=10, ha="center", va="center", fontsize="medium")

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
        ax.plot(xnew, ynew, c="#B30202", ls="--", zorder=3, **kwargs)

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

    plt.tight_layout()

    return ax


def utci_heatmap(
    utci_collection: HourlyContinuousCollection,
    ax: plt.Axes = None,
    utci_categories: CategoriesBase = UTCI_DEFAULT_CATEGORIES,
    **kwargs,
) -> plt.Axes:
    """Create a heatmap showing the annual hourly UTCI for this HourlyContinuousCollection.

    Args:
        utci_collection (HourlyContinuousCollection):
            An HourlyContinuousCollection containing UTCI.
        ax (plt.Axes, optional):
            A matplotlib Axes object to plot on. Defaults to None.
        utci_categories (Categories, optional):
            A Categories object with colors, ranges and limits. Defaults to UTCI_DEFAULT_CATEGORIES.
        **kwargs:
            Additional keyword arguments to pass to the heatmap function.

    Returns:
        plt.Axes:
            A matplotlib Axes object.
    """

    # TODO - make tyhe above/below and bins within the same length
    return heatmap(
        collection_to_series(utci_collection),
        ax=ax,
        cmap=utci_categories.cmap,
        norm=utci_categories.boundarynorm,
        extend="both",
        **kwargs,
    )


def utci_monthly_histogram(
    utci_collection: HourlyContinuousCollection,
    ax: plt.Axes = None,
    utci_categories: CategoriesBase = UTCI_DEFAULT_CATEGORIES,
    **kwargs,
) -> plt.Axes:
    """Create a stacked bar chart showing the monthly distribution of UTCI values.

    Args:
        utci_collection (HourlyContinuousCollection):
            An HourlyContinuousCollection containing UTCI.
        ax (plt.Axes, optional):
            A matplotlib Axes object to plot on. Defaults to None.
        utci_categories (Categories, optional):
            A Categories object with colors, ranges and limits. Defaults to UTCI_CATEGORIES.
        **kwargs:
            Additional keyword arguments to pass to the bar function.

    Returns:
        plt.Axes:
            A matplotlib Axes object.
    """

    if not isinstance(
        utci_collection.header.data_type, LB_UniversalThermalClimateIndex
    ):
        raise ValueError("Input collection is not a UTCI collection.")

    series = collection_to_series(utci_collection)

    if ax is None:
        ax = plt.gca()

    t = utci_categories.timeseries_summary_monthly(series, density=True).T
    t.T.plot.bar(
        ax=ax,
        stacked=True,
        color=utci_categories.colors,
        legend=False,
        width=1,
    )
    ax.set_xlabel(None)
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(t.columns, ha="center", rotation=0)
    ax.yaxis.set_major_formatter(ZeroPadPercentFormatter)

    # Add header percentages for bar plot - and check that ComfortClass is available on object using that if it is
    try:
        if len(utci_categories.comfort_classes) != len(
            utci_categories.categories_with_limits()
        ):
            raise ValueError(
                "Monkey-patched comfort_class attributes are not the same length as the categories within this Category object."
            )
        cold_stress = []
        no_stress = []
        heat_stress = []
        for n, col in t.iteritems():
            cs = 0
            ns = 0
            hs = 0
            for nn, i in enumerate(col):
                if utci_categories.comfort_classes[nn] == ComfortClass.TOO_COLD:
                    cs += i
                elif utci_categories.comfort_classes[nn] == ComfortClass.COMFORTABLE:
                    ns += i
                elif utci_categories.comfort_classes[nn] == ComfortClass.TOO_HOT:
                    hs += i
                else:
                    raise ValueError("How'd you get here?")
            cold_stress.append(cs)
            no_stress.append(ns)
            heat_stress.append(hs)

    except AttributeError:
        warnings.warn(
            'No ComfortClass found on given "utci_categories", defaulting to regex lookup of hot/cold names for categories to determine comfort.'
        )
        cold_stress = t.T.filter(regex="(?i)cool|cold|chill").sum(axis=1)
        heat_stress = t.T.filter(regex="(?i)heat|hot").sum(axis=1)
        no_stress = 1 - cold_stress - heat_stress

    for n, (i, j, k) in enumerate(zip(*[cold_stress, no_stress, heat_stress])):
        ax.text(
            n,
            1.02,
            f"{i:0.1%}",
            va="bottom",
            ha="center",
            color="#3C65AF",
            fontsize="small",
        )
        ax.text(
            n,
            1.02,
            f"{j:0.1%}\n",
            va="bottom",
            ha="center",
            color="#2EB349",
            fontsize="small",
        )
        ax.text(
            n,
            1.02,
            f"{k:0.1%}\n\n",
            va="bottom",
            ha="center",
            color="#C31F25",
            fontsize="small",
        )

    # TODO - add labels to histogram bars optional
    return ax


def utci_heatmap_histogram(
    utci_collection: HourlyContinuousCollection,
    utci_categories: CategoriesBase = UTCI_DEFAULT_CATEGORIES,
    **kwargs,
) -> Figure:
    """Create a combined heatmap/histoghram figure for a UTCI data collection.

    Args:
        utci_collection (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object.
        utci_categories (Categories, optional):
            A Categories object with colors, ranges and limits. Defaults to UTCI_DEFAULT_CATEGORIES.
        **kwargs:
            Additional keyword arguments to pass.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if not isinstance(
        utci_collection.header.data_type, LB_UniversalThermalClimateIndex
    ):
        raise ValueError(
            "Collection data type is not UTCI and cannot be used in this plot."
        )

    series = collection_to_series(utci_collection)

    title = kwargs.pop("title", None)
    figsize = kwargs.pop("figsize", (15, 5))

    # Instantiate figure
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(
        ncols=1, nrows=2, width_ratios=[1], height_ratios=[4, 2], hspace=0.0
    )
    heatmap_ax = fig.add_subplot(spec[0, 0])
    histogram_ax = fig.add_subplot(spec[1, 0])

    # Add heatmap
    utci_heatmap(
        utci_collection,
        ax=heatmap_ax,
        utci_categories=utci_categories,
        show_colorbar=False,
        **kwargs,
    )

    # Add stacked plot
    utci_monthly_histogram(
        utci_collection, ax=histogram_ax, utci_categories=utci_categories, **kwargs
    )

    # add colorbar
    categories = utci_categories.categories_with_limits()
    divider = make_axes_locatable(histogram_ax)
    colorbar_ax = divider.append_axes("bottom", size="20%", pad=0.65)
    cb = fig.colorbar(
        mappable=heatmap_ax.get_children()[0],
        cax=colorbar_ax,
        orientation="horizontal",
        drawedges=False,
        ticks=utci_categories._bin_edges()[1:-1],
        extend="both",
    )
    cb.outline.set_visible(False)
    for n, cat in enumerate(categories):
        if n == 0:
            ha = "right"
            position = cat.high_limit
        elif n == len(categories) - 1:
            ha = "left"
            position = cat.low_limit
        else:
            ha = "center"
            position = (cat.low_limit + cat.high_limit) / 2

        colorbar_ax.text(
            position,
            1,
            textwrap.fill(cat.name, 11),
            ha=ha,
            va="bottom",
            size="small",
            # transform=colorbar_ax.transAxes,
        )

    title = f"{series.name} - {title}" if title is not None else series.name
    heatmap_ax.set_title(title, y=1, ha="left", va="bottom", x=0)

    return fig


def utci_histogram(
    utci_collection: HourlyContinuousCollection,
    ax: plt.Axes = None,
    utci_categories: CategoriesBase = UTCI_DEFAULT_CATEGORIES,
    **kwargs,
) -> plt.Axes:
    """Create a histogram showing the distribution of UTCI values.

    Args:
        utci_collection (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object.
        ax (plt.Axes, optional):
            A matplotlib Axes object to plot on. Defaults to None.
        utci_categories (Categories, optional):
            A Categories object with colors, ranges and limits. Defaults to UTCI_DEFAULT_CATEGORIES.
        **kwargs:
            Additional keyword arguments to pass to the plotting function.

    Returns:
        plt.Axes:
            A matplotlib Axes object.
    """
    if ax is None:
        ax = plt.gca()

    ti = kwargs.pop("title", None)
    if ti is not None:
        ax.set_title(ti)

    color = kwargs.pop("color", "white")
    bg_lighten = kwargs.pop("bg_lighten", 0.8)
    alpha = kwargs.pop("alpha", 0.5)

    edges = utci_categories._bin_edges()

    # fill between ranges
    for cat in utci_categories.categories_with_limits():
        ax.axvspan(
            max([cat.low_limit, -100]),
            min([cat.high_limit, 100]),
            facecolor=lighten_color(cat.color, bg_lighten),
            label=cat.name,
        )

    # get the bins
    bins = kwargs.pop(
        "bins",
        np.linspace(
            edges[1] - 100,
            edges[-2] + 100,
            int((edges[-2] + 100) - (edges[1] - 100)) + 1,
        ),
    )
    density = kwargs.pop("density", True)

    # get the binned data within categories
    series = collection_to_series(utci_collection)
    cnt = utci_categories.categorise(series).value_counts()[
        [i.name for i in utci_categories.categories_with_limits()]
    ]
    cnt = cnt / sum(cnt)

    # plot data
    series.plot(
        kind="hist", ax=ax, bins=bins, color=color, alpha=alpha, density=density
    )

    # set xlims
    xlim = kwargs.pop(
        "xlim",
        (edges[1] - 10, edges[-2] + 10),
    )
    ax.set_xlim(xlim)

    # get positions for percentage labels
    _ylow, _yhigh = ax.get_ylim()
    _xlow, _xhigh = ax.get_xlim()
    for n, cat in enumerate(utci_categories.categories_with_limits()):
        midpt = cat.mid_point
        if n == 0:
            if np.isinf(midpt):
                midpt = (cat.high_limit + _xlow) / 2
            if midpt < _xlow:
                continue
        if n == len(utci_categories.categories_with_limits()) - 1:
            if np.isinf(midpt):
                midpt = (cat.low_limit + _xhigh) / 2
            if midpt > _xhigh:
                continue
        ax.text(
            midpt,
            _yhigh * 0.99,
            f"{textwrap.fill(cat.name, 10)}\n{cnt[n]:0.1%}",
            ha="center",
            va="top",
            color=contrasting_color(lighten_color(cat.color, bg_lighten)),
            fontsize="small",
        )
    # print(cnt)

    ax.set_xlabel(series.name)
    ax.yaxis.set_major_formatter(ZeroPadPercentFormatter)

    ax.set_xticks(edges[1:-1])

    return ax
