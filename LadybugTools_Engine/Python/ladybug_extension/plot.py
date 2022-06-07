from __future__ import annotations

from typing import List, Union
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW, AnalysisPeriod
import numpy as np
from ladybug_extension.datacollection import to_series, to_array
from ladybug_extension.header import to_string
from ladybug_extension.analysis_period import describe_analysis_period
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    Normalize,
)
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from ladybug.windrose import Compass, WindRose
from matplotlib.figure import Figure


def heatmap(
    collection: HourlyContinuousCollection,
    colormap: Colormap = "viridis",
    norm: BoundaryNorm = None,
    vlims: List[float] = None,
    title: str = None,
) -> Figure:
    """Plot a heatmap for a given Ladybug HourlyContinuousCollection.

    Args:
        collection (HourlyContinuousCollection): A Ladybug HourlyContinuousCollection object.
        colormap (Colormap, optional): The colormap to use in this heatmap. Defaults to "viridis".
        norm (BoundaryNorm, optional): A matplotlib BoundaryNorm object describing value thresholds. Defaults to None.
        vlims (List[float], optional): The limits to which values should be plotted (useful for comparing between different cases). Defaults to None.
        title (str, optional): A title to place at the top of the plot. Defaults to None.

    Returns:
        Figure: A matplotlib Figure object.
    """
    series = to_series(collection)

    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)

    if norm and vlims:
        raise ValueError("You cannot pass both vlims and a norm value to this method.")

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    # Reshape data into time/day matrix
    day_time_matrix = (
        series.to_frame()
        .pivot_table(columns=series.index.date, index=series.index.time)
        .values[::-1]
    )

    # Plot data
    heatmap = ax.imshow(
        day_time_matrix,
        extent=[
            mdates.date2num(series.index.min()),
            mdates.date2num(series.index.max()),
            726449,
            726450,
        ],
        aspect="auto",
        cmap=colormap,
        norm=norm,
        interpolation="none",
        vmin=None if vlims is None else vlims[0],
        vmax=None if vlims is None else vlims[1],
    )

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.tick_params(labelleft=True, labelright=True, labelbottom=True)
    plt.setp(ax.get_xticklabels(), ha="left", color="k")
    plt.setp(ax.get_yticklabels(), color="k")

    [
        ax.spines[spine].set_visible(False)
        for spine in ["top", "bottom", "left", "right"]
    ]

    ax.grid(b=True, which="major", color="white", linestyle=":", alpha=1)

    cb = fig.colorbar(
        heatmap,
        orientation="horizontal",
        drawedges=False,
        fraction=0.05,
        aspect=100,
        pad=0.075,
    )
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="k")
    cb.outline.set_visible(False)

    if title is None:
        ax.set_title(series.name, color="k", y=1, ha="left", va="bottom", x=0)
    else:
        ax.set_title(
            f"{series.name} - {title}",
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )

    # Tidy plot
    plt.tight_layout()

    return fig


def rose(
    epw: EPW,
    collection: HourlyContinuousCollection,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    colormap: Union[Colormap, str] = "jet",
    norm: BoundaryNorm = None,
    directions: int = 12,
    bins: List[float] = None,
    hide_label_legend: bool = False,
    title: str = None,
) -> Figure:
    """Generate a windrose plot for the given wind directions and variable.

    Args:
        epw (EPW): A ladybug EPW object.
        collection (HourlyContinuousCollection): Annual hourly variables to bin and color.
        analysis_period (AnalysisPeriod, optional): An analysis period within which to assess the input data. Defaults to Annual.
        colormap (Colormap, optional): A colormap to apply to the binned data. Defaults to None.
        directions (int, optional): The number of directions to bin wind-direction into. Defaults to 12.
        value_bins (List[float], optional): A set of bins into which data will be binned. Defaults to None.
        hide_label_legend (bool, optional): Hide the label and legend. Defaults to False.

    Returns:
        Figure: A matplotlib figure object.
    """

    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)

    if bins is None:
        bins = np.linspace(collection.min, collection.max, 11)

    colors = [colormap(i) for i in np.linspace(0, 1, len(bins))]

    ws_values = to_array(epw.wind_speed.filter_by_analysis_period(analysis_period))
    n_calm_hours = sum(ws_values == 0)

    not_calm: bool = to_array(epw.wind_speed) > 0
    filtered_collection = collection.filter_by_analysis_period(
        analysis_period
    ).filter_by_pattern(not_calm)
    filtered_wind_direction = epw.wind_direction.filter_by_analysis_period(
        analysis_period
    ).filter_by_pattern(not_calm)

    wr = WindRose(filtered_wind_direction, filtered_collection, directions)
    width = 360 / directions
    theta = np.radians(np.array(wr.angles) + (width / 2))[:-1]
    width = np.radians(width)

    binned_data = np.array([np.histogram(i, bins)[0] for i in wr.histogram_data])

    if title is None:
        title = "\n".join(
            [
                f"{to_series(collection).name} for {Path(epw.file_path).stem}",
                describe_analysis_period(analysis_period),
                f"Calm for {n_calm_hours / len(ws_values):0.2%} of the time ({n_calm_hours} hours)",
            ]
        )

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    plt.setp(ax.get_xticklabels(), fontsize="small")
    ax.spines["polar"].set_visible(False)
    ax.grid(True, which="both", ls="--")
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.set_xticks(np.radians(Compass.MAJOR_AZIMUTHS), minor=False)
    ax.set_xticklabels(Compass.MAJOR_TEXT, minor=False, **{"fontsize": "medium"})
    ax.set_xticks(np.radians(Compass.MINOR_AZIMUTHS), minor=True)
    ax.set_xticklabels(Compass.MINOR_TEXT, minor=True, **{"fontsize": "x-small"})

    bottom = np.zeros(directions)
    for n, d in enumerate(binned_data.T):
        ax.bar(
            x=theta,
            height=d,
            width=width,
            bottom=bottom,
            color=colors[n],
            ec=(1, 1, 1, 0.2),
            lw=0.5,
        )
        bottom += d

    if not hide_label_legend:
        norm = Normalize(vmin=bins[0], vmax=bins[-2]) if norm is None else norm
        colorbar_axes = fig.add_axes([1, 0.11, 0.03, 0.78])
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        colorbar = plt.colorbar(
            sm,
            ticks=bins,
            boundaries=bins,
            cax=colorbar_axes,
            label=to_series(collection).name,
        )
        colorbar.outline.set_visible(False)

        ax.set_title(title, ha="left", x=0)

    plt.tight_layout()

    return fig


def diurnal(
    dry_bulb_temperature: HourlyContinuousCollection,
    direct_normal_radiation: HourlyContinuousCollection,
    diffuse_horizontal_radiation: HourlyContinuousCollection,
    global_horizontal_radiation: HourlyContinuousCollection,
    moisture_collection: HourlyContinuousCollection,
    title: str = None,
) -> Figure:
    """Generate a monthly diurnal plot describing daily profiles for key EPW variables.

    Args:
        dry_bulb_temperature (HourlyContinuousCollection): An annual hourly ladybug data collection describing Dry-Bulb Temperature.
        direct_normal_radiation (HourlyContinuousCollection): An annual hourly ladybug data collection describing Direct Normal Radiation.
        diffuse_horizontal_radiation (HourlyContinuousCollection): An annual hourly ladybug data collection describing Diffuse Horizontal Radiation.
        global_horizontal_radiation (HourlyContinuousCollection): An annual hourly ladybug data collection describing Global Horizontal Radiation.
        moisture_collection (HourlyContinuousCollection): An annual hourly ladybug data collection describing either Relative Humidity, Dew Point Temperature or Wet Bulb Temperature.
        title (str, optional): A title to place at the top of the plot. Defaults to None.

    Returns:
        Figure: A matplotlib figure object.
    """

    dbt_color = "#bc204b"
    moisture_color = "#006da8"
    rad_color = "#eb671c"

    # Temperature
    dbt_min = dry_bulb_temperature._time_interval_operation(
        "monthlyperhour", "percentile", percentile=0
    )
    dbt_low = dry_bulb_temperature._time_interval_operation(
        "monthlyperhour", "percentile", percentile=5
    )
    dbt_avg = dry_bulb_temperature._time_interval_operation("monthlyperhour", "average")
    dbt_high = dry_bulb_temperature._time_interval_operation(
        "monthlyperhour", "percentile", percentile=95
    )
    dbt_max = dry_bulb_temperature._time_interval_operation(
        "monthlyperhour", "percentile", percentile=100
    )

    # Moisture
    mst_min = moisture_collection._time_interval_operation(
        "monthlyperhour", "percentile", percentile=0
    )
    mst_low = moisture_collection._time_interval_operation(
        "monthlyperhour", "percentile", percentile=5
    )
    mst_mid = moisture_collection._time_interval_operation("monthlyperhour", "average")
    mst_high = moisture_collection._time_interval_operation(
        "monthlyperhour", "percentile", percentile=95
    )
    mst_max = moisture_collection._time_interval_operation(
        "monthlyperhour", "percentile", percentile=100
    )

    # Radiation
    ghr_avg = global_horizontal_radiation._time_interval_operation(
        "monthlyperhour", "average"
    )
    dnr_avg = direct_normal_radiation._time_interval_operation(
        "monthlyperhour", "average"
    )
    dhr_avg = diffuse_horizontal_radiation._time_interval_operation(
        "monthlyperhour", "average"
    )

    # X axis values
    x_values = range(288)
    idx = [
        item
        for sublist in [
            [datetime(2021, month, 1, hour, 0, 0) for hour in range(24)]
            for month in range(1, 13)
        ]
        for item in sublist
    ]
    idx_str = [i.strftime("%b %H:%M") for i in idx]

    # Instantiate plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 5 * 1.5), sharex=True)

    for i in range(0, 288)[::24]:

        # Plot temperature
        axes[0].plot(
            x_values[i : i + 24],
            dbt_avg[i : i + 24],
            color=dbt_color,
            lw=2,
            label="Average",
            zorder=7,
        )
        axes[0].plot(
            x_values[i : i + 24],
            dbt_low[i : i + 24],
            color=dbt_color,
            lw=1,
            label="Average",
            ls=":",
        )
        axes[0].plot(
            x_values[i : i + 24],
            dbt_high[i : i + 24],
            color=dbt_color,
            lw=1,
            label="Average",
            ls=":",
        )
        axes[0].fill_between(
            x_values[i : i + 24],
            dbt_min[i : i + 24],
            dbt_max[i : i + 24],
            color=dbt_color,
            alpha=0.2,
            label="Range",
        )
        axes[0].fill_between(
            x_values[i : i + 24],
            dbt_low[i : i + 24],
            dbt_high[i : i + 24],
            color="white",
            alpha=0.5,
            label="Range",
        )
        axes[0].set_ylabel(
            to_string(dry_bulb_temperature.header),
            labelpad=2,
        )

        # Plot moisture
        axes[1].plot(
            x_values[i : i + 24],
            mst_mid[i : i + 24],
            color=moisture_color,
            lw=2,
            label="Average",
            zorder=7,
        )
        axes[1].plot(
            x_values[i : i + 24],
            mst_low[i : i + 24],
            color=moisture_color,
            lw=1,
            label="Average",
            ls=":",
        )
        axes[1].plot(
            x_values[i : i + 24],
            mst_high[i : i + 24],
            color=moisture_color,
            lw=1,
            label="Average",
            ls=":",
        )
        axes[1].fill_between(
            x_values[i : i + 24],
            mst_min[i : i + 24],
            mst_max[i : i + 24],
            color=moisture_color,
            alpha=0.2,
            label="Range",
        )
        axes[1].fill_between(
            x_values[i : i + 24],
            mst_low[i : i + 24],
            mst_high[i : i + 24],
            color="white",
            alpha=0.5,
            label="Range",
        )
        axes[1].set_ylabel(
            to_string(moisture_collection.header),
            labelpad=2,
        )

        # Plot radiation
        axes[2].plot(
            x_values[i : i + 24],
            dnr_avg[i : i + 24],
            color=rad_color,
            lw=1.5,
            ls="--",
            label="Direct normal radiation",
            zorder=7,
        )
        axes[2].plot(
            x_values[i : i + 24],
            dhr_avg[i : i + 24],
            color=rad_color,
            lw=2,
            ls=":",
            label="Diffuse horizontal radiation",
            zorder=7,
        )
        axes[2].plot(
            x_values[i : i + 24],
            ghr_avg[i : i + 24],
            color=rad_color,
            lw=2,
            ls="-",
            label="Global horizontal radiation",
            zorder=7,
        )
        axes[2].set_ylabel("Solar Radiation (W/m²)", labelpad=2)

    # Format plot area
    for n, ax in enumerate(axes):
        ax.xaxis.set_major_locator(mtick.FixedLocator(range(0, 288, 24)))
        ax.xaxis.set_minor_locator(mtick.FixedLocator(range(12, 288, 24)))
        ax.yaxis.set_major_locator(mtick.MaxNLocator(7))

        [ax.spines[spine].set_visible(False) for spine in ["top", "right"]]
        [ax.spines[j].set_color("k") for j in ["bottom", "left"]]

        ax.set_xlim([0, 287])
        ax.set_xticklabels(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            minor=False,
            ha="left",
            color="k",
        )
        ax.grid(b=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.2)
        ax.grid(b=True, which="minor", axis="both", c="k", ls=":", lw=1, alpha=0.1)

    # Legend
    handles, labels = axes[2].get_legend_handles_labels()
    handles = [
        Line2D([0], [0], label="Average", color="k"),
        Line2D([0], [0], label="5-95%ile", color="k", lw=1, ls=":"),
        mpatches.Patch(color="grey", label="Range", alpha=0.3),
    ] + handles[:3]

    lgd = axes[2].legend(
        handles=handles,
        bbox_to_anchor=(0.5, -0.3),
        loc=8,
        ncol=6,
        borderaxespad=0,
        frameon=False,
    )
    lgd.get_frame().set_facecolor((1, 1, 1, 0))
    [plt.setp(text, color="k") for text in lgd.get_texts()]

    # Title
    if title is None:
        fig.suptitle(
            "Monthly average diurnal profile",
            color="k",
            ha="left",
            va="bottom",
            x=0.05,
            y=0.92,
        )
    else:
        fig.suptitle(
            f"{title}\nMonthly average diurnal profile",
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )

    # Tidy plot
    plt.tight_layout()

    return fig


def radiation_tilt_orientation(radiation_matrix: pd.DataFrame, title: str = None,) -> Figure:
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
    cl = ax.contour(y, x, z_percent, levels=[50, 60, 70, 80, 90, 95, 99], colors=["w"], linewidths=[1], alpha=0.75, linestyles=[":"])
    ax.clabel(cl, fmt='%r %%')
    
    ax.scatter(y[ind], x[ind], c="k")
    ax.text(y[ind]+2, x[ind]-2, f"{z_max / 1000:0.0f}kWh/m${{^2}}$/year", ha="left", va="top", c="w")

    [
        ax.spines[spine].set_visible(False)
        for spine in ["top", "right"]
    ]

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
        ax.set_title(f"{title}\nAnnual cumulative radiation", x=0, ha="left", va="bottom")

    plt.tight_layout()

    return fig