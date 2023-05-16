import calendar
import subprocess
import tempfile
import textwrap
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.color import Colorset
from ladybug.compass import Compass
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import \
    UniversalThermalClimateIndex as LB_UniversalThermalClimateIndex
from ladybug.epw import EPW
from ladybug.sunpath import Sunpath
from ladybug.viewsphere import ViewSphere
from ladybug.wea import Wea
from ladybug.windrose import WindRose
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import (BoundaryNorm, Colormap, LinearSegmentedColormap,
                               ListedColormap, Normalize, is_color_like,
                               rgb2hex)
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.interpolate import make_interp_spline
from scipy.stats import exponweib

from ..external_comfort import HBR_FOLDERS
from ..external_comfort.utci import (UniversalThermalClimateIndex, categorise,
                                     utci_comfort_categories)
from ..helpers import (cardinality, contrasting_color, lighten_color,
                       rolling_window, validate_timeseries, weibull_pdf,
                       wind_direction_average)
from ..ladybug_extension.analysis_period import (analysis_period_to_boolean,
                                                 analysis_period_to_datetimes,
                                                 describe_analysis_period)
from ..ladybug_extension.datacollection import (collection_to_array,
                                                collection_to_series)
from ..ladybug_extension.epw import (EPW, degree_time, epw_to_dataframe,
                                     get_filename)
from ..ladybug_extension.location import location_to_string
from ..wind.direction_bins import DirectionBins


def colormap_sequential(
    *colors: Union[str, float, int, Tuple]
) -> LinearSegmentedColormap:
    """
    Create a sequential colormap from a list of input colors.

    Args:
        colors (Union[str, float, int, Tuple]):
            A list of colors according to their hex-code, string name, character code or
            RGBA values.

    Returns:
        LinearSegmentedColormap:
            A matplotlib colormap.

    Examples:
    >> colormap_sequential("green", "#F034A3", (0.5, 0.2, 0.8), "y")
    """
    for color in colors:
        if not isinstance(color, (str, float, int, tuple)):
            raise KeyError(f"{color} not recognised as a valid color string.")

    if len(colors) < 2:
        raise KeyError("Not enough colors input to create a colormap.")

    fixed_colors = []
    for c in colors:
        if is_color_like(c):
            try:
                fixed_colors.append(rgb2hex(c))
            except ValueError:
                fixed_colors.append(c)
        else:
            raise KeyError(f"{c} not recognised as a valid color string.")
    return LinearSegmentedColormap.from_list(
        f"{'_'.join(fixed_colors)}",
        list(zip(np.linspace(0, 1, len(fixed_colors)), fixed_colors)),
        N=256,
    )


def get_lb_colormap(name: Union[int, str] = "original") -> LinearSegmentedColormap:
    """Create a Matplotlib from a colormap provided by Ladybug.

    Args:
        name (Union[int, str], optional):
            The name of the colormap to create. Defaults to "original".

    Raises:
        ValueError:
            If an invalid LB colormap name is provided, return a list of potential values to use.

    Returns:
        LinearSegmentedColormap:
            A Matplotlib colormap object.
    """
    colorset = Colorset()

    cmap_strings = []
    for colormap in dir(colorset):
        if colormap.startswith("_"):
            continue
        if colormap == "ToString":
            continue
        cmap_strings.append(colormap)

    if name not in cmap_strings:
        raise ValueError(f"name must be one of {cmap_strings}")

    lb_cmap = getattr(colorset, name)()
    rgb = [[getattr(rgb, i) / 255 for i in ["r", "g", "b", "a"]] for rgb in lb_cmap]
    rgb = [tuple(i) for i in rgb]
    return colormap_sequential(*rgb)


UTCI_COLORMAP = ListedColormap(
    [
        "#262972",
        "#3452A4",
        "#3C65AF",
        "#37BCED",
        "#2EB349",
        "#F38322",
        "#C31F25",
        "#7F1416",
    ]
)
UTCI_COLORMAP.set_under("#0D104B")
UTCI_COLORMAP.set_over("#580002")
UTCI_LEVELS = [-40, -27, -13, 0, 9, 26, 32, 38, 46]
UTCI_LABELS = [
    "Extreme Cold Stress",
    "Very Strong Cold Stress",
    "Strong Cold Stress",
    "Moderate Cold Stress",
    "Slight Cold Stress",
    "No Thermal Stress",
    "Moderate Heat Stress",
    "Strong Heat Stress",
    "Very Strong Heat Stress",
    "Extreme Heat Stress",
]
UTCI_BOUNDARYNORM = BoundaryNorm(UTCI_LEVELS, UTCI_COLORMAP.N)

DBT_COLORMAP = colormap_sequential("white", "#bc204b")
RH_COLORMAP = colormap_sequential("white", "#8db9ca")
MRT_COLORMAP = colormap_sequential("white", "#6d104e")
WS_COLORMAP = colormap_sequential(
    "#d0e8e4",
    "#8db9ca",
    "#006da8",
    "#24135f",
)

BEAUFORT_BINS = [
    0,
    0.3,
    1.5,
    3.3,
    5.5,
    7.9,
    10.7,
    13.8,
    17.1,
    20.7,
    24.4,
    28.4,
    32.6,
]


def timeseries(
    series: pd.Series,
    ax: plt.Axes = None,
    xlims: Tuple[datetime] = None,
    ylims: Tuple[datetime] = None,
    **kwargs,
) -> plt.Axes:
    """Create a timeseries plot of a pandas Series.

    Args:
        series (pd.Series):
            The pandas Series to plot. Must have a datetime index.
        ax (plt.Axes, optional):
            An optional plt.Axes object to populate. Defaults to None, which creates a new plt.Axes object.
        xlims (Tuple[datetime], optional):
            Set the x-limits. Defaults to None.
        ylims (Tuple[datetime], optional):
            Set the y-limits. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the plt.plot() function.

    Returns:
        plt.Axes:
            The populated plt.Axes object.
    """

    validate_timeseries(series)

    if ax is None:
        ax = plt.gca()

    ax.plot(series.index, series.values, **kwargs)  ## example plot here

    # TODO - add cmap arg to color line by y value -  https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

    if xlims is None:
        ax.set_xlim(series.index.min(), series.index.max())
    else:
        ax.set_xlim(xlims)
    if ylims is None:
        ax.set_ylim(ax.get_ylim())
    else:
        ax.set_ylim(ylims)

    return ax


def heatmap(
    series: pd.Series,
    ax: plt.Axes = None,
    show_colorbar: bool = True,
    **kwargs,
) -> plt.Axes:
    """Create a heatmap of a pandas Series.

    Args:
        series (pd.Series):
            The pandas Series to plot. Must have a datetime index.
        ax (plt.Axes, optional):
            An optional plt.Axes object to populate. Defaults to None, which creates a new plt.Axes object.
        show_colorbar (bool, optional):
            If True, show the colorbar. Defaults to True.
        **kwargs:
            Additional keyword arguments to pass to plt.pcolormesh().

    Returns:
        plt.Axes:
            The populated plt.Axes object.
    """

    validate_timeseries(series)

    if ax is None:
        ax = plt.gca()

    day_time_matrix = (
        series.dropna()
        .to_frame()
        .pivot_table(columns=series.index.date, index=series.index.time)
    )
    x = mdates.date2num(day_time_matrix.columns.get_level_values(1))
    y = mdates.date2num(
        pd.to_datetime([f"2017-01-01 {i}" for i in day_time_matrix.index])
    )
    z = day_time_matrix.values

    # handle non-standard kwargs
    if "title" in kwargs:
        ax.set_title(kwargs.pop("title"))

    # Plot data
    pcm = ax.pcolormesh(
        x,
        y,
        z[:-1, :-1],
        **kwargs,
    )
    if show_colorbar:
        cb = plt.colorbar(
            pcm,
            ax=ax,
            orientation="horizontal",
            drawedges=False,
            fraction=0.05,
            aspect=100,
            pad=0.075,
        )

        plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="k")
        cb.outline.set_visible(False)

    ax.xaxis_date()
    if len(set(series.index.year)) > 1:
        date_formatter = mdates.DateFormatter("%b %Y")
    else:
        date_formatter = mdates.DateFormatter("%b")
    ax.yaxis.set_major_formatter(date_formatter)

    ax.yaxis_date()
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.tick_params(labelleft=True, labelright=True, labelbottom=True)
    plt.setp(ax.get_xticklabels(), ha="left", color="k")
    plt.setp(ax.get_yticklabels(), color="k")

    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(False)

    _ = [ax.axvline(i, color="w", ls=":", lw=0.5, alpha=0.5) for i in ax.get_xticks()]
    _ = [ax.axhline(i, color="w", ls=":", lw=0.5, alpha=0.5) for i in ax.get_yticks()]

    return ax


def monthly_histogram_proportion(
    series: pd.Series,
    bins: List[float],
    ax: plt.Axes = None,
    labels: List[str] = None,
    show_year_in_label: bool = False,
    show_labels: bool = False,
    **kwargs,
) -> plt.Axes:
    """Create a monthly histogram of a pandas Series.

    Args:
        series (pd.Series):
            The pandas Series to plot. Must have a datetime index.
        bins (List[float]):
            The bins to use for the histogram.
        ax (plt.Axes, optional):
            An optional plt.Axes object to populate. Defaults to None, which creates a new plt.Axes object.
        labels (List[str], optional):
            The labels to use for the histogram. Defaults to None, which uses the bin edges.
        show_year_in_label (bool, optional):
            Whether to show the year in the x-axis label. Defaults to False.
        show_labels (bool, optional):
            Whether to show the labels on the bars. Defaults to False.
        **kwargs:
            Additional keyword arguments to pass to plt.bar.

    Returns:
        plt.Axes:
            The populated plt.Axes object.
    """

    validate_timeseries(series)

    if ax is None:
        ax = plt.gca()

    t = pd.cut(series, bins=bins, labels=labels)
    t = t.groupby([t.index.year, t.index.month, t]).count().unstack().T
    t = t / t.sum()

    # adjust column labels
    if show_year_in_label:
        t.columns = [
            f"{year}\n{calendar.month_abbr[month]}" for year, month in t.columns.values
        ]
    else:
        t.columns = [f"{calendar.month_abbr[month]}" for _, month in t.columns.values]

    t.T.plot.bar(
        ax=ax,
        stacked=True,
        legend=False,
        width=1,
        **kwargs,
    )
    ax.set_xlim(-0.5, len(t.columns) - 0.5)
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), ha="center", rotation=0)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))

    if show_labels:
        for i, c in enumerate(ax.containers):
            label_colors = [contrasting_color(i.get_facecolor()) for i in c.patches]
            labels = [
                f"{v.get_height():0.1%}" if v.get_height() > 0.1 else "" for v in c
            ]
            ax.bar_label(
                c,
                labels=labels,
                label_type="center",
                color=label_colors[i],
                fontsize="xx-small",
            )

    return ax


def condensation_risk(
    dry_bulb_temperature: pd.Series,
    dew_point_temperature: pd.Series,
    ax: plt.Axes = None,
    dbt_quantile: float = 0.03,
    dpt_quantile: float = 0.97,
    title: str = None,
    **kwargs,
) -> plt.Axes:
    """Create a plot of the condensation risk for a given set of timeseries dry bulb temperature and dew point temperature.

    Args:
        dry_bulb_temperature (pd.Series):
            The dry bulb temperature timeseries.
        dew_point_temperature (pd.Series):
            The dew point temperature timeseries.
        ax (plt.Axes, optional):
            An optional plt.Axes object to populate. Defaults to None, which creates a new plt.Axes object.
        dbt_quantile (float, optional):
            The quantile of the dry bulb temperature to use for the condensation risk calculation. Defaults to 0.03.
        dpt_quantile (float, optional):
            The quantile of the dew point temperature to use for the condensation risk calculation. Defaults to 0.97.
        title (str, optional):
            The title of the plot. Defaults to None.
        **kwargs:
            A set of kwargs to pass to plt.plot.

    Returns:
        plt.Axes: The plt.Axes object populated with the plot.

    """

    # check that the series are the same length and have the same index and are both indexes of pd.DatetimeIndex
    if not (len(dry_bulb_temperature) == len(dew_point_temperature)):
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

    # groupby month and time of day
    dbt = dry_bulb_temperature.groupby(
        [dry_bulb_temperature.index.month, dry_bulb_temperature.index.hour]
    ).quantile(dbt_quantile)
    dpt = dew_point_temperature.groupby(
        [dew_point_temperature.index.month, dew_point_temperature.index.hour]
    ).quantile(dpt_quantile)
    # print(dbt)
    dbt.index = dbt.index.to_series().apply(
        lambda x: f"{calendar.month_abbr[x[0]]}\n{x[1]:02d}:00"
    )
    dpt.index = dpt.index.to_series().apply(
        lambda x: f"{calendar.month_abbr[x[0]]}\n{x[1]:02d}:00"
    )

    dbt_color = "red" if "dbt_color" not in kwargs else kwargs["dbt_color"]
    kwargs.pop("dbt_color", None)
    dpt_color = "blue" if "dbt_color" not in kwargs else kwargs["dpt_color"]
    kwargs.pop("dpt_color", None)
    risk_color = "orange" if "risk_color" not in kwargs else kwargs["risk_color"]
    kwargs.pop("risk_color", None)

    ax.plot(
        dbt.index,
        dbt.values,
        label=f"Dry bulb temperature ({dbt_quantile:0.0%}-ile)",
        color=dbt_color,
        **kwargs,
    )
    ax.plot(
        dpt.index,
        dpt.values,
        label=f"Dew-point temperature ({dpt_quantile:0.0%}-ile)",
        color=dpt_color,
        **kwargs,
    )
    ax.fill_between(
        dbt.index,
        dbt.values,
        dpt.values,
        where=dbt < dpt,
        color=risk_color,
        label="Highest condensation risk",
    )

    ax.text(
        1,
        1,
        f"{(dbt < dpt).sum() / len(dbt):0.1%} risk of condensation\n(using {dbt_quantile:0.0%}-ile DBT and {dpt_quantile:0.0%}-ile DPT)",
        ha="right",
        va="top",
        transform=ax.transAxes,
    )

    ax.set_xlim(0, len(dbt))
    ax.set_ylabel("Temperature (°C)")
    if title is None:
        ax.set_title(
            f"Condensation risk (for values between {min(dry_bulb_temperature.index):%b %Y} and {max(dry_bulb_temperature.index):%b %Y})"
        )
    else:
        ax.set_title(title)
    ax.set_xticks(ax.get_xticks()[0::24], ha="left")

    ax.legend(loc="upper left", bbox_to_anchor=(0, 1))

    return ax


def annotate_imshow(
    im: AxesImage,
    data: List[float] = None,
    valfmt: str = "{x:.2f}",
    textcolors: Tuple[str] = ("black", "white"),
    threshold: float = None,
    exclude_vals: List[float] = None,
    **text_kw,
) -> List[str]:
    """A function to annotate a heatmap.

    Args:
        im (AxesImage):
            The AxesImage to be labeled.
        data (List[float], optional):
            Data used to annotate. If None, the image's data is used. Defaults to None.
        valfmt (_type_, optional):
            The format of the annotations inside the heatmap. This should either use the string
            format method, e.g. "$ {x:.2f}", or be a `matplotlib.ticker.Formatter`.
            Defaults to "{x:.2f}".
        textcolors (Tuple[str], optional):
            A pair of colors.  The first is used for values below a threshold, the second for
            those above.. Defaults to ("black", "white").
        threshold (float, optional):
            Value in data units according to which the colors from textcolors are applied. If None
            (the default) uses the middle of the colormap as separation. Defaults to None.
        exclude_vals (float, optional):
            A list of values where text should not be added. Defaults to None.
        **text_kw (dict, optional):
            All other keyword arguments are passed on to the created `~matplotlib.text.Text`

    Returns:
        List[str]:
            The texts added to the AxesImage.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be overwritten by textkw.
    text_kw = {"ha": "center", "va": "center"}
    text_kw.update({"ha": "center", "va": "center"})

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] in exclude_vals:
                pass
            else:
                text_kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **text_kw)
                texts.append(text)

    return texts


def sunpath(
    epw: EPW,
    ax: plt.Axes = None,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    data_collection: HourlyContinuousCollection = None,
    cmap: Union[Colormap, str] = "viridis",
    norm: BoundaryNorm = None,
    show_title: bool = True,
    sun_size: float = 10,
    show_grid: bool = True,
    show_legend: bool = True,
) -> plt.Axes:
    """Plot a sun-path for the given EPW and analysis period.
    Args:
        epw (EPW):
            An EPW object.
        ax (plt.Axes, optional):
            A matplotlib Axes object. Defaults to None.
        analysis_period (AnalysisPeriod, optional):
            _description_. Defaults to None.
        data_collection (HourlyContinuousCollection, optional):
            An aligned data collection. Defaults to None.
        cmap (str, optional):
            The colormap to apply to the aligned data_collection. Defaults to None.
        norm (BoundaryNorm, optional):
            A matplotlib BoundaryNorm object containing colormap boundary mapping information.
            Defaults to None.
        show_title (bool, optional):
            Set to True to include a title in the plot. Defaults to True.
        sun_size (float, optional):
            The size of each sun in the plot. Defaults to 0.2.
        show_grid (bool, optional):
            Set to True to show the grid. Defaults to True.
        show_legend (bool, optional):
            Set to True to include a legend in the plot if data_collection passed. Defaults to True.
    Returns:
        plt.Axes:
            A matplotlib Axes object.
    """

    if ax is None:
        ax = plt.gca()

    sunpath_obj = Sunpath.from_location(epw.location)
    all_suns = [
        sunpath_obj.calculate_sun_from_date_time(i) for i in analysis_period.datetimes
    ]
    suns = [i for i in all_suns if i.altitude > 0]
    suns_x, suns_y = np.array([sun.position_2d().to_array() for sun in suns]).T

    day_suns = []
    for month in [6, 9, 12]:
        date = pd.to_datetime(f"2017-{month:02d}-21")
        day_idx = pd.date_range(
            date, date + pd.Timedelta(hours=24), freq="1T", closed="left"
        )
        _ = []
        for idx in day_idx:
            s = sunpath_obj.calculate_sun_from_date_time(idx)
            if s.altitude > 0:
                _.append(np.array(s.position_2d().to_array()))
        day_suns.append(np.array(_))

    ax.set_aspect("equal")
    ax.set_xlim(-101, 101)
    ax.set_ylim(-101, 101)
    ax.axis("off")

    if show_grid:
        compass = Compass()
        ax.add_patch(
            plt.Circle(
                (0, 0),
                100,
                zorder=1,
                lw=0.5,
                ec="#555555",
                fc=(0, 0, 0, 0),
                ls="-",
            )
        )
        for pt, lab in list(zip(*[compass.major_azimuth_points, compass.MAJOR_TEXT])):
            _x, _y = np.array([[0, 0]] + [pt.to_array()]).T
            ax.plot(_x, _y, zorder=1, lw=0.5, ls="-", c="#555555", alpha=0.5)
            t = ax.text(_x[1], _y[1], lab, ha="center", va="center", fontsize="medium")
            t.set_bbox(
                {"facecolor": "white", "alpha": 1, "edgecolor": None, "linewidth": 0}
            )
        for pt, lab in list(zip(*[compass.minor_azimuth_points, compass.MINOR_TEXT])):
            _x, _y = np.array([[0, 0]] + [pt.to_array()]).T
            ax.plot(_x, _y, zorder=1, lw=0.5, ls="-", c="#555555", alpha=0.5)
            t = ax.text(_x[1], _y[1], lab, ha="center", va="center", fontsize="small")
            t.set_bbox(
                {"facecolor": "white", "alpha": 1, "edgecolor": None, "linewidth": 0}
            )

    if data_collection is not None:
        new_idx = analysis_period_to_datetimes(analysis_period)
        series = collection_to_series(data_collection)
        vals = (
            series.reindex(new_idx)
            .interpolate()
            .values[[i.altitude > 0 for i in all_suns]]
        )
        dat = ax.scatter(
            suns_x, suns_y, c=vals, s=sun_size, cmap=cmap, norm=norm, zorder=3
        )

        if show_legend:
            cb = ax.figure.colorbar(
                dat,
                pad=0.09,
                shrink=0.8,
                aspect=30,
                label=f"{series.name}",
            )
            cb.outline.set_visible(False)
    else:
        ax.scatter(suns_x, suns_y, c="#FFCF04", s=sun_size, zorder=3)

    # add equinox/solstice curves
    for day_sun in day_suns:
        _x, _y = day_sun.T
        ax.plot(
            _x,
            _y,
            c="k",
            alpha=0.6,
            zorder=1,
            ls=":",
            lw=0.75,
        )

    if show_title:
        title_string = "\n".join(
            [
                location_to_string(epw.location),
                describe_analysis_period(analysis_period),
            ]
        )
        if show_grid:
            ax.set_title(title_string, ha="left", x=0, y=1.05)
        else:
            ax.set_title(title_string, ha="left", x=0)

    plt.tight_layout()

    return ax


def week_profile(
    series: pd.Series,
    ax: plt.Axes = None,
    title: str = None,
    **kwargs,
) -> plt.Axes:
    """Plot a profile aggregated across days of week in a given Series.

    Args:
        series (pd.Series):
            A time-indexed Pandas Series object.
        ax (plt.Axes, optional):
            A matplotlib Axes object. Defaults to None.
        title (str, optional):
            A title to place at the top of the plot. Defaults to None.
        **kwargs (Dict[str, Any], optional):
            Additional keyword arguments to pass to the matplotlib plotting function.

    Returns:
        plt.Axes:
            A matplotlib Axes object.
    """

    if ax is None:
        ax = plt.gca()

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series passed is not datetime indexed.")

    days_included = series.index.day_of_week.unique()
    for day_of_week in range(7):
        if day_of_week not in days_included:
            warnings.warn(
                f"Series passed does not include any {calendar.day_name[day_of_week]}."
            )

    minmax_range = [0.0001, 0.9999]
    q_range = [0.05, 0.95]
    q_alpha = 0.3
    minmax_alpha = 0.1
    color = kwargs.get("color", "slategray")

    # Remove outliers
    series = series[
        (series >= series.quantile(minmax_range[0]))
        & (series <= series.quantile(minmax_range[1]))
    ]

    # remove nan/inf
    series = series.replace(-np.inf, np.nan).replace(np.inf, np.nan).dropna()

    start_idx = series.dropna().index.min()
    end_idx = series.dropna().index.max()

    # group data
    group = series.groupby([series.index.dayofweek, series.index.time])

    # count n samples per timestep
    n_samples = group.count().mean()

    # Get groupped data
    minima = group.min()
    lower = group.quantile(q_range[0])
    median = group.median()
    mean = group.mean()
    upper = group.quantile(q_range[1])
    maxima = group.max()

    # create df for re-indexing
    df = pd.concat([minima, lower, median, mean, upper, maxima], axis=1)
    df.columns = ["minima", "lower", "median", "mean", "upper", "maxima"]
    df = df.replace(-np.inf, np.nan).replace(np.inf, np.nan).fillna(0)

    # reset index and rename
    df = df.reset_index()
    idx = []
    for day, hour, minute in list(
        zip(
            *[
                [i + 1 for i in df.level_0.values],
                [i.hour for i in df.level_1.values],
                [i.minute for i in df.level_1.values],
            ]
        )
    ):
        idx.append(pd.to_datetime(f"2007-01-{day:02d} {hour:02d}:{minute:02d}:00"))
    df.index = idx
    df.drop(columns=["level_0", "level_1"], inplace=True)

    # q-q
    ax.fill_between(
        df.index,
        df["lower"],
        df["upper"],
        alpha=q_alpha,
        color=color,
        lw=None,
        ec=None,
        label=f"{q_range[0]:0.0%}ile-{q_range[1]:0.0%}ile",
    )
    # q-extreme
    ax.fill_between(
        df.index,
        df["lower"],
        df["minima"],
        alpha=minmax_alpha,
        color=color,
        lw=None,
        ec=None,
        label="min-max",
    )
    ax.fill_between(
        df.index,
        df["upper"],
        df["maxima"],
        alpha=minmax_alpha,
        color=color,
        lw=None,
        ec=None,
        label="_nolegend_",
    )
    # mean/median
    ax.plot(df.index, df["mean"], c=color, ls="-", lw=1, label="Mean")
    ax.plot(df.index, df["median"], c=color, ls="--", lw=1, label="Median")

    # format axes
    ax.set_xlim(df.index.min(), df.index.max())
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("k")

    ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.2)
    ax.grid(visible=True, which="minor", axis="both", c="k", ls=":", lw=1, alpha=0.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))

    # legend
    lgd = ax.legend(
        bbox_to_anchor=(0.5, -0.2),
        loc=8,
        ncol=6,
        borderaxespad=0,
        frameon=False,
    )
    lgd.get_frame().set_facecolor((1, 1, 1, 0))
    for text in lgd.get_texts():
        plt.setp(text, color="k")

    ti = f"Typical week between {start_idx:%Y-%m-%d} and {end_idx:%Y-%m-%d} (~{n_samples:0.1f} samples per timestep)"
    if title is not None:
        ti += "\n" + title
    ax.set_title(
        ti,
        ha="left",
        x=0,
    )

    plt.tight_layout()

    return ax


def timeseries_diurnal(
    series: pd.Series,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot a heatmap for a Pandas Series object.

    Args:
        series (pd.Series):
            A time-indexed Pandas Series object.
        ax (plt.Axes, optional):
            A matplotlib Axes object to plot on. Defaults to None.
        **kwargs (Dict[str, Any]):

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if ax is None:
        ax = plt.gca()

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series passed is not datetime indexed.")

    target_index = pd.MultiIndex.from_arrays(
        [
            [x for xs in [[i + 1] * 24 for i in range(12)] for x in xs],
            [x for xs in [range(0, 24, 1) for i in range(12)] for x in xs],
        ],
        names=["month", "hour"],
    )

    # groupby and obtain min, quant, mean, quant, max values
    grp = series.groupby([series.index.month, series.index.hour], axis=0)
    _min = grp.min().reindex(target_index)
    _lower = grp.quantile(0.05).reindex(target_index)
    _mean = grp.mean().reindex(target_index)
    _upper = grp.quantile(0.95).reindex(target_index)
    _max = grp.max().reindex(target_index)

    color = kwargs.get("color", "k")
    ylabel = kwargs.get("ylabel", series.name)

    x_values = range(288)

    # for each month, plot the diurnal profile
    for i in range(0, 288)[::24]:
        ax.plot(
            x_values[i : i + 24],
            _mean[i : i + 24],
            color=color,
            lw=2,
            label="Average",
            zorder=7,
        )
        ax.plot(
            x_values[i : i + 24],
            _lower[i : i + 24],
            color=color,
            lw=1,
            label="Average",
            ls=":",
        )
        ax.plot(
            x_values[i : i + 24],
            _upper[i : i + 24],
            color=color,
            lw=1,
            label="Average",
            ls=":",
        )
        ax.fill_between(
            x_values[i : i + 24],
            _min[i : i + 24],
            _max[i : i + 24],
            color=color,
            alpha=0.2,
            label="Range",
        )
        ax.fill_between(
            x_values[i : i + 24],
            _lower[i : i + 24],
            _upper[i : i + 24],
            color="white",
            alpha=0.5,
            label="Range",
        )
        ax.set_ylabel(
            ylabel,
            labelpad=2,
        )

    ax.xaxis.set_major_locator(mticker.FixedLocator(range(0, 288, 24)))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(range(12, 288, 24)))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(7))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("k")

    ax.set_xlim([0, 287])
    months = [calendar.month_abbr[i] for i in range(1, 13, 1)]
    ax.set_xticklabels(
        months,
        minor=False,
        ha="left",
        color="k",
    )
    ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.2)
    ax.grid(visible=True, which="minor", axis="both", c="k", ls=":", lw=1, alpha=0.1)

    handles = [
        mlines.Line2D([0], [0], label="Average", color="k", lw=2),
        mlines.Line2D([0], [0], label="5-95%ile", color="k", lw=1, ls=":"),
        mpatches.Patch(color="grey", label="Range", alpha=0.3),
    ]

    lgd = ax.legend(
        handles=handles,
        bbox_to_anchor=(0.5, -0.2),
        loc=8,
        ncol=6,
        borderaxespad=0,
        frameon=False,
    )
    lgd.get_frame().set_facecolor((1, 1, 1, 0))
    for text in lgd.get_texts():
        plt.setp(text, color="k")

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ylims = kwargs.get("ylims", None)
    if ylims is not None:
        ax.set_ylim(ylims)

    title = kwargs.get("title", None)
    if title is None:
        ax.set_title(
            f"Monthly average diurnal profile\n{series.name}",
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )
    else:
        ax.set_title(f"{title}", color="k", y=1, ha="left", va="bottom", x=0)

    plt.tight_layout()

    return ax


def wind_matrix(
    wind_speeds: List[List[float]],
    wind_directions: List[List[float]],
    ax: plt.Axes = None,
    show_values: bool = False,
    speed_lims: Tuple[float] = None,
    **kwargs,
) -> plt.Axes:
    """Generate a wind-speed/direction matrix from a set of 24*12 speed and direction values.

    Args:
        wind_speeds (List[List[float]]):
            A list of 24*12 wind speeds.
        wind_directions (List[List[float]]):
            A list of 24*12 wind directions (in degrees from north clockwise).
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        show_values (bool, optional):
            Show the values in the matrix. Defaults to False.
        speed_lims (Tuple[float], optional):
            The minimum and maximum wind speed values to use for the colorbar. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the matplotlib pcolor function.
    Returns:
        Figure:
            A matplotlib figure object.
    """

    # TODO - add kwargs here
    cmap = kwargs.get("cmap", "Spectral_r")
    title = kwargs.get("title", None)

    if ax is None:
        ax = plt.gca()

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

    pc = ax.pcolor(
        wind_speeds,
        cmap=cmap,
        vmin=min(speed_lims) if speed_lims else None,
        vmax=max(speed_lims) if speed_lims else None,
    )
    ax.invert_yaxis()
    ax.get_xlim()

    norm = Normalize(vmin=wind_speeds.min(), vmax=wind_speeds.max(), clip=True)
    mapper = ScalarMappable(norm=norm, cmap=cmap)

    _x = -np.sin(np.deg2rad(wind_directions))
    _y = -np.cos(np.deg2rad(wind_directions))

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

    return ax


def wind_speed_direction_frequency(
    wind_speed: List[float],
    wind_direction: List[float],
    ax: plt.Axes = None,
    speed_bins: List[float] = None,
    n_directions: float = 8,
    **kwargs,
) -> plt.Axes:
    """Create an image containing a matrix of wind speed/direction frequencies.

    Args:
        wind_speed (List[float]):
            A list if wind speeds to process.
        wind_direction (List[float]):
            A list of wind directions to process.
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        speed_bins (List[float], optional):
            A set of bins into which wind speed will be sorted. Defaults to None which divides the
            range 0-MAX into 10 bins.
        n_directions (float, optional):
            A number of cardinal directions into which wind_direction will be grouped.
            Default is 8.
        **kwargs:
            Additional keyword arguments to pass to the function.
    Returns:
        plt.Axes:
            A matplotlib Axes object.
    """

    cmap = kwargs.get("cmap", "Purples")
    title = kwargs.get("title", None)

    if ax is None:
        ax = plt.gca()

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

    return ax


def utci_comfort_band_comparison(
    utcis: Tuple[HourlyContinuousCollection],
    ax: plt.Axes = None,
    analysis_periods: Tuple[AnalysisPeriod] = (AnalysisPeriod()),
    identifiers: Tuple[str] = None,
    simplified: bool = True,
    comfort_limits: Tuple[float] = (9, 26),
    **kwargs,
) -> plt.Figure:
    """Create a proportional bar chart showing how differnet UTCI collections compare in terms of time within each simplified comfort band.

    Args:
        utcis (List[HourlyContinuousCollection]):
            A list if UTCI collections.
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        analysis_periods (AnalysisPeriod, optional):
            An set of analysis periods, where the combination of these is applied to all collections. Defaults to [AnalysisPeriod()].
        identifiers (List[str], optional):
            A list of names to give each collection. Defaults to None.
        simplified (bool, optional):
            Use simplified comfort bands. Defaults to True.
        comfort_limits (List[float], optional):
            Modify the default comfort limits. Defaults to [9, 26].
        **kwargs:
            Additional keyword arguments to pass to the function.

    Returns:
        plt.Figure:
            A figure object.
    """

    title = kwargs.get("title", None)

    if ax is None:
        ax = plt.gca()

    if identifiers is None:
        identifiers = [f"{n}" for n in range(len(utcis))]
    if len(identifiers) != len(utcis):
        raise ValueError(
            "The number of identifiers given does not match the number of UTCI collections given!"
        )

    labels, _ = utci_comfort_categories(
        simplified=simplified, comfort_limits=comfort_limits, rtype="category"
    )
    colors, _ = utci_comfort_categories(
        simplified=simplified, comfort_limits=comfort_limits, rtype="color"
    )

    df = pd.concat(
        [collection_to_series(col) for col in utcis],
        axis=1,
    ).loc[analysis_period_to_boolean(analysis_periods)]
    df.columns = [textwrap.fill(label, 15) for label in identifiers]

    # categorise values into comfort bands
    df_cat = categorise(
        df, fmt="category", simplified=simplified, comfort_limits=comfort_limits
    )

    # get value counts per collection/series
    counts = (
        (df_cat.apply(pd.value_counts) / len(df_cat)).reindex(labels).T.fillna(0)
    )[labels]

    counts.plot(ax=ax, kind="bar", stacked=True, color=colors, width=0.8, legend=False)

    handles, labels = ax.get_legend_handles_labels()
    _ = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=[0.5, -0.15],
        frameon=False,
        fontsize="small",
        ncol=3 if simplified else 5,
        title="Comfort categories",
    )
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    for c in ax.containers:
        # Optional: if the segment is small or 0, customize the labels
        labels = [f"{v.get_height():0.1%}" if v.get_height() > 0.035 else "" for v in c]

        # remove the labels parameter if it's not needed for customized labels
        ax.bar_label(c, labels=labels, label_type="center")

    plt.xticks(rotation=0)
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if title is None:
        plt.suptitle(
            f"{describe_analysis_period(analysis_periods)}",
            y=0.85,
            x=0.04,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize="x-large",
        )
    else:
        plt.suptitle(
            f"{title}\n{describe_analysis_period(analysis_periods)}",
            y=0.85,
            x=0.04,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize="x-large",
        )

    plt.tight_layout()

    return ax


def utci_comparison_diurnal_day(
    collections: List[HourlyContinuousCollection],
    ax: plt.Axes = None,
    month: int = 6,
    collection_ids: List[str] = None,
    agg: str = "mean",
    **kwargs,
) -> Figure:
    """Plot a set of UTCI collections on a single figure for monthly diurnal periods.

    Args:
        collections (List[HourlyContinuousCollection]):
            A list of UTCI collections.
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        month (int, optional):
            The month to get the typical day from. Default is 6.
        collection_ids (List[str], optional):
            A list of descriptions for each of the input collections. Defaults to None.
        agg (str, optional):
            How to generate the "typical" day. Defualt is "mean" which uses the mean for each timestep in that month.
        **kwargs:
            Additional keyword arguments to pass to the matplotlib plot function.

    Returns:
        Figure:
            A matplotlib figure object.
    """

    title = kwargs.get("title", None)
    colors = kwargs.get("colors", None)

    if ax is None:
        ax = plt.gca()

    if agg not in ["min", "mean", "max", "median"]:
        raise ValueError("agg is not of a possible type.")

    if collection_ids is None:
        collection_ids = [f"{i:02d}" for i in range(len(collections))]
    assert len(collections) == len(
        collection_ids
    ), "The length of collections_ids must match the number of collections."

    for n, col in enumerate(collections):
        if not isinstance(col.header.data_type, LB_UniversalThermalClimateIndex):
            raise ValueError(
                f"Collection {n} data type is not UTCI and cannot be used in this plot."
            )

    if colors is not None:
        if len(colors) != len(collections):
            raise ValueError(
                "The number of colors must match the number of collections."
            )
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)

    # combine utcis and add names to columns
    df = pd.concat(
        [collection_to_series(i) for i in collections], axis=1, keys=collection_ids
    )
    ylims = df.min().min(), df.max().max()
    df_agg = df.groupby([df.index.month, df.index.hour], axis=0).agg(agg).loc[month]
    df_agg.index = range(24)
    # add a final value to close the day
    df_agg.loc[24] = df_agg.loc[0]

    for col in df_agg.columns:
        ax.plot(df_agg[col].index, df_agg[col].values, label=col, zorder=3, **kwargs)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("k")

    # Fill between ranges
    utci_handles = []
    utci_labels = []
    for low, high, color, category in list(
        zip(
            *[
                ([-100] + UTCI_LEVELS + [100])[0:-1],
                ([-100] + UTCI_LEVELS + [100])[1:],
                [rgb2hex(UTCI_COLORMAP.get_under())]
                + UTCI_COLORMAP.colors
                + [rgb2hex(UTCI_COLORMAP.get_over())],
                UTCI_LABELS,
            ]
        )
    ):
        cc = lighten_color(color, 0.2)
        ax.axhspan(low, high, color=cc, zorder=2)
        # Get fille color attributes
        utci_labels.append(category)
        utci_handles.append(mpatches.Patch(color=cc, label=category))

    # get handles
    mitigation_handles, mitigation_labels = ax.get_legend_handles_labels()

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
        ax.axvline(i, color="k", ls=":", lw=0.5, alpha=0.1, zorder=5)

    for i in ax.get_yticks():
        ax.axhline(i, color="k", ls=":", lw=0.5, alpha=0.1, zorder=5)

    # ax.grid(
    #     visible=True,
    #     which="both",
    #     axis="both",
    #     c="k",
    #     ls=":",
    #     lw=1,
    #     alpha=0.1,
    #     zorder=5,
    # )

    # Construct legend
    handles = utci_handles + mitigation_handles
    labels = utci_labels + mitigation_labels
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=[1, 0.9],
        frameon=False,
        fontsize="small",
        ncol=1,
    )

    ti = f"{calendar.month_name[month]} typical day ({agg})"
    if title is not None:
        ti = f"{ti}\n{title}"
    ax.set_title(
        ti,
        ha="left",
        va="bottom",
        x=0,
    )

    plt.tight_layout()

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
            Additional keyword arguments to pass to the plotting function.
    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if ax is None:
        ax = plt.gca()

    # construct kwargs to pass to heatmap
    kwargs = {
        "vmin": kwargs.get("vmin", -10),
        "vmax": kwargs.get("vmax", 10),
        "title": kwargs.get("title", None),
        "cmap": kwargs.get("cmap", colormap_sequential("#00A9E0", "w", "#702F8A")),
    }

    return heatmap(
        collection_to_series(utci_collection2) - collection_to_series(utci_collection1),
        ax=ax,
        **kwargs,
    )


def utci_pie(
    utci_collection: HourlyContinuousCollection,
    ax: plt.Axes = None,
    analysis_periods: Tuple[AnalysisPeriod] = (AnalysisPeriod()),
    show_legend: bool = True,
    show_title: bool = True,
    show_values: bool = False,
    **kwargs,
) -> plt.Axes:
    """Create a figure showing the UTCI proportion for the given analysis period.

    Args:
        utci_collection (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object.
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        analysis_period (Tuple[AnalysisPeriod], optional):
            A ladybug analysis period or list of analysis periods.
        show_legend (bool, optional):
            Set to True to plot the legend also. Default is True.
        show_title (bool, optional):
            Set to True to show title. Default is True.
        show_values (bool, optional):
            Set to True to show values. Default is True.
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

    title = kwargs.get("title", None)

    mask = analysis_period_to_boolean(analysis_periods)
    series = collection_to_series(utci_collection)[mask]

    if len(analysis_periods) > 1:
        analysis_periods = [analysis_periods]
        ti = f"{series.name}\n{describe_analysis_period(analysis_periods[0], include_timestep=False)}"
    else:
        descs = [
            describe_analysis_period(i, include_timestep=False)
            for i in analysis_periods
        ]
        ti = f"{series.name}\n{', '.join(descs)}"

    series_cut = pd.cut(series, bins=[-100] + UTCI_LEVELS + [100], labels=UTCI_LABELS)
    sizes = (series_cut.value_counts() / len(series))[UTCI_LABELS]
    colors = (
        [UTCI_COLORMAP.get_under()] + UTCI_COLORMAP.colors + [UTCI_COLORMAP.get_over()]
    )

    def func(pct, allvals):
        absolute = int(np.round(pct / 100.0 * np.sum(allvals)))
        if pct <= 0.05:
            return ""
        return f"{pct:.1f}%"

    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=colors,
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
        # construct custom legend including values
        legend_elements = [
            mpatches.Patch(
                facecolor=color, edgecolor=None, label=f"[{sizes[label]:05.1%}] {label}"
            )
            for label, color in list(zip(*[UTCI_LABELS, colors]))
        ]
        lgd = ax.legend(handles=legend_elements, loc="center", frameon=False)
        lgd.get_frame().set_facecolor((1, 1, 1, 0))

    if title is not None:
        ti += "\n" + title

    if show_title:
        ax.set_title(ti, ha="left", va="bottom", x=0.1, y=0.9)

    plt.tight_layout()

    return ax


def utci_journey(
    utci_values: Tuple[float],
    ax: plt.Axes = None,
    names: Tuple[str] = None,
    curve: bool = False,
    show_legend: bool = False,
    show_grid: bool = False,
    **kwargs,
) -> plt.Axes:
    """Create a figure showing the pseudo-journey between different UTCI conditions at a
        given time of year

    Args:
        utci_values (float):
            A list of UTCI values.
        names (List[str], optional):
            A list of names to label each value with. Defaults to None.
        curve (bool, optional):
            Whether to plot the pseudo-journey as a spline. Defaults to False.
        show_legend (bool, optional):
            Set to True to plot the UTCI comfort band legend also.
        show_grid (bool, optional):
            Set to True to include a grid on the plot.
        **kwargs:
            Additional keyword arguments to pass to the plotting function.

    Returns:
        plt.Axes: A matplotlib Axes object.
    """

    if names:
        if len(utci_values) != len(names):
            raise ValueError("Number of values and names must be equal.")
    else:
        names = [str(i) for i in range(len(utci_values))]

    if ax is None:
        ax = plt.gca()

    ylims = kwargs.get("ylims", None)
    title = kwargs.get("title", None)
    if (ylims is not None) and len(ylims) != 2:
        raise ValueError("ylims must be a list/tuple of size 2.")

    # Convert collections into series and combine
    df_pit = pd.Series(utci_values, index=names)

    # Add UTCI background colors to the canvas
    background_colors = np.array(
        [rgb2hex(UTCI_COLORMAP.get_under())]
        + UTCI_COLORMAP.colors
        + [rgb2hex(UTCI_COLORMAP.get_over())]
    )
    background_ranges = rolling_window(np.array([-100] + UTCI_LEVELS + [100]), 2)
    for bg_color, (bg_start, bg_end), label in list(
        zip(*[background_colors, background_ranges, UTCI_LABELS])
    ):
        ax.axhspan(
            bg_start, bg_end, facecolor=lighten_color(bg_color, 0.3), label=label
        )

    # add UTCI instance values to canvas
    for n, (idx, val) in enumerate(df_pit.items()):
        ax.scatter(n, val, c="white", s=400, zorder=9)
        ax.text(
            n, val, idx, c="k", zorder=10, ha="center", va="center", fontsize="medium"
        )

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
        ax.plot(xnew, ynew, c="#B30202", ls="--", zorder=3)

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

    if title is not None:
        ax.set_title(
            title,
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )

    plt.tight_layout()

    return ax


def utci_heatmap(
    utci_collection: HourlyContinuousCollection, ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    """Create a heatmap showing the annual hourly UTCI for this HourlyContinuousCollection.

    Args:
        utci_collection (HourlyContinuousCollection):
            An HourlyContinuousCollection containing UTCI.
        ax (plt.Axes, optional):
            A matplotlib Axes object to plot on. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the heatmap function.

    Returns:
        plt.Axes:
            A matplotlib Axes object.
    """

    return heatmap(
        collection_to_series(utci_collection),
        ax=ax,
        title=kwargs.get("title", None),
        cmap=UTCI_COLORMAP,
        norm=UTCI_BOUNDARYNORM,
    )


def utci_monthly_histogram(
    utci_collection: HourlyContinuousCollection,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Create a stacked bar chart showing the monthly distribution of UTCI values.

    Args:
        utci_collection (HourlyContinuousCollection):
            An HourlyContinuousCollection containing UTCI.
        ax (plt.Axes, optional):
            A matplotlib Axes object to plot on. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the bar function.

    Returns:
        plt.Axes:
            A matplotlib Axes object.
    """

    title = kwargs.get("title", None)

    series = collection_to_series(utci_collection)

    if ax is None:
        ax = plt.gca()

    # Add stacked plot
    t = pd.cut(series, [-100] + UTCI_LEVELS + [100], labels=UTCI_LABELS)
    t = t.groupby([t.index.month, t]).count().unstack().T
    t = t / t.sum()
    months = [calendar.month_abbr[i] for i in range(1, 13, 1)]
    t.T.plot.bar(
        ax=ax,
        stacked=True,
        color=[rgb2hex(UTCI_COLORMAP.get_under())]
        + UTCI_COLORMAP.colors
        + [rgb2hex(UTCI_COLORMAP.get_over())],
        legend=False,
        width=1,
    )
    ax.set_xlabel(None)
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(months, ha="center", rotation=0, color="k")
    plt.setp(ax.get_yticklabels(), color="k")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("k")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))

    # # Add header percentages for bar plot
    cold_stress = t.T.filter(regex="Cold").sum(axis=1)
    heat_stress = t.T.filter(regex="Heat").sum(axis=1)
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

    return ax


def skymatrix(
    epw: EPW,
    ax: plt.Axes = None,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    density: int = 1,
    show_title: bool = True,
    show_colorbar: bool = True,
    **kwargs,
) -> plt.Axes:
    """Create a sky matrix image.

    Args:
        epw (EPW):
            A EPW object.
        analysis_period (AnalysisPeriod, optional):
            An AnalysisPeriod. Defaults to AnalysisPeriod().
        density (int, optional):
            Sky matrix density. Defaults to 1.
        show_title (bool, optional):
            Show the title. Defaults to True.
        show_colorbar (bool, optional):
            Show the colorbar. Defaults to True.
        **kwargs:
            Additional keyword arguments to pass to the plotting function.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if ax is None:
        ax = plt.gca()

    cmap = kwargs.get("cmap", "viridis")

    # create wea
    wea = Wea.from_epw_file(
        epw.file_path, analysis_period.timestep
    ).filter_by_analysis_period(analysis_period)
    wea_duration = len(wea) / wea.timestep
    wea_folder = Path(tempfile.gettempdir())
    wea_path = wea_folder / "skymatrix.wea"
    wea_file = wea.write(wea_path.as_posix())

    # run gendaymtx
    gendaymtx_exe = (Path(HBR_FOLDERS.radbin_path) / "gendaymtx.exe").as_posix()
    cmds = [gendaymtx_exe, "-m", str(density), "-d", "-O1", "-A", wea_file]
    with subprocess.Popen(cmds, stdout=subprocess.PIPE, shell=True) as process:
        stdout = process.communicate()
    dir_data_str = stdout[0].decode("ascii")
    cmds = [gendaymtx_exe, "-m", str(density), "-s", "-O1", "-A", wea_file]
    with subprocess.Popen(cmds, stdout=subprocess.PIPE, shell=True) as process:
        stdout = process.communicate()
    diff_data_str = stdout[0].decode("ascii")

    def _broadband_rad(data_str: str) -> List[float]:
        _ = data_str.split("\r\n")[:8]
        data = np.array(
            [[float(j) for j in i.split()] for i in data_str.split("\r\n")[8:]][1:-1]
        )
        patch_values = (np.array([0.265074126, 0.670114631, 0.064811243]) * data).sum(
            axis=1
        )
        patch_steradians = np.array(ViewSphere().dome_patch_weights(density))
        broadband_radiation = patch_values * patch_steradians * wea_duration / 1000
        return broadband_radiation

    dir_vals = _broadband_rad(dir_data_str)
    diff_vals = _broadband_rad(diff_data_str)

    # create patches to plot
    patches = []
    for face in ViewSphere().dome_patches(density)[0].face_vertices:
        patches.append(mpatches.Polygon(np.array([i.to_array() for i in face])[:, :2]))
    p = PatchCollection(patches, alpha=1, cmap=cmap)

    p.set_array(dir_vals + diff_vals)  # SET DIR/DIFF/TOTAL VALUES HERE

    # plot!
    ax.add_collection(p)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    if show_colorbar:
        cbar = plt.colorbar(p, ax=ax)
        cbar.outline.set_visible(False)
        cbar.set_label("Cumulative irradiance (W/m$^{2}$)")
    ax.set_aspect("equal")
    ax.axis("off")

    if show_title:
        ax.set_title(
            f"{location_to_string(epw.location)}\n{describe_analysis_period(analysis_period)}",
            ha="left",
            x=0,
        )

    plt.tight_layout()

    return ax


# TODO - add Climate Consultant-style "wind wheel" plot here (http://2.bp.blogspot.com/-F27rpZL4VSs/VngYxXsYaTI/AAAAAAAACAc/yoGXmk13uf8/s1600/CC-graphics%2B-%2BWind%2BWheel.jpg)


def wind_cumulative_probability(
    wind_speeds: Tuple[float],
    ax: plt.Axes = None,
    speed_bins: Tuple[float] = None,
    percentiles: Tuple[float] = None,
    **kwargs,
) -> plt.Axes:
    """Plot a cumulative probability graph, showing binned wind speeds in a cumulative
        frequency histogram.

    Args:
        wind_speeds (List[float]):
            A list of wind speeds.
        ax (plt.Axes, optional):
            A matplotlib Axes object. Defaults to None.
        speed_bins (List[float], optional):
            A set of bin edges to categorise the wind speeds into. Defaults to None.
        percentiles (List[float], optional):
            A list of percentiles to show on the chart. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the plotting function.

    Returns:
        plt.Figure:
            A Figure object.
    """
    if ax is None:
        ax = plt.gca()

    title = kwargs.get("title", None)

    if percentiles is None:
        percentiles = [0.5, 0.95]
    if (min(percentiles) < 0) or (max(percentiles) > 1):
        raise ValueError("percentiles must fall within the range 0-1.")

    if speed_bins is None:
        speed_bins = np.linspace(0, 25, 50)
    if min(speed_bins) < 0:
        raise ValueError("Minimum bin value must be >= 0")

    x = speed_bins
    y = [stats.percentileofscore(wind_speeds, i) / 100 for i in speed_bins]

    # remove all but one element from lists where probability is == 1
    if len([i for i in y if i == 1]) > 2:
        idxmax = np.where([i == 1 for i in y])[0][0]
        x = x[: idxmax + 1]
        y = y[: idxmax + 1]

    ax.plot(x, y, c="grey")
    ax.set_xlim(0, max(x))
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=0))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.25)

    for percentile in percentiles:
        val = np.quantile(wind_speeds, percentile)
        ax.hlines(percentile, 0, val, ls="--", lw=1, colors="k")
        ax.vlines(val, 0, percentile, ls="--", lw=1, colors="k")
        ax.text(val + 0.1, 0, f"{val:0.1f}m/s", ha="left", va="bottom")
        ax.text(
            0.1,
            percentile + 0.02 if percentile < 0.1 else percentile - 0.02,
            f"{percentile:0.0%}",
            ha="left",
            va="bottom" if percentile < 0.1 else "top",
        )

    ax.set_ylim(0, 1.01)

    if title is not None:
        ax.set_title(title, x=0, ha="left", va="bottom")

    plt.tight_layout()

    return ax


def seasonal_speed(
    wind_speeds: pd.Series,
    ax: plt.Axes = None,
    percentiles: Tuple[float] = (0.1, 0.25, 0.75, 0.9),  # type: ignore
    color: str = "k",
    title: str = "",
    **kwargs,
) -> plt.Axes:  # type: ignore
    """Plot the wind-speed/frequency histogram for collection of wind speeds.

    Args:
        wind_speeds (pd.Series):
            A time-indexed collection of wind speeds.
        percentiles (Tuple[float], optional):
            A list of percentiles to show on the chart. Defaults to (0.1, 0.25, 0.75, 0.9).
        color (str, optional):
            The color of the plot. Defaults to "k".
        title (str, optional):
            A title to add to the resulting plot. Defaults to None.

    Returns:
        plt.Figure:
            A Figure object.
    """

    # if ax is None:
    #     ax = plt.gca()

    # if speed_bins is None:
    #     speed_bins = np.linspace(min(wind_speeds), np.quantile(wind_speeds, 0.999), 16)

    # if percentiles and ((min(percentiles) < 0) or (max(percentiles) > 1)):
    #     raise ValueError("percentiles must fall within the range 0-1.")

    # ax.hist(wind_speeds, bins=speed_bins, density=True, color="grey")

    # if weibull != False:
    #     if isinstance(weibull, bool):
    #         params = weibull_pdf(wind_speeds)
    #     elif not isinstance(weibull, bool):
    #         params = weibull
    #     new_x = np.linspace(min(speed_bins), max(speed_bins), 100)
    #     new_y = exponweib.pdf(new_x, *params)
    #     ax.plot(new_x, new_y, label="Weibull (PDF)", c="k")

    # low, _ = ax.get_ylim()
    # for percentile in percentiles:
    #     x = np.quantile(wind_speeds, percentile)
    #     ax.axvline(x, 0, 1, ls="--", lw=1, c="k")
    #     ax.text(x - 0.1, low, f"{percentile:0.0%}", ha="right", va="bottom")
    #     ax.text(x + 0.1, low, f"{x:0.1f}m/s", ha="left", va="bottom")

    # ax.set_xlim(0, max(speed_bins))
    # ax.set_xlabel("Wind Speed (m/s)")
    # ax.set_ylabel("Frequency")
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))

    # for spine in ["top", "right"]:
    #     ax.spines[spine].set_visible(False)
    # ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.25)

    # ax.legend(
    #     bbox_to_anchor=(1, 1),
    #     ncol=1,
    #     loc="upper right",
    #     borderaxespad=0.0,
    #     frameon=False,
    #     fontsize="small",
    # )

    # if title is not None:
    #     ax.set_title(title, x=0, ha="left", va="bottom")

    # plt.tight_layout()

    # TODO - reimplement figure and kwargs usage.
    raise NotImplementedError("This function is not fully implemented.")


def wind_speed_frequency(
    wind_speeds: List[float],
    ax: plt.Axes = None,
    speed_bins: List[float] = None,
    weibull: Union[bool, Tuple[float]] = False,
    percentiles: Tuple[float] = (),
    **kwargs,
) -> plt.Axes:
    """Plot the wind-speed/frequency histogram for collection of wind speeds.

    Args:
        wind_speeds (List[float]):
            A collection of wind speeds.
        ax (plt.Axes, optional):
            A matplotlib Axes object to plot on. Defaults to None.
        speed_bins (List[float], optional):
            A set of bins to fit the input wind speeds into. Defaults to None.
        weibull (Union[bool, Tuple[float]], optional):
            Include the weibull curve on the plot. Defaults to False.
        percentiles (Tuple[float], optional):
            A list of percentiles to show on the chart. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the matplotlib hist function.

    Returns:
        plt.Figure:
            A Figure object.
    """

    if ax is None:
        ax = plt.gca()

    title = kwargs.get("title", None)

    if speed_bins is None:
        speed_bins = np.linspace(min(wind_speeds), np.quantile(wind_speeds, 0.999), 16)

    if percentiles and ((min(percentiles) < 0) or (max(percentiles) > 1)):
        raise ValueError("percentiles must fall within the range 0-1.")

    ax.hist(wind_speeds, bins=speed_bins, density=True, color="grey")

    if weibull != False:
        if isinstance(weibull, bool):
            params = weibull_pdf(wind_speeds)
        elif not isinstance(weibull, bool):
            params = weibull
        new_x = np.linspace(min(speed_bins), max(speed_bins), 100)
        new_y = exponweib.pdf(new_x, *params)
        ax.plot(new_x, new_y, label="Weibull (PDF)", c="k")

    low, _ = ax.get_ylim()
    for percentile in percentiles:
        x = np.quantile(wind_speeds, percentile)
        ax.axvline(x, 0, 1, ls="--", lw=1, c="k")
        ax.text(x - 0.1, low, f"{percentile:0.0%}", ha="right", va="bottom")
        ax.text(x + 0.1, low, f"{x:0.1f}m/s", ha="left", va="bottom")

    ax.set_xlim(0, max(speed_bins))
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=1))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.25)

    ax.legend(
        bbox_to_anchor=(1, 1),
        ncol=1,
        loc="upper right",
        borderaxespad=0.0,
        frameon=False,
        fontsize="small",
    )

    if title is not None:
        ax.set_title(title, x=0, ha="left", va="bottom")

    plt.tight_layout()

    return ax


def wind_timeseries(
    wind_speeds: pd.Series,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot a time-series of wind speeds

    Args:
        wind_speeds (pd.Series): A timeseries collection of wind speed data
        ax (plt.Axes, optional): The matplotlib axes to plot on. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the matplotlib plot function.

    Returns:
        plt.Figure: _description_
    """

    # TODO - replace with standard timeseries plot, and implement kwargs.
    title = kwargs.get("title", None)
    color = kwargs.get("color", "viridis")

    if ax is None:
        ax = plt.gca()

    if not isinstance(wind_speeds.index, pd.DatetimeIndex):
        raise ValueError("The wind_speeds given should be datetime-indexed.")

    wind_speeds.plot(ax=ax, c=color, lw=0.5)

    ax.set_xlim(wind_speeds.index.min(), wind_speeds.index.max())
    ax.set_ylabel("Wind Speed (m/s)")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.25)

    if title is not None:
        ax.set_title(title, x=0, color="k", ha="left")

    plt.tight_layout()

    return ax


def wind_windrose(
    wind_direction: List[float],
    ax: plt.Axes = None,
    data: List[float] = None,
    direction_bins: DirectionBins = DirectionBins(),
    data_bins: Union[int, List[float]] = None,
    include_legend: bool = True,
    include_percentages: bool = False,
    **kwargs,
) -> plt.Axes:
    """Plot a windrose for a collection of wind speeds and directions.

    Args:
        wind_directions (List[float]):
            A collection of wind-directions.
        data (List[float]):
            A collection of direction-associated data.
        direction_bins (DirectionBins, optional):
            A DirectionBins object.
        data_bins (List[float], optional):
            Bins to sort data into. Defaults to the boundaries for Beaufort wind conditions.
        include_legend (bool, optional):
            Set to True to include the legend. Defaults to True.
        include_percentages (bool, optional):
            Add bin totals as % to rose. Defaults to False.

    Returns:
        plt.Figure:
            A Figure object.
    """

    # TODO- replace standard windrose with this one, and implement kwargsm nd decouple from .
    title = kwargs.get("title", None)
    cmap = kwargs.get(
        "cmap",
        ListedColormap(
            colors=[
                "#FFFFFF",
                "#CCFFFF",
                "#99FFCC",
                "#99FF99",
                "#99FF66",
                "#99FF00",
                "#CCFF00",
                "#FFFF00",
                "#FFCC00",
                "#FF9900",
                "#FF6600",
                "#FF3300",
                "#FF0000",
            ]
        ),
    )

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # set data binning defaults (beaufort bins)
    if data_bins is None:
        data_bins = BEAUFORT_BINS
    if isinstance(data_bins, int):
        data_bins = np.linspace(min(data), max(data), data_bins + 1).round(1)

    # bin input data
    thetas = np.deg2rad(direction_bins.midpoints)
    width = np.deg2rad(direction_bins.bin_width)
    binned_data = direction_bins.bin_data(wind_direction, data)
    radiis = np.array(
        [
            np.histogram(a=values, bins=data_bins, density=False)[0]
            for _, values in binned_data.items()
        ]
    )
    bottoms = np.vstack([[0] * len(direction_bins.midpoints), radiis.cumsum(axis=1).T])[
        :-1
    ].T
    colors = [cmap(i) for i in np.linspace(0, 1, len(data_bins) - 1)]

    # create figure
    ax.set
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # plot binned data
    for theta, radii, bottom in zip(*[thetas, radiis, bottoms]):
        _ = ax.bar(theta, radii, width=width, bottom=bottom, color=colors, zorder=2)

    if include_percentages:
        percentages = [
            len(vals) / len(wind_direction) for (low, high), vals in binned_data.items()
        ]
        for theta, radii, percentage in zip(*[thetas, radiis.sum(axis=1), percentages]):
            tt = ax.text(
                theta,
                radii,
                f"{percentage:0.1%}",
                fontsize="x-small",
                ha="center",
                va="center",
            )
            tt.set_bbox(dict(facecolor="white", alpha=0.5, linewidth=0))

    # format plot area
    ax.spines["polar"].set_visible(False)
    ax.grid(True, which="both", ls="--", zorder=0, alpha=0.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    plt.setp(ax.get_yticklabels(), fontsize="small")
    ax.set_xticks(np.radians((0, 90, 180, 270)), minor=False)
    ax.set_xticklabels(("N", "E", "S", "W"), minor=False, **{"fontsize": "medium"})
    ax.set_xticks(
        np.radians(
            (22.5, 45, 67.5, 112.5, 135, 157.5, 202.5, 225, 247.5, 292.5, 315, 337.5)
        ),
        minor=True,
    )
    ax.set_xticklabels(
        (
            "NNE",
            "NE",
            "ENE",
            "ESE",
            "SE",
            "SSE",
            "SSW",
            "SW",
            "WSW",
            "WNW",
            "NW",
            "NNW",
        ),
        minor=True,
        **{"fontsize": "x-small"},
    )

    # construct legend
    if include_legend:
        handles = [
            mpatches.Patch(color=colors[n], label=f"{i} to {j}")
            for n, (i, j) in enumerate(rolling_window(data_bins, 2))
        ]
        _ = ax.legend(
            handles=handles,
            bbox_to_anchor=(1.1, 0.5),
            loc="center left",
            ncol=1,
            borderaxespad=0,
            frameon=False,
            fontsize="small",
            title="m/s",
        )

    if title:
        ax.set_title(title, x=0, ha="left", va="bottom")

    plt.tight_layout()

    return ax


def radial_histogram(
    radial_values: List[float],
    radial_bins: List[float],
    values: List[List[float]],
    ax: plt.Axes = None,
    cmap: Union[Colormap, str] = None,
    include_labels: bool = False,
    include_cbar: bool = True,
    cmap_label: str = None,
    cbar_freq: bool = False,
    title: str = None,
    **kwargs,
) -> plt.Axes:
    """Create a radial 2d heatmap-histogram.

    Args:
        radial_values (List[float]): _description_
        radial_bins (List[float]): _description_
        values (List[List[float]]): _description_
        cmap (Union[Colormap, str], optional): _description_. Defaults to None.
        include_labels (bool, optional): _description_. Defaults to False.
        include_cbar (bool, optional): _description_. Defaults to True.
        cmap_label (str, optional): _description_. Defaults to None.
        cbar_freq (bool, optional): _description_. Defaults to False.
        title (str, optional): _description_. Defaults to None.

    Raises:
        NotImplementedError: _description_

    Returns:
        plt.Axes: _description_
    """

    # TODO - deal with kwargs

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # set cmap defaults
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if cmap is None:
        cmap = plt.get_cmap("magma_r")

    # plot figure
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    pc = ax.pcolormesh(
        radial_values,
        radial_bins,
        values,
        cmap=cmap,
        alpha=1,
        ec="none",
        lw=0,
        zorder=0,
    )

    # format plot area
    ax.spines["polar"].set_visible(False)
    ax.grid(True, which="both", ls="--", zorder=0, alpha=0.25, c="k")
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    plt.setp(ax.get_yticklabels(), fontsize="small")
    ax.set_xticks(np.radians((0, 90, 180, 270)), minor=False)
    ax.set_xticklabels(("N", "E", "S", "W"), minor=False, **{"fontsize": "medium"})
    ax.set_xticks(
        np.radians(
            (22.5, 45, 67.5, 112.5, 135, 157.5, 202.5, 225, 247.5, 292.5, 315, 337.5)
        ),
        minor=True,
    )
    ax.set_xticklabels(
        (
            "NNE",
            "NE",
            "ENE",
            "ESE",
            "SE",
            "SSE",
            "SSW",
            "SW",
            "WSW",
            "WNW",
            "NW",
            "NNW",
        ),
        minor=True,
        **{"fontsize": "x-small"},
    )

    if include_cbar:
        cb = plt.colorbar(
            pc,
            ax=ax,
            pad=0.07,
            drawedges=False,
            label=cmap_label,
            extend="max",
            aspect=50,
        )
        cb.outline.set_visible(False)
        if cbar_freq:
            cb.ax.axes.yaxis.set_major_formatter(mticker.PercentFormatter(1))

    if include_labels:
        raise NotImplementedError("TODO - add functionality")

    if title:
        ax.set_title(title, x=0, ha="left", va="bottom")

    plt.tight_layout()

    return ax


def windrose(
    epw: EPW,
    collection: HourlyContinuousCollection,
    ax: plt.Axes = None,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    directions: int = 12,
    bins: List[float] = None,
    hide_label_legend: bool = False,
    **kwargs,
) -> plt.Axes:
    """Generate a wind-rose plot for the given wind directions and variable.

    Args:
        epw (EPW):
            A ladybug EPW object.
        collection (HourlyContinuousCollection):
            Annual hourly variables to bin and color. Typically this is wind speed, but can be
            another variable instead.
        analysis_period (AnalysisPeriod, optional):
            An analysis period within which to assess the input data. Defaults to Annual.
        directions (int, optional):
            The number of directions to bin wind-direction into. Defaults to 12.
        bins (List[float], optional):
            A set of bins into which data will be binned. Defaults to None.
        hide_label_legend (bool, optional):
            Hide the label and legend. Defaults to False.
        **kwargs:
            Additional keyword arguments are passed to the matplotlib plot.

    Returns:
        plt.Axes:
            A matplotlib Axes object.
    """

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    cmap = kwargs.get("cmap", "jet")
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    norm = kwargs.get("norm", None)
    title = kwargs.get("title", None)
    if bins is None:
        bins = np.linspace(collection.min, collection.max, 11)

    colors = [cmap(i) for i in np.linspace(0, 1, len(bins))]

    ws_values = collection_to_array(
        epw.wind_speed.filter_by_analysis_period(analysis_period)
    )
    n_calm_hours = sum(ws_values == 0)

    not_calm: bool = collection_to_array(epw.wind_speed) > 0
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
                f"{collection_to_series(collection).name} for {get_filename(epw)}",
                describe_analysis_period(analysis_period),
                f"Calm for {n_calm_hours / len(ws_values):0.2%} of the time ({n_calm_hours} hours)",
            ]
        )

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
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        colorbar = plt.colorbar(
            sm,
            ticks=bins,
            boundaries=bins,
            ax=ax,
            label=collection_to_series(collection).name,
        )
        colorbar.outline.set_visible(False)

        ax.set_title(title, ha="left", x=0)

    plt.tight_layout()

    return ax


def _add_value_labels(ax: plt.Axes, spacing: float = 5) -> None:
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes):
            The matplotlib object containing the axes of the plot to annotate.
        spacing (float, optional):
            The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing

        # Vertical alignment for positive values
        va = "bottom"

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = "top"

        # Use Y value as label and format number with one decimal place
        label = f"{y_value:.0f}"

        # Create annotation
        ax.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(0, space),  # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha="center",  # Horizontally center label
            va=va,
        )  # Vertically align label differently for
        # positive and negative values.


def cooling_degree_days(
    epw: EPW, ax: plt.Axes = None, cool_base: float = 23, **kwargs
) -> plt.Axes:
    """Plot the cooling degree days from a given EPW
    object.

    Args:
        epw (EPW):
            An EPW object.
        ax (plt.Axes, optional):
            A matplotlib Axes object. Defaults to None.
        cool_base (float, optional):
            The temperature at which cooling kicks in. Defaults to 23.
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
    ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.2)
    _add_value_labels(ax)

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
    ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.2)
    _add_value_labels(ax)

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


def utci_shade_benefit(
    utci_shade_benefit_categories: pd.Series, **kwargs
) -> plt.Figure:
    """Plot the shade benefit category.

    Args:
        utci_shade_benefit_categories (pd.Series):
            A series containing shade benefit categories.
        **kwargs:
            Optional arguments to pass to the plot. These can include:
                - title (str, optional):
                    A title to add to the plot. Defaults to None.
                - epw (EPW, optional):
                    If included, plot the sun up hours. Defaults to None.

    Returns:
        plt.Figure:
            A figure object.
    """

    warnings.warn(
        "This method is mostly broken, though it *nearly* works. Included here just to inspire someone to fix it! Please."
    )

    if not isinstance(utci_shade_benefit_categories, pd.Series):
        raise ValueError(
            f"shade_benefit_categories must be of type pd.Series, it is currently {type(utci_shade_benefit_categories)}"
        )

    epw = kwargs.get("epw", None)
    title = kwargs.get("title", None)

    if epw is not None:
        if not isinstance(epw, EPW):
            raise ValueError(
                f"include_sun must be of type EPW, it is currently {type(epw)}"
            )
        if len(epw.dry_bulb_temperature) != len(utci_shade_benefit_categories):
            raise ValueError(
                f"Input sizes do not match ({len(utci_shade_benefit_categories)} != {len(epw.dry_bulb_temperature)})"
            )

    # convert values into categories
    cat = pd.Categorical(utci_shade_benefit_categories)

    # get numeric values
    numeric = pd.Series(cat.codes, index=utci_shade_benefit_categories.index)

    # create colormap
    colors = ["#00A499", "#5D822D", "#EE7837", "#585253"]
    if len(colors) != len(cat.categories):
        raise ValueError(
            f"The number of categories does not match the number of colours in the colormap ({len(colors)} != {len(cat.categories)})."
        )
    cmap = ListedColormap(colors)

    # create tcf_properties
    imshow_properties = {
        "cmap": cmap,
    }

    # create canvas
    fig = plt.figure(constrained_layout=True)
    spec = fig.add_gridspec(
        ncols=1, nrows=3, width_ratios=[1], height_ratios=[4, 2, 0.5], hspace=0.0
    )
    heatmap_ax = fig.add_subplot(spec[0, 0])
    histogram_ax = fig.add_subplot(spec[1, 0])
    cb_ax = fig.add_subplot(spec[2, 0])

    # Add heatmap
    hmap = heatmap(numeric, ax=heatmap_ax, show_colorbar=False, **imshow_properties)

    # add sun up indicator lines
    if epw is not None:
        sp = Sunpath.from_location(epw.location)
        sun_up_down = pd.DataFrame(
            [
                sp.calculate_sunrise_sunset_from_datetime(i)
                for i in utci_shade_benefit_categories.resample("D").count().index
            ]
        ).reset_index(drop=True)
        sun_up_down.index = sun_up_down.index + mdates.date2num(numeric.index.min())
        sunrise = pd.Series(
            data=[
                726449
                + timedelta(hours=i.hour, minutes=i.minute, seconds=i.second).seconds
                / 86400
                for i in sun_up_down.sunrise
            ],
            index=sun_up_down.index,
        )
        sunrise = sunrise.reindex(
            sunrise.index.tolist() + [sunrise.index[-1] + 1]
        ).ffill()
        sunset = pd.Series(
            data=[
                726449
                + timedelta(hours=i.hour, minutes=i.minute, seconds=i.second).seconds
                / 86400
                for i in sun_up_down.sunset
            ],
            index=sun_up_down.index,
        )
        sunset = sunset.reindex(sunset.index.tolist() + [sunset.index[-1] + 1]).ffill()
        for s in [sunrise, sunset]:
            heatmap_ax.plot(s.index, s.values, zorder=9, c="#F0AC1B", lw=1)

    # Add colorbar legend and text descriptors for comfort bands
    ticks = np.linspace(0, len(cat.categories), (len(cat.categories) * 2) + 1)[1::2]
    # cb = fig.colorbar(
    #     hmap,
    #     ax=heatmap_ax,
    #     cax=cb_ax,
    #     orientation="horizontal",
    #     ticks=ticks,
    #     drawedges=False,
    # )
    # cb.outline.set_visible(False)
    # plt.setp(plt.getp(cb_ax, "xticklabels"), color="none")
    # cb.set_ticks([])

    # Add labels to the colorbar
    tick_locs = np.linspace(0, len(cat.categories) - 1, len(cat.categories) + 1)
    tick_locs = (tick_locs[1:] + tick_locs[:-1]) / 2
    category_percentages = (
        utci_shade_benefit_categories.value_counts()
        / utci_shade_benefit_categories.count()
    )
    for n, (tick_loc, category) in enumerate(zip(*[tick_locs, cat.categories])):
        cb_ax.text(
            tick_loc,
            1.05,
            textwrap.fill(category, 15) + f"\n{category_percentages[n]:0.0%}",
            ha="center",
            va="bottom",
            size="small",
        )

    # Add stacked plot
    t = utci_shade_benefit_categories
    t = t.groupby([t.index.month, t]).count().unstack().T
    t = t / t.sum()
    months = [calendar.month_abbr[i] for i in range(1, 13, 1)]
    t.T.plot.bar(
        ax=histogram_ax,
        stacked=True,
        color=colors,
        legend=False,
        width=1,
    )
    histogram_ax.set_xlabel(None)
    histogram_ax.set_xlim(-0.5, 11.5)
    histogram_ax.set_ylim(0, 1)
    histogram_ax.set_xticklabels(months, ha="center", rotation=0, color="k")
    plt.setp(histogram_ax.get_yticklabels(), color="k")
    for spine in ["top", "right"]:
        histogram_ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        histogram_ax.spines[spine].set_color("k")
    histogram_ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))

    # # Add header percentages for bar plot
    for month, row in (
        (
            utci_shade_benefit_categories.groupby(
                utci_shade_benefit_categories.index.month
            )
            .value_counts()
            .unstack()
            .T
            / utci_shade_benefit_categories.groupby(
                utci_shade_benefit_categories.index.month
            ).count()
        )
        .T.fillna(0)
        .iterrows()
    ):
        txt = ""
        for n, val in enumerate(row.values[::-1]):
            txtx = f"{val:0.0%}{txt}"
            histogram_ax.text(
                month - 1,
                1.02,
                txtx,
                va="bottom",
                ha="center",
                color=colors[::-1][n],
                fontsize="small",
            )
            txt += "\n"

    title_base = "Shade benefit"
    if title is None:
        heatmap_ax.set_title(title_base, color="k", y=1, ha="left", va="bottom", x=0)
    else:
        heatmap_ax.set_title(
            f"{title_base}\n{title}",
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )

    return fig
