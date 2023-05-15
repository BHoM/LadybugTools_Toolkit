import calendar
import textwrap
from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybug.sunpath import Sunpath
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_shade_benefit(
    shade_benefit_categories: pd.Series, title: str = None, epw: EPW = None
) -> plt.Figure:
    """Plot the shade benefit category.

    Args:
        shade_benefit_categories (pd.Series):
            A series containing shade benefit categories.
        title (str, optional):
            A title to add to the plot. Defaults to None.
        epw (EPW, optional):
            If included, plot the sun up hours. Defaults to None.

    Returns:
        plt.Figure:
            A figure object.
    """

    if not isinstance(shade_benefit_categories, pd.Series):
        raise ValueError(
            f"shade_benefit_categories must be of type pd.Series, it is currently {type(shade_benefit_categories)}"
        )

    if epw is not None:
        if not isinstance(epw, EPW):
            raise ValueError(
                f"include_sun must be of type EPW, it is currently {type(epw)}"
            )
        if len(epw.dry_bulb_temperature) != len(shade_benefit_categories):
            raise ValueError(
                f"Input sizes do not match ({len(shade_benefit_categories)} != {len(epw.dry_bulb_temperature)})"
            )

    # convert values into categories
    cat = pd.Categorical(shade_benefit_categories)

    # get numeric values
    numeric = pd.Series(cat.codes, index=shade_benefit_categories.index)

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
        ncols=1, nrows=2, width_ratios=[1], height_ratios=[4, 2], hspace=0.0
    )
    heatmap_ax = fig.add_subplot(spec[0, 0])
    histogram_ax = fig.add_subplot(spec[1, 0])
    divider = make_axes_locatable(histogram_ax)
    colorbar_ax = divider.append_axes("bottom", size="20%", pad=0.75)

    # Add heatmap
    heatmap = heatmap_ax.imshow(
        pd.pivot_table(
            numeric.to_frame(),
            index=numeric.index.time,
            columns=numeric.index.date,
            values=numeric.name,
        ).values[::-1],
        extent=[
            mdates.date2num(numeric.index.min()),
            mdates.date2num(numeric.index.max()),
            726449,
            726450,
        ],
        aspect="auto",
        interpolation="none",
        **imshow_properties,
    )
    heatmap_ax.xaxis_date()
    heatmap_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    heatmap_ax.yaxis_date()
    heatmap_ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    heatmap_ax.tick_params(labelleft=True, labelright=True, labelbottom=True)
    plt.setp(heatmap_ax.get_xticklabels(), ha="left", color="k")
    plt.setp(heatmap_ax.get_yticklabels(), color="k")
    for spine in ["top", "bottom", "left", "right"]:
        heatmap_ax.spines[spine].set_visible(False)
        heatmap_ax.spines[spine].set_color("k")
    heatmap_ax.grid(visible=True, which="major", color="k", linestyle=":", alpha=0.5)

    # add sun up indicator lines
    if epw is not None:
        sunpath = Sunpath.from_location(epw.location)
        sun_up_down = pd.DataFrame(
            [
                sunpath.calculate_sunrise_sunset_from_datetime(i)
                for i in shade_benefit_categories.resample("D").count().index
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
    cb = fig.colorbar(
        heatmap,
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=ticks,
        drawedges=False,
    )
    cb.outline.set_visible(False)
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="none")
    cb.set_ticks([])

    # Add labels to the colorbar
    tick_locs = np.linspace(0, len(cat.categories) - 1, len(cat.categories) + 1)
    tick_locs = (tick_locs[1:] + tick_locs[:-1]) / 2
    category_percentages = (
        shade_benefit_categories.value_counts() / shade_benefit_categories.count()
    )
    for n, (tick_loc, category) in enumerate(zip(*[tick_locs, cat.categories])):
        colorbar_ax.text(
            tick_loc,
            1.05,
            textwrap.fill(category, 15) + f"\n{category_percentages[n]:0.0%}",
            ha="center",
            va="bottom",
            size="small",
        )

    # Add stacked plot
    t = shade_benefit_categories
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
            shade_benefit_categories.groupby(shade_benefit_categories.index.month)
            .value_counts()
            .unstack()
            .T
            / shade_benefit_categories.groupby(
                shade_benefit_categories.index.month
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
