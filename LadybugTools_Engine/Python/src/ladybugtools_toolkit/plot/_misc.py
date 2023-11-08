"""Miscellaneous plots that don't really fit anywhere else."""
# pylint: disable=line-too-long
from calendar import month_abbr  # pylint: disable=E0401

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import numpy as np
import pandas as pd

from ladybug.epw import EPW, Location
from ladybug.sunpath import Sunpath

from ..ladybug_extension.datacollection import collection_to_series
from ..helpers import sunrise_sunset
from ._heatmap import heatmap
from ..categorical.categories import Categorical


def cloud_cover_categories(epw: EPW, ax: plt.Axes = None) -> plt.Axes:
    """Plot cloud cover categories from an EPW file.

    Args:
        epw (EPW):
            The EPW file to plot.
        ax (plt.Axes, optional):
            A matploltib axes to plot on. Defaults to None.

    Returns:
        plt.Axes:
            The matplotlib axes.
    """
    if ax is None:
        ax = plt.gca()

    s = collection_to_series(epw.opaque_sky_cover)
    mtx = s.groupby([s.index.month, s.index.day]).value_counts().unstack().fillna(0)
    mapper = {
        0: "clear",
        1: "clear",
        2: "mostly clear",
        3: "mostly clear",
        4: "partly cloudy",
        5: "partly cloudy",
        6: "mostly cloudy",
        7: "mostly cloudy",
        8: "overcast",
        9: "overcast",
        10: "overcast",
    }
    mtx = mtx.T.groupby(mapper).sum()[
        ["clear", "mostly clear", "partly cloudy", "mostly cloudy", "overcast"]
    ]
    mtx = (mtx.T / mtx.sum(axis=1)).T

    mtx.plot.area(ax=ax, color=["#95B5DF", "#B1C4DD", "#C9D0D9", "#ACB0B6", "#989CA1"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_xticks(
        ticks=[n for n, i in enumerate(mtx.index) if i[1] == 1],
        labels=[month_abbr[i] for i in range(1, 13, 1)],
        ha="left",
    )
    ax.set_xlim(0, len(mtx))
    ax.set_ylim(0, 1)
    ax.legend(
        bbox_to_anchor=(0.5, -0.05),
        loc="upper center",
        ncol=5,
        title="Cloud cover categories",
    )

    return ax


def hours_sunlight(location: Location, ax: plt.Axes = None) -> plt.Axes:
    """Plot the hours of sunlight for a location.

    Args:
        location (Location):
            The location to plot.
        ax (plt.Axes, optional):
            A matploltib axes to plot on. Defaults to None.

    Returns:
        plt.Axes:
            The matplotlib axes.
    """
    srss_df = sunrise_sunset(location)
    #
    ## hours of daylight
    daylight = pd.Series(
        [i.seconds / (60 * 60) for i in (srss_df["sunset"] - srss_df["sunrise"])],
        name="daylight",
        index=srss_df.index,
    )
    civil_twilight = (
        [
            i.seconds / (60 * 60)
            for i in (srss_df["civil twilight end"] - srss_df["civil twilight start"])
        ]
        - daylight
    ).rename("civil twilight")
    nautical_twilight = (
        [
            i.seconds / (60 * 60)
            for i in (
                srss_df["nautical twilight end"] - srss_df["nautical twilight start"]
            )
        ]
        - daylight
        - civil_twilight
    ).rename("nautical twilight")
    astronomical_twilight = (
        [
            i.seconds / (60 * 60)
            for i in (
                srss_df["astronomical twilight end"]
                - srss_df["astronomical twilight start"]
            )
        ]
        - daylight
        - civil_twilight
        - nautical_twilight
    ).rename("astronomical twilight")
    night = (
        24 - (daylight + civil_twilight + nautical_twilight + astronomical_twilight)
    ).rename("night")

    temp = pd.concat(
        [daylight, civil_twilight, nautical_twilight, astronomical_twilight, night],
        axis=1,
    )
    ax = plt.gca()
    ax.stackplot(
        daylight.index,
        temp.values.T,
        colors=["#FCE49D", "#B9AC86", "#908A7A", "#817F76", "#717171"],
        labels=temp.columns,
    )
    ax.set_title("Hours of daylight and twilight")
    ax.set_ylim(0, 24)
    ax.set_xlim(temp.index.min(), temp.index.max())
    ax.set_ylabel("Hours")

    # plot min/max/med days
    ax.plot(daylight, c="k")

    ax.scatter(temp.daylight.idxmax(), temp.daylight.max(), c="k", s=10)
    ax.text(
        temp.daylight.idxmax(),
        temp.daylight.max() - 0.5,
        f"Summer solstice\n{np.floor(temp.daylight.max()):0.0f} hrs, {(temp.daylight.max() % 1) * 60:0.0f} mins\n{temp.daylight.idxmax():%d %b}",
        ha="center",
        va="top",
    )

    ax.scatter(temp.daylight.idxmin(), temp.daylight.min(), c="k", s=10)
    ax.text(
        temp.daylight.idxmin(),
        temp.daylight.min() - 0.5,
        f"Winter solstice\n{np.floor(temp.daylight.min()):0.0f} hrs, {(temp.daylight.min() % 1) * 60:0.0f} mins\n{temp.daylight.idxmin():%d %b}",
        ha="right" if temp.daylight.idxmin().month > 6 else "left",
        va="top",
    )

    equionoosss = abs((temp.daylight - temp.daylight.median())).sort_values()
    ax.scatter(equionoosss.index[0], temp.daylight[equionoosss.index[0]], c="k", s=10)
    ax.text(
        equionoosss.index[0],
        temp.daylight[equionoosss.index[0]] - 0.5,
        f"Equinox\n{np.floor(temp.daylight[equionoosss.index[0]]):0.0f} hrs, {(temp.daylight[equionoosss.index[0]] % 1) * 60:0.0f} mins\n{equionoosss.index[0]:%d %b}",
        ha="right" if equionoosss.index[0].month > 6 else "left",
        va="top",
    )

    ix = None
    for ix in equionoosss.index:
        if ix.month != equionoosss.index[0].month:
            break
    ax.scatter(ix, temp.daylight[ix], c="k", s=10)
    ax.text(
        ix,
        temp.daylight[ix] - 0.5,
        f"Equinox\n{np.floor(temp.daylight[ix]):0.0f} hrs, {(temp.daylight[ix] % 1) * 60:0.0f} mins\n{ix:%d %b}",
        ha="right" if ix.month > 6 else "left",
        va="top",
    )
    ax.legend(
        bbox_to_anchor=(0.5, -0.05),
        loc="upper center",
        ncol=5,
        title="Day period",
    )

    return ax


def hours_sunrise_sunset(location: Location, ax: plt.Axes = None) -> plt.Axes:
    """Plot the hours of sunrise and sunset for a location.

    Args:
        location (Location):
            The location to plot.
        ax (plt.Axes, optional):
            A matploltib axes to plot on. Defaults to None.

    Returns:
        plt.Axes:
            The matplotlib axes.
    """
    if ax is None:
        ax = plt.gca()

    srss_df = sunrise_sunset(location)
    seconds = srss_df.map(lambda a: ((a - a.normalize()) / pd.Timedelta("1 second")))
    hours = seconds / (60 * 60)

    ## hours of daylight
    daylight = pd.Series(
        [i.seconds / (60 * 60) for i in (srss_df["sunset"] - srss_df["sunrise"])],
        name="daylight",
        index=srss_df.index,
    )
    civil_twilight = (
        [
            i.seconds / (60 * 60)
            for i in (srss_df["civil twilight end"] - srss_df["civil twilight start"])
        ]
        - daylight
    ).rename("civil twilight")
    nautical_twilight = (
        [
            i.seconds / (60 * 60)
            for i in (
                srss_df["nautical twilight end"] - srss_df["nautical twilight start"]
            )
        ]
        - daylight
        - civil_twilight
    ).rename("nautical twilight")
    astronomical_twilight = (
        [
            i.seconds / (60 * 60)
            for i in (
                srss_df["astronomical twilight end"]
                - srss_df["astronomical twilight start"]
            )
        ]
        - daylight
        - civil_twilight
        - nautical_twilight
    ).rename("astronomical twilight")
    night = (
        24 - (daylight + civil_twilight + nautical_twilight + astronomical_twilight)
    ).rename("night")

    temp = pd.concat(
        [daylight, civil_twilight, nautical_twilight, astronomical_twilight, night],
        axis=1,
    )

    ax = plt.gca()
    ax.fill_between(
        hours.index,
        np.zeros_like(hours["astronomical twilight start"]),
        hours["astronomical twilight start"],
        fc="#717171",
        label="Night",
    )
    ax.fill_between(
        hours.index,
        hours["astronomical twilight start"],
        hours["nautical twilight start"],
        fc="#817F76",
        label="Astonomical twilight",
    )
    ax.fill_between(
        hours.index,
        hours["nautical twilight start"],
        hours["civil twilight start"],
        fc="#908A7A",
        label="Nautical twilight",
    )
    ax.fill_between(
        hours.index,
        hours["civil twilight start"],
        hours["sunrise"],
        fc="#B9AC86",
        label="Civil twilight",
    )
    ax.fill_between(
        hours.index, hours["sunrise"], hours["sunset"], fc="#FCE49D", label="Day"
    )
    ax.fill_between(
        hours.index,
        hours["sunset"],
        hours["civil twilight end"],
        fc="#B9AC86",
        label="__nolegend__",
    )
    ax.fill_between(
        hours.index,
        hours["civil twilight end"],
        hours["nautical twilight end"],
        fc="#908A7A",
        label="__nolegend__",
    )
    ax.fill_between(
        hours.index,
        hours["nautical twilight end"],
        hours["astronomical twilight end"],
        fc="#817F76",
        label="__nolegend__",
    )
    ax.fill_between(
        hours.index,
        hours["astronomical twilight end"],
        np.ones_like(hours["astronomical twilight start"]) * 24,
        fc="#717171",
        label="__nolegend__",
    )

    [ax.plot(hours.index, hours[i], c="k", lw=1) for i in ["sunrise", "noon", "sunset"]]

    ax.axvline(temp.daylight.idxmax(), c="k", ls=":", alpha=0.5)
    ax.text(
        temp.daylight.idxmax(),
        hours.noon.loc[temp.daylight.idxmax()] + 0.5,
        f"Summer solstice ({temp.daylight.idxmax():%d %b})\n{np.floor(temp.daylight.max()):02.0f} hrs, {(temp.daylight.max() % 1) * 60:02.0f} mins",
        ha="center",
        va="bottom",
    )
    ax.text(
        temp.daylight.idxmax(),
        hours.sunrise.loc[temp.daylight.idxmax()] + 0.5,
        f"Sunrise\n{int(np.floor((hours.sunrise.loc[temp.daylight.idxmax()]))):02.0f}:{(hours.sunrise.loc[temp.daylight.idxmax()] % 1) * 60:02.0f}",
        ha="center",
        va="bottom",
    )
    ax.text(
        temp.daylight.idxmax(),
        hours.sunset.loc[temp.daylight.idxmax()] + 0.5,
        f"Sunset\n{int(np.floor((hours.sunset.loc[temp.daylight.idxmax()]))):02.0f}:{(hours.sunset.loc[temp.daylight.idxmax()] % 1) * 60:02.0f}",
        ha="center",
        va="bottom",
    )

    ax.axvline(temp.daylight.idxmin(), c="k", ls=":", alpha=0.5)
    ax.text(
        temp.daylight.idxmin(),
        hours.noon.loc[temp.daylight.idxmin()] + 0.5,
        f"Winter solstice ({temp.daylight.idxmin():%d %b})\n{np.floor(temp.daylight.min()):02.0f} hrs, {(temp.daylight.min() % 1) * 60:02.0f} mins",
        ha="right" if temp.daylight.idxmin().month > 6 else "left",
        va="bottom",
    )
    ax.text(
        temp.daylight.idxmin(),
        hours.sunrise.loc[temp.daylight.idxmin()] + 0.5,
        f"Sunrise\n{int(np.floor((hours.sunrise.loc[temp.daylight.idxmin()]))):02.0f}:{(hours.sunrise.loc[temp.daylight.idxmin()] % 1) * 60:02.0f}",
        ha="right" if temp.daylight.idxmin().month > 6 else "left",
        va="bottom",
    )
    ax.text(
        temp.daylight.idxmin(),
        hours.sunset.loc[temp.daylight.idxmin()] + 0.5,
        f"Sunset\n{int(np.floor((hours.sunset.loc[temp.daylight.idxmin()]))):02.0f}:{(hours.sunset.loc[temp.daylight.idxmin()] % 1) * 60:02.0f}",
        ha="right" if temp.daylight.idxmin().month > 6 else "left",
        va="bottom",
    )

    equionoosss = abs((temp.daylight - temp.daylight.median())).sort_values()
    ax.axvline(equionoosss.index[0], c="k", ls=":", alpha=0.5)
    ax.text(
        equionoosss.index[0],
        hours.noon.loc[temp.daylight.idxmin()] + 0.5,
        f"Equinox ({equionoosss.index[0]:%d %b})\n{np.floor(temp.daylight[equionoosss.index[0]]):0.0f} hrs, {(temp.daylight[equionoosss.index[0]] % 1) * 60:0.0f} mins",
        ha="right" if equionoosss.index[0].month > 6 else "left",
        va="bottom",
    )

    ix = None
    for ix in equionoosss.index:
        if ix.month != equionoosss.index[0].month:
            break
    ax.axvline(ix, c="k", ls=":", alpha=0.5)
    ax.text(
        ix,
        hours.noon.loc[ix] + 0.5,
        f"Equinox ({ix:%d %b})\n{np.floor(temp.daylight[ix]):0.0f} hrs, {(temp.daylight[ix] % 1) * 60:0.0f} mins",
        ha="right" if ix.month > 6 else "left",
        va="bottom",
    )

    ax.set_title("Sunrise and sunset")
    ax.set_ylim(0, 24)
    ax.set_xlim(hours.index.min(), hours.index.max())
    ax.set_ylabel("Hours")
    ax.legend(
        bbox_to_anchor=(0.5, -0.05),
        loc="upper center",
        ncol=5,
        title="Day period",
    )

    return ax


def solar_elevation_azimuth(location: Location, ax: plt.Axes = None) -> plt.Axes:
    """Plot the solar elevation and azimuth for a location.

    Args:
        location (Location):
            The location to plot.
        ax (plt.Axes, optional):
            A matploltib axes to plot on. Defaults to None.

    Returns:
        plt.Axes:
            The matplotlib axes.
    """

    cat = Categorical(
        colors=(
            "#809FB4",
            "#90ACBE",
            "#9FC7A2",
            "#90BF94",
            "#9FC7A2",
            "#CF807A",
            "#C86C65",
            "#CF807A",
            "#C6ACA0",
            "#BD9F92",
            "#C6ACA0",
            "#90ACBE",
            "#809FB4",
        ),
        bins=[
            0,
            22.5,
            45,
            67.5,
            112.5,
            135,
            157.5,
            202.5,
            225,
            247.5,
            292.5,
            315,
            337.5,
            360,
        ],
        bin_names=[
            "North",
            "NNE",
            "ENE",
            "East",
            "ESE",
            "SSE",
            "South",
            "SSW",
            "WSW",
            "West",
            "WNW",
            "NNW",
            "North",
        ],
        name="directions",
    )

    if ax is None:
        ax = plt.gca()

    sp = Sunpath.from_location(location)
    idx = pd.date_range("2017-01-01 00:00:00", "2018-01-01 00:00:00", freq="10T")
    suns = [sp.calculate_sun_from_date_time(i) for i in idx]
    a = pd.DataFrame(index=idx)
    a["altitude"] = [i.altitude for i in suns]
    a["azimuth"] = [i.azimuth for i in suns]

    ax = plt.gca()
    heatmap(
        a.azimuth,
        ax=ax,
        cmap=cat.cmap,
        norm=cat.norm,
        title="Solar Elevation and Azimuth",
    )
    cb = ax.collections[-1].colorbar
    cb.set_ticks(
        [0, 45, 90, 135, 180, 225, 270, 315, 360],
        labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"],
    )
    # create matrix of monthday/hour for pcolormesh
    pvt = a.pivot_table(columns=a.index.date, index=a.index.time)

    # plot the contours for sun positions
    x = mdates.date2num(pvt["altitude"].columns)
    y = mdates.date2num(pd.to_datetime([f"2017-01-01 {i}" for i in pvt.index]))
    z = pvt["altitude"].values
    # z = np.ma.masked_array(z, mask=z < 0)
    ct = ax.contour(x, y, z, colors="k", levels=np.arange(0, 91, 10))
    ax.clabel(ct, inline=1, fontsize="small")

    return ax
