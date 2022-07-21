import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybug.sunpath import Sunpath
from ladybug.windrose import Compass
from ladybugtools_toolkit.ladybug_extension.analysis_period.describe import (
    describe as describe_analysis_period,
)
from ladybugtools_toolkit.ladybug_extension.analysis_period.to_datetimes import (
    to_datetimes,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.ladybug_extension.location.to_string import (
    to_string as location_to_string,
)
from matplotlib.figure import Figure


def sunpath(
    epw: EPW,
    analysis_period: AnalysisPeriod = None,
    data_collection: HourlyContinuousCollection = None,
    cmap: str = None,
    show_title: bool = True,
) -> Figure:

    if cmap is None:
        cmap = "viridis"

    if analysis_period is None:
        analysis_period = AnalysisPeriod()

    # Create sunpath object
    sunpath = Sunpath.from_location(epw.location)

    # create suns for each hour in analysis period
    suns = [sunpath.calculate_sun_from_hoy(i, False) for i in analysis_period.hoys]

    # Construct sun position dataframe
    df = pd.DataFrame(index=to_datetimes(analysis_period))
    df["altitude_rad"] = [i.altitude_in_radians for i in suns]
    df["altitude_deg"] = [i.altitude for i in suns]
    df["azimuth_rad"] = [i.azimuth_in_radians for i in suns]
    df["azimuth_deg"] = [i.azimuth for i in suns]
    df["apparent_zenith_rad"] = np.pi / 2 - df["altitude_rad"]
    df["apparent_zenith_deg"] = np.degrees(df["apparent_zenith_rad"])
    df = df[df.altitude_deg >= 0]

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax = plt.subplot(1, 1, 1, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rmax(90)
    ax.spines["polar"].set_visible(False)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_facecolor((1, 1, 1, 0))

    # Add ticks
    ax.set_xticks(np.radians(Compass.MAJOR_AZIMUTHS), minor=False)
    ax.set_xticklabels(Compass.MAJOR_TEXT, minor=False, **{"fontsize": "medium"})
    ax.set_xticks(np.radians(Compass.MINOR_AZIMUTHS), minor=True)
    ax.set_xticklabels(Compass.MINOR_TEXT, minor=True, **{"fontsize": "x-small"})

    # Add solstice and equinox lines
    day_lines = {
        f"{'Summer' if epw.location.latitude > 0 else 'Winter'} solstice": pd.to_datetime(
            f"{df.index.year[0]}-06-21"
        ),
        "Equinox": pd.to_datetime(f"{df.index.year[0]}-03-21"),
        f"{'Summer' if epw.location.latitude < 0 else 'Winter'} solstice": pd.to_datetime(
            f"{df.index.year[0]}-12-21"
        ),
    }
    for label, date in day_lines.items():
        day_idx = pd.date_range(
            date, date + pd.Timedelta(hours=24), freq="1T", closed="left"
        )
        day_sun_positions = [sunpath.calculate_sun_from_date_time(i) for i in day_idx]
        day_df = pd.DataFrame(index=day_idx)
        day_df["altitude_rad"] = [i.altitude_in_radians for i in day_sun_positions]
        day_df["azimuth_rad"] = [i.azimuth_in_radians for i in day_sun_positions]
        day_df["apparent_zenith_deg"] = np.degrees(np.pi / 2 - day_df.altitude_rad)
        day_sun_up_mask = day_df.altitude_rad >= 0
        label = label
        ax.plot(
            day_df.azimuth_rad[day_sun_up_mask],
            day_df.apparent_zenith_deg[day_sun_up_mask],
            c="k",
            label=label,
            alpha=0.6,
            zorder=1,
            ls=":",
            lw=0.75,
        )

    # Add suns
    if data_collection is None:
        color = "#FFCF04"
        cmap = None
        main_title = location_to_string(epw.location)
    else:
        series = to_series(data_collection.filter_by_analysis_period(analysis_period))
        color = series.values
        main_title = series.name
        cmap = cmap

    points = ax.scatter(
        df.azimuth_rad,
        df.apparent_zenith_deg,
        s=0.2,
        label=None,
        c=color,
        zorder=5,
        cmap=cmap,
    )

    if data_collection is not None:
        cb = ax.figure.colorbar(
            points,
            pad=0.09,
            shrink=0.8,
            aspect=30,
            label=f"{series.name}",
        )
        cb.outline.set_visible(False)

    # Add title
    if show_title:
        title_string = "\n".join(
            [
                main_title,
                describe_analysis_period(analysis_period),
            ]
        )
        ax.set_title(title_string, ha="left", x=0)

        # ax.legend(
        #     ncol=3,
        #     bbox_to_anchor=(0.5, -0.14),
        #     loc=8,
        #     borderaxespad=0,
        #     frameon=False,
        # )

    plt.tight_layout()

    return fig
