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
    sun_size: float = 0.2,
    show_legend: bool = True,
) -> Figure:
    """Plot a sun-path for the given EPW and analysis period.
    Args:
        epw (EPW):
            An EPW object.
        analysis_period (AnalysisPeriod, optional):
            _description_. Defaults to None.
        data_collection (HourlyContinuousCollection, optional):
            An aligned data collection. Defaults to None.
        cmap (str, optional):
            The colormap to apply to the aligned data_collection. Defaults to None.
        show_title (bool, optional):
            Set to True to include a title in the plot. Defaults to True.
        show_legend (bool, optional):
            Set to True to include a legend in the plot if data_collection passed. Defaults to True.
        sun_size (float, optional):
            The size of each sun in the plot. Defaults to 0.2.
    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if cmap is None:
        cmap = "viridis"

    if analysis_period is None:
        analysis_period = AnalysisPeriod()

    if (data_collection is not None) and (
        len(data_collection) < len(epw.dry_bulb_temperature)
    ):
        raise ValueError(
            "The data collection passed is not for an entire year and cannot "
            "be used in conjunction with this EPW object.",
        )

    idx = to_datetimes(analysis_period)

    # calculate sun positions for times in analysis_period
    sunpath_obj = Sunpath.from_location(epw.location)
    suns = np.array(
        [sunpath_obj.calculate_sun_from_hoy(i, False) for i in analysis_period.hoys]
    )
    df = pd.DataFrame(index=idx)
    df["altitude_deg"] = np.vectorize(lambda x: x.altitude)(suns)
    df["altitude_rad"] = np.deg2rad(df.altitude_deg)
    df["azimuth_deg"] = np.vectorize(lambda x: x.azimuth)(suns)
    df["azimuth_rad"] = np.deg2rad(df.azimuth_deg)
    df["apparent_zenith_rad"] = np.pi / 2 - df.altitude_rad
    df["apparent_zenith_deg"] = np.rad2deg(df.apparent_zenith_rad)

    # calculate color mapping values if datacollection passed
    if data_collection is not None:
        series = (
            to_series(data_collection).reindex(idx).interpolate()[df.altitude_deg >= 0]
        )
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
    ax.set_yticklabels([])

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
        day_sun_positions = [
            sunpath_obj.calculate_sun_from_date_time(i) for i in day_idx
        ]
        day_df = pd.DataFrame(index=day_idx)
        day_df["altitude_rad"] = [i.altitude_in_radians for i in day_sun_positions]
        day_df["azimuth_rad"] = [i.azimuth_in_radians for i in day_sun_positions]
        day_df["apparent_zenith_deg"] = np.degrees(np.pi / 2 - day_df.altitude_rad)
        day_sun_up_mask = day_df.altitude_rad >= 0
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
        color = series.values
        main_title = series.name

    if data_collection is None:
        points = ax.scatter(
            df.azimuth_rad,
            df.apparent_zenith_deg,
            s=sun_size,
            label=None,
            c=color,
            zorder=5,
            cmap=cmap,
        )
    else:
        temp = pd.DataFrame()
        temp["a"] = df.azimuth_rad
        temp["b"] = df.apparent_zenith_deg
        temp["z"] = color
        temp.sort_values(by="z", inplace=True, ascending=True)
        points = ax.scatter(
            temp["a"].values,
            temp["b"].values,
            s=sun_size,
            label=None,
            c=temp["z"].values,
            zorder=5,
            cmap=cmap,
        )

    if data_collection is not None:
        if show_legend:
            cb = ax.figure.colorbar(
                points,
                pad=0.09,
                shrink=0.8,
                aspect=30,
                label=f"{series.name}",
            )
            cb.outline.set_visible(False)
        else:
            plt.axis("off")
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     # ax.set_zticks([])
        #     ax.grid(False, which="both", ls="--", alpha=0.5)

    # Add title
    if show_title:
        title_string = "\n".join(
            [
                main_title,
                describe_analysis_period(analysis_period),
            ]
        )
        ax.set_title(title_string, ha="left", x=0)

    plt.tight_layout()

    return fig
