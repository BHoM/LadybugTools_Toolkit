"""Methods for plotting sun-paths."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.compass import Compass
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.location import Location
from ladybug.sunpath import Sunpath
from matplotlib.colors import BoundaryNorm, Colormap

from ..bhom import decorator_factory
from ..ladybug_extension.analysisperiod import (
    analysis_period_to_datetimes,
    describe_analysis_period,
)
from ..ladybug_extension.datacollection import collection_to_series
from ..ladybug_extension.location import location_to_string


@decorator_factory()
def sunpath(
    location: Location,
    ax: plt.Axes = None,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    data_collection: HourlyContinuousCollection = None,
    cmap: Colormap | str = "viridis",
    norm: BoundaryNorm = None,
    sun_size: float = 10,
    show_grid: bool = True,
    show_legend: bool = True,
) -> plt.Axes:
    """Plot a sun-path for the given Location and analysis period.
    Args:
        location (Location):
            A ladybug Location object.
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

    sunpath_obj = Sunpath.from_location(location)
    all_suns = [
        sunpath_obj.calculate_sun_from_date_time(i) for i in analysis_period.datetimes
    ]
    suns = [i for i in all_suns if i.altitude > 0]
    suns_x, suns_y = np.array([sun.position_2d().to_array() for sun in suns]).T

    day_suns = []
    for month in [6, 9, 12]:
        date = pd.to_datetime(f"2017-{month:02d}-21")
        day_idx = pd.date_range(date, date + pd.Timedelta(hours=24), freq="1T")
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
            c="black",
            alpha=0.6,
            zorder=1,
            ls=":",
            lw=0.75,
        )

    title_string = "\n".join(
        [
            location_to_string(location),
            describe_analysis_period(analysis_period),
        ]
    )
    ax.set_title(title_string, ha="left", x=0, y=1.05)

    plt.tight_layout()

    return ax
