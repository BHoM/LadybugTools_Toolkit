from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.compass import Compass
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybug.sunpath import Sunpath
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
from matplotlib.colors import BoundaryNorm, Colormap
from matplotlib.figure import Figure


from ladybugtools_toolkit import analytics


@analytics
def sunpath(
    epw: EPW,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    data_collection: HourlyContinuousCollection = None,
    cmap: Union[Colormap, str] = "viridis",
    norm: BoundaryNorm = None,
    show_title: bool = True,
    sun_size: float = 10,
    show_grid: bool = True,
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
        norm (BoundaryNorm, optional):
            A matploltib BoundaryNorm object containing colormap boundary mapping information.
            Defaults to None.
    Returns:
        Figure:
            A matplotlib Figure object.
    """

    sp = Sunpath.from_location(epw.location)
    all_suns = [sp.calculate_sun_from_date_time(i) for i in analysis_period.datetimes]
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
            s = sp.calculate_sun_from_date_time(idx)
            if s.altitude > 0:
                _.append(np.array(s.position_2d().to_array()))
        day_suns.append(np.array(_))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_xlim(-101, 101)
    ax.set_ylim(-101, 101)
    ax.axis("off")

    if show_grid:
        compass = Compass()
        ax.add_patch(
            plt.Circle(
                (0, 0), 100, zorder=1, lw=0.5, ec="#555555", fc=(0, 0, 0, 0), ls="--"
            )
        )
        pts = compass.minor_azimuth_points + compass.major_azimuth_points
        pt_labels = compass.MINOR_TEXT + compass.MAJOR_TEXT
        for pt, lab in zip(*[pts, pt_labels]):
            _x, _y = np.array([[0, 0]] + [pt.to_array()]).T
            ax.plot(_x, _y, zorder=1, lw=0.5, ls="--", c="#555555")
            ax.text(_x[1], _y[1], lab, ha="center", va="center")

    if data_collection is not None:
        new_idx = to_datetimes(analysis_period)
        series = to_series(data_collection)
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

    return fig
