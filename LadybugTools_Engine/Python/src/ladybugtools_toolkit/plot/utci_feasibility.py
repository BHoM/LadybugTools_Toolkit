import textwrap
from calendar import month_abbr
from typing import Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from ladybug.datacollection import AnalysisPeriod, HourlyContinuousCollection
from ladybug.epw import EPW
from matplotlib.figure import Figure

from ..external_comfort.utci import (
    UniversalThermalClimateIndex,
    feasible_comfort_category,
    utci_comfort_categories,
)
from ..ladybug_extension.analysis_period import describe as describe_ap
from ..ladybug_extension.datacollection import to_series
from ..ladybug_extension.location import to_string
from .colormaps import UTCI_BOUNDARYNORM, UTCI_COLORMAP
from .lighten_color import lighten_color
from .timeseries_heatmap import timeseries_heatmap


def utci_feasibility(
    epw: EPW,
    simplified: bool = False,
    comfort_limits: tuple = (9, 26),
    included_additional_moisture: bool = False,
    analysis_periods: Union[AnalysisPeriod, Tuple[AnalysisPeriod]] = (AnalysisPeriod()),
    met_rate_adjustment: float = None,
) -> Figure:
    """Plot the UTCI feasibility for each month of the year.

    Args:
        epw (EPW):
            An EPW object.
        simplified (bool, optional):
            Default is False.
        comfort_limits (tuple, optional):
            Default is (9, 26). Only used if simplified is True.
        included_additional_moisture (bool, optional):
            Default is False. If True, then include evap cooling in this analysis.
    Returns:
        Figure:
            A matplotlib Figure object.
    """

    df = feasible_comfort_category(
        epw,
        simplified=simplified,
        comfort_limits=comfort_limits,
        include_additional_moisture=included_additional_moisture,
        analysis_periods=analysis_periods,
        met_rate_adjustment_value=met_rate_adjustment,
    )

    labels, _ = utci_comfort_categories(
        simplified=simplified,
        comfort_limits=comfort_limits,
        rtype="category",
    )
    colors, _ = utci_comfort_categories(
        simplified=simplified,
        comfort_limits=comfort_limits,
        rtype="color",
    )

    fig, axes = plt.subplots(1, 12, figsize=(10, 4), sharey=True, sharex=False)

    ypos = range(len(df))
    for n, ax in enumerate(axes):

        # get values
        low = df.iloc[n].filter(regex="lowest")
        high = df.iloc[n].filter(regex="highest")
        ypos = range(len(low))

        ax.barh(
            ypos,
            width=high.values - low.values,
            left=low.values,
            color=colors,
            zorder=3,
            alpha=0.8,
        )

        for rect in ax.patches:
            width = rect.get_width()
            height = rect.get_height()
            _x = rect.get_x()
            _y = rect.get_y()
            if width == 0:
                if _x == 1:
                    # text saying 100% of hours are in this category
                    ax.text(
                        0.5,
                        _y + (height / 2),
                        textwrap.fill("All times", 15),
                        ha="center",
                        va="center",
                        rotation=0,
                        fontsize="xx-small",
                        zorder=3,
                    )
                continue

            ax.text(
                _x - 0.03,
                _y + (height),
                f"{_x:0.1%}",
                ha="right",
                va="top",
                rotation=90,
                fontsize="xx-small",
                zorder=3,
            )
            ax.text(
                _x + width + 0.03,
                _y + (height),
                f"{_x + width:0.1%}",
                ha="left",
                va="top",
                rotation=90,
                fontsize="xx-small",
                zorder=3,
            )

        if simplified:
            for nn, i in enumerate(colors):
                ax.axhspan(ymin=nn - 0.5, ymax=nn + 0.5, fc=i, alpha=0.2, zorder=1)
        else:
            for nn, i in enumerate(UniversalThermalClimateIndex):
                ax.axhspan(
                    ymin=nn - 0.5, ymax=nn + 0.5, fc=i.color, alpha=0.2, zorder=1
                )

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.5, len(ypos) - 0.5)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelleft=False, left=False)
        ax.set_xticks([-0.1, 0.5, 1.1])
        ax.set_xticklabels(["", month_abbr[n + 1], ""])
        ax.grid(False)

        if n == 5:
            handles = []
            if simplified:
                for col, lab in list(zip(*[["#3C65AF", "#2EB349", "#C31F25"], labels])):
                    handles.append(mpatches.Patch(color=col, label=lab, alpha=0.3))
            else:
                for i in UniversalThermalClimateIndex:
                    handles.append(
                        mpatches.Patch(color=i.color, label=i.value, alpha=0.3)
                    )

            ax.legend(
                handles=handles,
                bbox_to_anchor=(0.5, -0.1),
                loc="upper center",
                ncol=3 if simplified else 4,
                borderaxespad=0,
                frameon=False,
            )

        ti = f"{to_string(epw.location)}\nFeasible ranges of UTCI temperatures ({describe_ap(analysis_periods)})"
        if met_rate_adjustment:
            ti += f" with MET rate adjustment to {met_rate_adjustment} MET"
        plt.suptitle(
            textwrap.fill(ti, 90),
            x=0.075,
            y=0.9,
            ha="left",
            va="bottom",
        )

    plt.tight_layout()
    return fig
