import matplotlib.pyplot as plt
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybugtools_toolkit.ladybug_extension.analysis_period.describe import (
    describe as describe_analysis_period,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.plot.colormaps import UTCI_COLORMAP, UTCI_LABELS, UTCI_LEVELS
from matplotlib.figure import Figure
from matplotlib.patches import Patch


def utci_pie(
    utci_collection: HourlyContinuousCollection,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    show_legend: bool = True,
    title: str = None,
    show_title: bool = True,
) -> Figure:
    """Create a figure showing the UTCI proportion for the given analysis period.

    Args:
        utci_collection (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object.
        analysis_period (AnalysisPeriod, optional):
            A ladybug analysis period.
        show_legend (bool, optional):
            Set to True to plot the legend also. Default is True.
        title (str, optional):
            Add a title to the plot. Default is None.
        show_title (bool, optional):
            Set to True to show title. Default is True.

    Returns:
        Figure: A matplotlib figure object.
    """

    if not isinstance(utci_collection.header.data_type, UniversalThermalClimateIndex):
        raise ValueError("Input collection is not a UTCI collection.")

    series = to_series(utci_collection.filter_by_analysis_period(analysis_period))

    series_cut = pd.cut(series, bins=[-100] + UTCI_LEVELS + [100], labels=UTCI_LABELS)
    sizes = (series_cut.value_counts() / len(series))[UTCI_LABELS]
    colors = (
        [UTCI_COLORMAP.get_under()] + UTCI_COLORMAP.colors + [UTCI_COLORMAP.get_over()]
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "w", "linewidth": 1},
    )

    centre_circle = plt.Circle((0, 0), 0.60, fc="white")
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    if show_legend:
        # construct custom legend including values
        legend_elements = [
            Patch(
                facecolor=color, edgecolor=None, label=f"[{sizes[label]:05.1%}] {label}"
            )
            for label, color in list(zip(*[UTCI_LABELS, colors]))
        ]
        lgd = ax.legend(handles=legend_elements, loc="center", frameon=False)
        lgd.get_frame().set_facecolor((1, 1, 1, 0))

    ti = f"{series.name}\n{describe_analysis_period(analysis_period, include_timestep=False)}"
    if title is not None:
        ti += "\n" + title

    if show_title:
        ax.set_title(ti, ha="left", va="bottom", x=0.1, y=0.9)

    plt.tight_layout()

    return fig
