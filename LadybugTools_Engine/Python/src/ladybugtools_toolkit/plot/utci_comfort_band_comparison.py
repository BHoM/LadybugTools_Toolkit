import textwrap
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection

from ..ladybug_extension.datacollection import to_series
from .colormaps import UTCI_COLORMAP, UTCI_LABELS, UTCI_LEVELS


def utci_comfort_band_comparison_simple(
    utcis: Tuple[HourlyContinuousCollection],
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    identifiers: Tuple[str] = None,
    title: str = None,
    comfort_limits: Tuple[float] = (9, 26),
) -> plt.Figure:
    """Create a proportional bar chart showing how differnet UTCI collections compare in terms of time within each simplified comfort band.

    Args:
        utcis (List[HourlyContinuousCollection]):
            A list if UTCI collections.
        analysis_period (AnalysisPeriod, optional):
            An analysis period ot a[p[ly to all collections. Defaults to AnalysisPeriod().
        identifiers (List[str], optional):
            A list of names to give each collection. Defaults to None.
        title (str, optional):
            An optional title. Defaults to None.
        comfort_limits (List[float], optional):
            Modify the default comfort limits. Defaults to [9, 26].

    Returns:
        plt.Figure:
            A figure object.
    """

    if identifiers is None:
        identifiers = [f"{n}" for n in range(len(utcis))]
    if len(identifiers) != len(utcis):
        raise ValueError(
            "The number of identifiers given does not match the number of UTCI collections given!"
        )

    colors = ["#3C65AF", "#2EB349", "#C31F25"]
    levels = [-100, min(comfort_limits), max(comfort_limits), 100]
    labels = ["Too cold", "Comfortable", "Too hot"]

    df = pd.concat(
        [to_series(col.filter_by_analysis_period(analysis_period)) for col in utcis],
        axis=1,
    )
    df.columns = [textwrap.fill(label, 15) for label in identifiers]

    # cut utci values into bins based on thresholds
    counts = pd.concat(
        [pd.cut(df[i], bins=levels, labels=labels).value_counts() for i in df.columns],
        axis=1,
    )
    counts = counts / counts.sum(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(2 * len(utcis), 5))
    counts.T.plot(ax=ax, kind="bar", stacked=True, color=colors, width=0.8)

    handles, labels = ax.get_legend_handles_labels()
    _ = ax.legend(
        reversed(handles),
        reversed(labels),
        loc="upper left",
        bbox_to_anchor=[1, 1],
        frameon=False,
        fontsize="small",
        ncol=1,
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
        ax.set_title(f"{analysis_period}", y=1, x=0.06, ha="left", va="bottom")
    else:
        ax.set_title(f"{title}\n{analysis_period}", y=1, x=0.06, ha="left", va="bottom")

    plt.tight_layout()

    return fig


def utci_comfort_band_comparison(
    utcis: Tuple[HourlyContinuousCollection],
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    identifiers: Tuple[str] = None,
    title: str = None,
) -> plt.Figure:
    """Plot a collection of UTCIs as thermal comfort category bars.

    Args:
        utcis (List[HourlyContinuousCollection]):
            A list of UTCI collections.
        analysis_period (AnalysisPeriod):
            The analysis period to apply to this comparison.
        identifiers (List[str], optional):
            A set of identifiers to label each column in the bar chart. Defaults to None.
        title (str, optional):
            A title to give this plot.

    Returns:
        plt.Figure:
            A figure object.
    """

    if identifiers is None:
        identifiers = [f"{n}" for n in range(len(utcis))]
    if len(identifiers) != len(utcis):
        raise ValueError(
            "The number of identifiers given does not match the number of UTCI collections given!"
        )

    colors = (
        [UTCI_COLORMAP.get_under()] + UTCI_COLORMAP.colors + [UTCI_COLORMAP.get_over()]
    )

    df = pd.concat(
        [to_series(col.filter_by_analysis_period(analysis_period)) for col in utcis],
        axis=1,
    )
    df.columns = [textwrap.fill(label, 15) for label in identifiers]

    # cut utci values into bins based on thresholds
    counts = pd.concat(
        [
            pd.cut(
                df[i], bins=[-100] + UTCI_LEVELS + [100], labels=UTCI_LABELS
            ).value_counts()
            for i in df.columns
        ],
        axis=1,
    )
    counts = counts / counts.sum(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(2 * len(utcis), 5))
    counts.T.plot(ax=ax, kind="bar", stacked=True, color=colors, width=0.8)

    handles, labels = ax.get_legend_handles_labels()
    _ = ax.legend(
        reversed(handles),
        reversed(labels),
        loc="upper left",
        bbox_to_anchor=[1, 1],
        frameon=False,
        fontsize="small",
        ncol=1,
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
        ax.set_title(f"{analysis_period}", y=1, x=0.06, ha="left", va="bottom")
    else:
        ax.set_title(f"{title}\n{analysis_period}", y=1, x=0.06, ha="left", va="bottom")

    plt.tight_layout()

    return fig
