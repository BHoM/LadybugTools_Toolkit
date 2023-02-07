import textwrap
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection

from ..external_comfort.utci import categorise, utci_comfort_categories
from ..ladybug_extension.analysis_period import describe, to_boolean
from ..ladybug_extension.datacollection import to_series


def utci_comfort_band_comparison(
    utcis: Tuple[HourlyContinuousCollection],
    analysis_periods: Tuple[AnalysisPeriod] = (AnalysisPeriod()),
    identifiers: Tuple[str] = None,
    title: str = None,
    simplified: bool = True,
    comfort_limits: Tuple[float] = (9, 26),
) -> plt.Figure:
    """Create a proportional bar chart showing how differnet UTCI collections compare in terms of time within each simplified comfort band.

    Args:
        utcis (List[HourlyContinuousCollection]):
            A list if UTCI collections.
        analysis_period (AnalysisPeriod, optional):
            An set of analysis periods, where the combination of these is applied to all collections. Defaults to [AnalysisPeriod()].
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

    labels, _ = utci_comfort_categories(
        simplified=simplified, comfort_limits=comfort_limits, rtype="category"
    )
    colors, _ = utci_comfort_categories(
        simplified=simplified, comfort_limits=comfort_limits, rtype="color"
    )

    df = pd.concat(
        [to_series(col) for col in utcis],
        axis=1,
    ).loc[to_boolean(analysis_periods)]
    df.columns = [textwrap.fill(label, 15) for label in identifiers]

    # categorise values into comfort bands
    df_cat = categorise(
        df, fmt="category", simplified=simplified, comfort_limits=comfort_limits
    )

    # get value counts per collection/series
    counts = (
        (df_cat.apply(pd.value_counts) / len(df_cat)).reindex(labels).T.fillna(0)
    )[labels]

    fig, ax = plt.subplots(1, 1, figsize=(2 * len(utcis), 5))
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
            f"{describe(analysis_periods)}",
            y=0.85,
            x=0.04,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize="x-large",
        )
    else:
        plt.suptitle(
            f"{title}\n{describe(analysis_periods)}",
            y=0.85,
            x=0.04,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize="x-large",
        )

    plt.tight_layout()

    return fig
