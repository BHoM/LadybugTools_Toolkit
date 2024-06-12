"""Methods for plotting binned data per-month."""
import calendar  # pylint: disable=E0401

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from ..bhom.analytics import bhom_analytics
from ..helpers import validate_timeseries
from .utilities import contrasting_color
from .base._monthly_proportional_histogram import monthly_proportional_histogram


@bhom_analytics()
def monthly_histogram_proportion(
    series: pd.Series,
    bins: list[float],
    ax: plt.Axes = None,
    labels: list[str] = None,
    show_year_in_label: bool = False,
    show_labels: bool = False,
    show_legend: bool = False,
    **kwargs,
) -> plt.Axes:
    """Create a monthly histogram of a pandas Series.

    Args:
        series (pd.Series):
            The pandas Series to plot. Must have a datetime index.
        bins (list[float]):
            The bins to use for the histogram.
        ax (plt.Axes, optional):
            An optional plt.Axes object to populate. Defaults to None, which creates a new plt.Axes object.
        labels (list[str], optional):
            The labels to use for the histogram. Defaults to None, which uses the bin edges.
        show_year_in_label (bool, optional):
            Whether to show the year in the x-axis label. Defaults to False.
        show_labels (bool, optional):
            Whether to show the labels on the bars. Defaults to False.
        show_legend (bool, optional):
            Whether to show the legend. Defaults to False.
        **kwargs:
            Additional keyword arguments to pass to plt.bar.

    Returns:
        plt.Axes:
            The populated plt.Axes object.
    """

    monthly_proportional_histogram(series, bins, ax, labels, show_year_in_label, show_labels, show_legend, **kwargs)

    return ax
