from ladybug.epw import EPW, EPWFields
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from ladybugtools_toolkit.ladybug_extension.epw import collection_to_series, EPW_PROPERTIES
from ladybugtools_toolkit.bhom.analytics import bhom_analytics
from .base._compare import compare_histogram, compare_line

@bhom_analytics()
def compare_epw_key_hist(
    epws: list[EPW],
    key: str,
    bins: list[float] = None,
    colours: list[str] = None,
    **kwargs
    ) -> plt.Axes:
    """Compare two or more EPW files against each other based upon a single key, to create a histogram plot.
    
    Args:
        epws (list[EPW]):
            list of epw objects to compare
        key (str):
            epw property to use for comparison
        bins (list[float]):
            list of bins to use for the histogram
            defaults to None, which causes the plot to create its own bins using np.linspace between the min and max values.

    Returns:
        plt.Axes:
            a matplotlib Axes object that contains the plotted histogram.
    """

    # done so that keys like "Dry Bulb Temperature" are valid
    key = key.lower().replace(" ", "_")

    if key not in EPW_PROPERTIES:
        raise ValueError(f"The key: {key}, is not a valid epw key. Please select one from the list in: ladybugtools_toolkit.ladybug_extension.epw EPW_PROPERTIES")

    serieses = [collection_to_series(getattr(i, key)) for i in epws]
    
    ax = compare_histogram(serieses, bins, names=[Path(epw.file_path).stem for epw in epws], colours=colours, **kwargs)

    ax.legend()
    ax.set_ylabel("Number of hours (/8760)")
    ax.set_xlabel(serieses[0].name)
    return ax

@bhom_analytics()
def compare_epw_key_line(
    epws: list[EPW],
    key: str,
    ):
    """Compare two or more EPW files against each other based upon a single key, to create a line plot.
    
    Args:
        epws (list[EPW]):
            list of epw objects to compare
        key (str):
            epw property to use for comparison

    Returns:
        plt.Axes:
            a matplotlib Axes object that contains the plotted line chart.
    """
    
    # done so that keys like "Dry Bulb Temperature" are valid
    key = key.lower().replace(" ", "_")

    if key not in EPW_PROPERTIES:
        raise ValueError(f"The key: {key}, is not a valid epw key. Please select one from the list in: ladybugtools_toolkit.ladybug_extension.epw EPW_PROPERTIES")

    serieses = [collection_to_series(getattr(i, key)) for i in epws]

    ax = compare_line(serieses, names=[Path(epw.file_path).stem for epw in epws])
    ax.set_ylabel(serieses[0].name)

    return ax