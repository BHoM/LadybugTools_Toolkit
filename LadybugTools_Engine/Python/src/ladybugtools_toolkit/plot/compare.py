from ladybug.epw import EPW, EPWFields
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from ..ladybug_extension.epw import collection_to_series, EPW_PROPERTIES
from python_toolkit.bhom.analytics import bhom_analytics
from python_toolkit.plot.timeseries import timeseries

@bhom_analytics()
def compare_epw_key_hist(
    epws: list[EPW],
    key: str,
    bins: list[float] = None,
    ) -> plt.Axes:

    if key not in EPW_PROPERTIES:
        raise ValueError(f"The key: {key}, is not a valid epw key. Please select one from the list in: ladybugtools_toolkit.ladybug_extension.epw EPW_PROPERTIES")

    serieses = [collection_to_series(getattr(i, key)) for i in epws]
    df = pd.concat(serieses, axis=1, keys=[Path(epw.file_path).stem for epw in epws])
    
    if bins is None:
        bins = np.linspace(df.values.min(), df.values.max(), 31)
    elif len(bins) == 0:
        bins = np.linspace(df.values.min(), df.values.max(), 31)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.hist(df.values, bins=bins, label = df.columns, density=False)
    ax.legend()
    ax.set_ylabel("Number of hours (/8760)")
    ax.set_xlabel(serieses[0].name)
    return ax

@bhom_analytics()
def compare_epw_key_line(
    epws: list[EPW],
    key: str,
    ax: plt.Axes = None,
    **kwargs
    ) -> plt.Axes:

    if key not in EPW_PROPERTIES:
        raise ValueError(f"The key: {key}, is not a valid epw key. Please select one from the list in: ladybugtools_toolkit.ladybug_extension.epw EPW_PROPERTIES")

    serieses = [collection_to_series(getattr(epw, key)) for epw in epws]
    ylabel_name = serieses[0].name

    style_context = kwargs.get("style_context", "python_toolkit.bhom")
    with plt.style.context(style_context):
        if ax is None:
            ax = plt.gca()

        for series, epw in zip(serieses, epws):
            ax = timeseries(series, ax=ax, label=Path(epw.file_path).stem, **kwargs)

        ax.set_ylabel(serieses[0].name)
        ax.legend()

    return ax