import textwrap
import matplotlib as mpl

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybug.analysisperiod import AnalysisPeriod
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.plot.colormaps_local import (
    UTCI_LOCAL_BOUNDARYNORM,
    UTCI_LOCAL_BOUNDARYNORM_IP,
    UTCI_LOCAL_COLORMAP,
    UTCI_LOCAL_LABELS,
    UTCI_LOCAL_LEVELS,
    UTCI_LOCAL_LEVELS_IP,
)
from matplotlib.colors import rgb2hex
from matplotlib.figure import Figure
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List

def utci_stacked_bars(
    collections: HourlyContinuousCollection, collectionNames: str=None, title: str = None, show_legend: bool = True, IP: bool = True, masked: bool = True, analysis_period: AnalysisPeriod = AnalysisPeriod(st_hour=0,end_hour=23)
) -> Figure:
    """Create a histogram showing the annual hourly UTCI values associated with this Typology.

    Args:
        collection (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object.
        collectionNames (list(str), optional): 
            A list of collection names
        title (str, optional):
            A title to add to the resulting figure. Default is None.
        show_legend (bool, optional):
            Set to True to plot the legend. Default is True.
        IP (bool, optional):
            Convert data to IP unit. Default is True.
        masekd (bool, optional):
            Set to True to mask UTCI with ananlysis period. Default is True.
        analysis_period (AnalysisPeriod, optional):
            A ladybug analysis period.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if not isinstance(collections[0].header.data_type, UniversalThermalClimateIndex):
        raise ValueError(
            "Collection data type is not UTCI and cannot be used in this plot."
        )
    # Convert data to IP unit
    if IP:
        collections = [x.to_ip() for x in collections]
    if collectionNames is None:
        collectionNames = map(str, np.arange(len(collections))) 

    df = pd.DataFrame(index = UTCI_LOCAL_LABELS)
    df_cols = collectionNames

    # Instantiate figure
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    spec = fig.add_gridspec(
        ncols=1, nrows=1, width_ratios=[1], height_ratios=[1], hspace=0.0
    )
    bar_ax = fig.add_subplot(spec[0, 0])
    bottom = np.zeros(len(df_cols))

    # Construct DF series
    for i in range(len(df_cols)):
        series_adjust = to_series(collections[i].filter_by_analysis_period(analysis_period))
        series_cut = pd.cut(series_adjust, bins=[-100] + UTCI_LOCAL_LEVELS_IP + [200], labels=UTCI_LOCAL_LABELS)
        sizes = (series_cut.value_counts() / len(series_adjust))[UTCI_LOCAL_LABELS]
        colors = (
            [UTCI_LOCAL_COLORMAP.get_under()] + UTCI_LOCAL_COLORMAP.colors + [UTCI_LOCAL_COLORMAP.get_over()]
        )
        # print(sizes)
        catagories = sizes.index.tolist()
        df[df_cols[i]] = sizes.values
    #
    df = df.T.reset_index()

    df = df.rename(columns={'index': 'Location'})

    for i in range(len(df.columns)-1):
        # print(col)
        
        # if i != 0:
            # print(df[df.columns[i]])
            # print(df[col])
        val = df[df.columns[i+1]]*100
        bar_ax.bar(df_cols,val,bottom=bottom,color=colors[i], label=catagories[i])
        bottom += val

        # bar_ax.get_yaxis().set_visible(False)
        # bar_ax.legend(loc="upper right")
    if show_legend:
        bar_ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)
    # bar_ax.spines['bottom'].set_visible(False)
    # bar_ax.spines['left'].set_visible(False)

    
    for i in range(len(bar_ax.patches)):
        # if round(df[df.columns[i+1]]*100,0) > 1:
        if round(bar_ax.patches[i].get_height(), 0) > 1:
            # print(round(sizes[i]*100,0))
            bar_ax.text(bar_ax.patches[i].get_x() + bar_ax.patches[i].get_width() / 2,
            bar_ax.patches[i].get_height() / 2 + bar_ax.patches[i].get_y(),
                str(int(round(bar_ax.patches[i].get_height(), 0))) + "%", ha = 'center',
                color = 'k', size = 12)
    
    if title is None:
        bar_ax.set_title("Design focus period: full day", color="k", size="small")
    else:
        bar_ax.set_title(title, color="k", size="small", y=1.05)
        
    return fig