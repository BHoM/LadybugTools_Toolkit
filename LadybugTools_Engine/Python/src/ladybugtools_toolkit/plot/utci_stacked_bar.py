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

def utci_stacked_bar(
    collections: HourlyContinuousCollection, collectionNames: str=None, title: str = None, show_legend: bool = True, analysis_period: AnalysisPeriod = AnalysisPeriod(), rotation: bool = False
) -> Figure:
    """Create a histogram showing the annual hourly UTCI values associated with this Typology.

    Args:
        collections (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object.
        collectionNames (list(str), optional): 
            A list of collection names
        title (str, optional):
            A title to add to the resulting figure. Default is None.
        show_legend (bool, optional):
            Set to True to plot the legend. Default is True.
        analysis_period (AnalysisPeriod, optional):
            A ladybug analysis period.
        rotation (bool, optional):
            Set to True to rotate the x label text. Default is False.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if not isinstance(collections[0].header.data_type, UniversalThermalClimateIndex):
        raise ValueError(
            "Collection data type is not UTCI and cannot be used in this plot."
        )
    # Convert data to IP unit
    if collectionNames is None:
        collectionNames = list(map(str, np.arange(len(collections))))

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
        series_cut = pd.cut(series_adjust, bins=[-100] + UTCI_LOCAL_LEVELS + [200], labels=UTCI_LOCAL_LABELS)
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
    if rotation:
        bar_ax.set_xticklabels(df_cols, rotation=45, ha='right', fontsize=24)
    else:
        bar_ax.set_xticklabels(df_cols, fontsize=24)
    
    ylabel = np.arange(0,101,20)
    bar_ax.set_yticklabels(ylabel, fontsize=24)
    
    for i in range(len(bar_ax.patches)):
        # if round(df[df.columns[i+1]]*100,0) > 1:
        if round(bar_ax.patches[i].get_height(), 0) > 1:
            # print(round(sizes[i]*100,0))
            bar_ax.text(bar_ax.patches[i].get_x() + bar_ax.patches[i].get_width() / 2,
            bar_ax.patches[i].get_height() / 2 + bar_ax.patches[i].get_y(),
                str(int(round(bar_ax.patches[i].get_height(), 0))) + "%", ha = 'center',
                color = 'k', size = 24)
    
    # Add title
    st_hour = analysis_period.st_hour
    end_hour = analysis_period.end_hour
    st_day = analysis_period.doys_int[0] - 1
    end_day = analysis_period.doys_int[-1] - 1
    end_count = 23 - end_hour - 1
    start_count = 23 - st_hour

    if st_hour == 23:
        st_hour_title = str(st_hour - 12) + " am"
    elif st_hour < 12:
        st_hour_title = str(st_hour) + " am"
    else:
        st_hour_title = str(st_hour - 12) + " pm"

    if end_hour == 23:
        end_hour_title = str(end_hour - 12 + 1) + "am"
    elif end_hour < 12:
        end_hour_title = str(end_hour + 1) + "am"
    else:
        end_hour_title = str(end_hour - 12 + 1) + " pm"
    bar_title = "Design focus period: " + st_hour_title + " to " + end_hour_title

    if (st_hour == 0 and end_hour == 23):
        bar_title = "Design focus period: full day"

    if title is None:
        bar_ax.set_title(bar_title, color="k", size=24, y=1.03)
    else:
        bar_ax.set_title(title, color="k", size=24, y=1.03)

    return fig