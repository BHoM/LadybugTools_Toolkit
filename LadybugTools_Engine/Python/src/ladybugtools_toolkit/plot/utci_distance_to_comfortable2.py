from ladybug.datacollection import HourlyContinuousCollection
from matplotlib.figure import Figure

from ..ladybug_extension.datacollection import to_series
from .colormaps import UTCI_BOUNDARYNORM, UTCI_COLORMAP
from .timeseries_heatmap import timeseries_heatmap
from .colormap_sequential import colormap_sequential

import pandas as pd
import numpy as np


def utci_distance_to_comfortable_jr(
    utci_collection: HourlyContinuousCollection, title: str = None
) -> Figure:
    """Create a heatmap showing the annual hourly UTCI for this HourlyContinuousCollection.

    Args:
        collection (HourlyContinuousCollection):
            An HourlyContinuousCollection containing UTCI.
        title (str, optional):
            Default is None.

    Returns:
        Figure:
            A matplotlib Figure object.
    """



    def score_collection_from_utci(utci_collection):
        vals = [score_from_utci(v) for v in to_series(utci_collection)]
        return HourlyContinuousCollection(utci_collection.header, vals)
        

    def score_from_utci(utci_val):
        if utci_val < 9:
            return utci_val - 9
        elif utci_val > 26:
            return utci_val - 26
        
        return 0

    return timeseries_heatmap(
        to_series(score_collection_from_utci(utci_collection)), cmap = colormap_sequential("#00A9E0", "w", "#ba000d"), vlims=(-20,20), title="Distance to comfortable"
    )
