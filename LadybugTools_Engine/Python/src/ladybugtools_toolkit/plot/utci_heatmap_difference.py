from typing import Tuple

from ladybug.datacollection import HourlyContinuousCollection
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.plot.colormap_sequential import colormap_sequential
from ladybugtools_toolkit.plot.timeseries_heatmap import timeseries_heatmap
from matplotlib.figure import Figure


from ladybugtools_toolkit import analytics


@analytics
def utci_heatmap_difference(
    utci_collection1: HourlyContinuousCollection,
    utci_collection2: HourlyContinuousCollection,
    title: str = None,
    vlims: Tuple[float] = (-10, 10),
) -> Figure:
    """Create a heatmap showing the annual hourly UTCI difference between collections.

    Args:
        utci_collection1 (HourlyContinuousCollection):
            The first UTCI collection.
        utci_collection2 (HourlyContinuousCollection):
            The second UTCI collection.
        title (str, optional):
            Default is None.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if len(vlims) != 2:
        raise ValueError("vlims must be a list of length == 2")

    cmap = colormap_sequential("#00A9E0", "w", "#702F8A")

    return timeseries_heatmap(
        to_series(utci_collection2) - to_series(utci_collection1),
        cmap=cmap,
        title=title,
        vlims=vlims,
    )
