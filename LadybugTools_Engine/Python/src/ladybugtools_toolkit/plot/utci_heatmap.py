from ladybug.datacollection import HourlyContinuousCollection
from matplotlib.figure import Figure

from ..ladybug_extension.datacollection import to_series
from .colormaps import UTCI_BOUNDARYNORM, UTCI_COLORMAP
from .timeseries_heatmap import timeseries_heatmap


def utci_heatmap(
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

    return timeseries_heatmap(
        to_series(utci_collection), UTCI_COLORMAP, UTCI_BOUNDARYNORM, title=title
    )
