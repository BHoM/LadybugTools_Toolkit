from ladybug.datacollection import HourlyContinuousCollection
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.plot.colormaps import UTCI_BOUNDARYNORM, UTCI_COLORMAP
from ladybugtools_toolkit.plot.timeseries_heatmap import timeseries_heatmap
from matplotlib.figure import Figure


from python_toolkit.bhom.analytics import analytics


@analytics
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
