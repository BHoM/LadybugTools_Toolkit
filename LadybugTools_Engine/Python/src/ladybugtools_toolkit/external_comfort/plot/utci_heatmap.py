from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybugtools_toolkit.external_comfort.plot.colormaps import (
    UTCI_BOUNDARYNORM,
    UTCI_COLORMAP,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from matplotlib.figure import Figure
from python_toolkit.plot.chart.timeseries_heatmap import timeseries_heatmap


def utci_heatmap(
    utci_collection: HourlyContinuousCollection, title: str = None
) -> Figure:
    """Create a heatmap showing the annual hourly UTCI for this HourlyContinuousCollection.

    Args:
        collection (HourlyContinuousCollection): An HourlyContinuousCollection containing UTCI.
        title (str, optional): Default is None.

    Returns:
        Figure: A matplotlib Figure object.
    """

    if not isinstance(utci_collection, UniversalThermalClimateIndex):
        raise ValueError(
            f"Collection given is not of type {UniversalThermalClimateIndex()}"
        )

    return timeseries_heatmap(
        to_series(utci_collection), UTCI_COLORMAP, UTCI_BOUNDARYNORM, title=title
    )
