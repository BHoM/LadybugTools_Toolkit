from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)
from ladybugtools_toolkit.plot.colormaps import UTCI_BOUNDARYNORM
from matplotlib.colors import BoundaryNorm


from python_toolkit.bhom.analytics import analytics


@analytics
def spatial_metric_boundarynorm(spatial_metric: SpatialMetric) -> BoundaryNorm:
    """Return the associated matplotlib boundarynorm for a given SpatialMetric."""
    if spatial_metric == SpatialMetric.RAD_DIFFUSE:
        return None
    if spatial_metric == SpatialMetric.RAD_DIRECT:
        return None
    if spatial_metric == SpatialMetric.RAD_TOTAL:
        return None
    if spatial_metric == SpatialMetric.DBT_EPW:
        return None
    if spatial_metric == SpatialMetric.RH_EPW:
        return None
    if spatial_metric == SpatialMetric.WD_EPW:
        return None
    if spatial_metric == SpatialMetric.WS_EPW:
        return None
    if spatial_metric == SpatialMetric.WS_CFD:
        return None
    if spatial_metric == SpatialMetric.EVAP_CLG:
        return None
    if spatial_metric == SpatialMetric.DBT_EVAP:
        return None
    if spatial_metric == SpatialMetric.RH_EVAP:
        return None
    if spatial_metric == SpatialMetric.MRT_INTERPOLATED:
        return None
    if spatial_metric == SpatialMetric.UTCI_CALCULATED:
        return UTCI_BOUNDARYNORM
    if spatial_metric == SpatialMetric.UTCI_INTERPOLATED:
        return UTCI_BOUNDARYNORM
    if spatial_metric == SpatialMetric.SKY_VIEW:
        return None

    raise NotImplementedError(
        f"A boundarynorm is not defined for {spatial_metric.name}"
    )
