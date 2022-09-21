from typing import List

import numpy as np
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)
from ladybugtools_toolkit.plot.colormaps import UTCI_LEVELS


from ladybugtools_toolkit import analytics


@analytics
def spatial_metric_levels(spatial_metric: SpatialMetric) -> List[float]:
    """Return the associated matplotlib levels for a given SpatialMetric."""
    if spatial_metric == SpatialMetric.RAD_DIFFUSE:
        return None
    if spatial_metric == SpatialMetric.RAD_DIRECT:
        return None
    if spatial_metric == SpatialMetric.RAD_TOTAL:
        return None
    if spatial_metric == SpatialMetric.DBT_EPW:
        return None
    if spatial_metric == SpatialMetric.RH_EPW:
        return np.linspace(0, 100, 101)
    if spatial_metric == SpatialMetric.WD_EPW:
        return None
    if spatial_metric == SpatialMetric.WS_EPW:
        return None
    if spatial_metric == SpatialMetric.WS_CFD:
        return None
    if spatial_metric == SpatialMetric.EVAP_CLG:
        return np.linspace(0, 1, 101)
    if spatial_metric == SpatialMetric.DBT_EVAP:
        return None
    if spatial_metric == SpatialMetric.RH_EVAP:
        return np.linspace(0, 100, 101)
    if spatial_metric == SpatialMetric.MRT_INTERPOLATED:
        return None
    if spatial_metric == SpatialMetric.UTCI_CALCULATED:
        return UTCI_LEVELS
    if spatial_metric == SpatialMetric.UTCI_INTERPOLATED:
        return UTCI_LEVELS
    if spatial_metric == SpatialMetric.SKY_VIEW:
        return np.linspace(0, 100, 11)

    raise NotImplementedError(f"A levels is not defined for {spatial_metric.name}")
