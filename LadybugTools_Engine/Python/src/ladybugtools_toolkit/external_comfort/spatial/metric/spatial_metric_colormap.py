import matplotlib.pyplot as plt
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)
from ladybugtools_toolkit.plot.colormaps import UTCI_COLORMAP
from matplotlib.colors import Colormap


def spatial_metric_colormap(spatial_metric: SpatialMetric) -> Colormap:
    """Return the associated matplotlib colormap for a given SpatialMetric."""
    if spatial_metric == SpatialMetric.RAD_DIFFUSE:
        return plt.get_cmap("bone_r")
    if spatial_metric == SpatialMetric.RAD_DIRECT:
        return plt.get_cmap("bone_r")
    if spatial_metric == SpatialMetric.RAD_TOTAL:
        return plt.get_cmap("bone_r")
    if spatial_metric == SpatialMetric.DBT_EPW:
        return plt.get_cmap("YlOrRd")
    if spatial_metric == SpatialMetric.RH_EPW:
        return plt.get_cmap("YlGnBu")
    if spatial_metric == SpatialMetric.WD_EPW:
        return plt.get_cmap("twilight")
    if spatial_metric == SpatialMetric.WS_EPW:
        return plt.get_cmap("PuBu")
    if spatial_metric == SpatialMetric.WS_CFD:
        return plt.get_cmap("PuBu")
    if spatial_metric == SpatialMetric.EVAP_CLG:
        return plt.get_cmap("cividis_r")
    if spatial_metric == SpatialMetric.DBT_EVAP:
        return plt.get_cmap("YlOrRd")
    if spatial_metric == SpatialMetric.RH_EVAP:
        return plt.get_cmap("YlGnBu")
    if spatial_metric == SpatialMetric.MRT_INTERPOLATED:
        return plt.get_cmap("inferno")
    if spatial_metric == SpatialMetric.UTCI_CALCULATED:
        return UTCI_COLORMAP
    if spatial_metric == SpatialMetric.UTCI_INTERPOLATED:
        return UTCI_COLORMAP
    if spatial_metric == SpatialMetric.SKY_VIEW:
        return plt.get_cmap("Spectral_r")

    raise NotImplementedError(f"A colormap is not defined for {spatial_metric.name}")
