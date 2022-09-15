from pathlib import Path

from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)


from python_toolkit.bhom.analytics import analytics


@analytics
def spatial_metric_filepath(
    simulation_directory: Path, spatial_metric: SpatialMetric
) -> Path:
    """Return the expected filepath for a given SpatialMetric.

    Args:
        simulation_directory (Path):
            A location containing simulated spatial thermal comfort results.
        spatial_metric (SpatialMetric):
            The metric to obtain the filepath for.

    Returns:
        Path:
            The path expected for any stored metric dataset.
    """
    return simulation_directory / f"{spatial_metric.name.lower()}.parquet"
