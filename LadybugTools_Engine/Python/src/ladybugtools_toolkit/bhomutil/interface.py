import tempfile
from pathlib import Path

from ladybug.epw import EPW

from ..helpers import figure_to_base64
from ..ladybug_extension.datacollection import to_series
from ..plot.timeseries_heatmap import timeseries_heatmap


def plot_example(epw_file: Path, output_type: str = "path") -> str:
    """An example method that generates a matplotlib figure object and
    returns either a path to a file on disk, or a base64 encoded string of
    that image."""

    if output_type not in ["path", "base64"]:
        raise ValueError('output_type must be one of "path" or "base64".')

    series = to_series(EPW(epw_file).dry_bulb_temperature)
    fig = timeseries_heatmap(series=series)

    if output_type == "path":
        save_path = Path(tempfile.gettempdir()) / "example.png"
        fig.savefig(save_path, transparent=True)
        return save_path.as_posix()

    return figure_to_base64(fig)
