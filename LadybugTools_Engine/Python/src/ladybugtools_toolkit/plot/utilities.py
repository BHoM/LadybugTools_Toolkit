"""Color handling utilities"""

from ladybug.color import Colorset
from matplotlib.colors import (
    LinearSegmentedColormap
)
from python_toolkit.plot.utilities import (
    animation,
    colormap_sequential,
    relative_luminance,
    contrasting_colour as contrasting_color,
    annotate_imshow,
    lighten_color,
    create_title,
    average_color,
    base64_to_image,
    image_to_base64,
    figure_to_base64,
    figure_to_image,
    tile_images,
    triangulation_area,
    create_triangulation,
    format_polar_plot
)

def lb_colormap(name: int | str = "original") -> LinearSegmentedColormap:
    """Create a Matplotlib from a colormap provided by Ladybug.

    Args:
        name (int | str, optional):
            The name of the colormap to create. Defaults to "original".

    Raises:
        ValueError:
            If an invalid LB colormap name is provided, return a list of potential values to use.

    Returns:
        LinearSegmentedColormap:
            A Matplotlib colormap object.
    """
    colorset = Colorset()

    cmap_strings = []
    for colormap in dir(colorset):
        if colormap.startswith("_"):
            continue
        if colormap == "ToString":
            continue
        cmap_strings.append(colormap)

    if name not in cmap_strings:
        raise ValueError(f"name must be one of {cmap_strings}")

    lb_cmap = getattr(colorset, name)()
    rgb = [[getattr(rgb, i) / 255 for i in ["r", "g", "b", "a"]] for rgb in lb_cmap]
    rgb = [tuple(i) for i in rgb]
    return colormap_sequential(*rgb)