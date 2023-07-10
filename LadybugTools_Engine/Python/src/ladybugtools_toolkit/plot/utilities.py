from typing import List, Tuple, Union

import matplotlib.image as mimage
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from ladybug.color import Colorset
from matplotlib.colors import LinearSegmentedColormap, is_color_like, rgb2hex


def colormap_sequential(
    *colors: Union[str, float, int, Tuple]
) -> LinearSegmentedColormap:
    """
    Create a sequential colormap from a list of input colors.

    Args:
        colors (Union[str, float, int, Tuple]):
            A list of colors according to their hex-code, string name, character code or
            RGBA values.

    Returns:
        LinearSegmentedColormap:
            A matplotlib colormap.

    Examples:
    >> colormap_sequential("green", "#F034A3", (0.5, 0.2, 0.8), "y")
    """
    for color in colors:
        if not isinstance(color, (str, float, int, tuple)):
            raise KeyError(f"{color} not recognised as a valid color string.")

    if len(colors) < 2:
        raise KeyError("Not enough colors input to create a colormap.")

    fixed_colors = []
    for c in colors:
        if is_color_like(c):
            try:
                fixed_colors.append(rgb2hex(c))
            except ValueError:
                fixed_colors.append(c)
        else:
            raise KeyError(f"{c} not recognised as a valid color string.")
    return LinearSegmentedColormap.from_list(
        f"{'_'.join(fixed_colors)}",
        list(zip(np.linspace(0, 1, len(fixed_colors)), fixed_colors)),
        N=256,
    )


def lb_colormap(name: Union[int, str] = "original") -> LinearSegmentedColormap:
    """Create a Matplotlib from a colormap provided by Ladybug.

    Args:
        name (Union[int, str], optional):
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


def annotate_imshow(
    im: mimage.AxesImage,
    data: List[float] = None,
    valfmt: str = "{x:.2f}",
    textcolors: Tuple[str] = ("black", "white"),
    threshold: float = None,
    exclude_vals: List[float] = None,
    **text_kw,
) -> List[str]:
    """A function to annotate a heatmap.

    Args:
        im (AxesImage):
            The AxesImage to be labeled.
        data (List[float], optional):
            Data used to annotate. If None, the image's data is used. Defaults to None.
        valfmt (_type_, optional):
            The format of the annotations inside the heatmap. This should either use the string
            format method, e.g. "$ {x:.2f}", or be a `matplotlib.ticker.Formatter`.
            Defaults to "{x:.2f}".
        textcolors (Tuple[str], optional):
            A pair of colors.  The first is used for values below a threshold, the second for
            those above.. Defaults to ("black", "white").
        threshold (float, optional):
            Value in data units according to which the colors from textcolors are applied. If None
            (the default) uses the middle of the colormap as separation. Defaults to None.
        exclude_vals (float, optional):
            A list of values where text should not be added. Defaults to None.
        **text_kw (dict, optional):
            All other keyword arguments are passed on to the created `~matplotlib.text.Text`

    Returns:
        List[str]:
            The texts added to the AxesImage.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be overwritten by textkw.
    text_kw = {"ha": "center", "va": "center"}
    text_kw.update({"ha": "center", "va": "center"})

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] in exclude_vals:
                pass
            else:
                text_kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **text_kw)
                texts.append(text)

    return texts


def add_bar_labels(ax: plt.Axes, padding: float = 5, vertical: bool = True) -> None:
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes):
            The matplotlib object containing the axes of the plot to annotate.
        padding (float, optional):
            The distance between the labels and the bars.
        vertical (bool, optional):
            If True, the labels are placed above the bars. If False, they are placed to the right

    """

    # For each bar: Place a label
    if vertical:
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = padding

            # Vertical alignment for positive values
            va = "bottom"

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = "top"

            # Use Y value as label and format number with one decimal place
            label = f"{y_value:.0f}"

            # Create annotation
            ax.annotate(
                label,  # Use `label` as label
                (x_value, y_value),  # Place label at end of the bar
                xytext=(0, space),  # Vertically shift label by `space`
                textcoords="offset points",  # Interpret `xytext` as offset in points
                ha="center",  # Horizontally center label
                va=va,
            )  # Vertically align label differently for
            # positive and negative values.
    else:
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_y() + rect.get_height() / 2
            x_value = rect.get_width()

            # Number of points between bar and label. Change to your liking.
            space = padding

            # Horizontal alignment for positive values
            ha = "left"

            # If value of bar is negative: Place label left of bar
            if x_value < 0:
                # Invert space to place label left
                space *= -1
                # Horizontally align label at right
                ha = "right"

            # Use X value as label and format number with one decimal place
            label = f"{x_value:.0f}"

            # Create annotation
            ax.annotate(
                label,  # Use `label` as label
                (x_value, y_value),  # Place label at end of the bar
                xytext=(0, space),  # Vertically shift label by `space`
                textcoords="offset points",  # Interpret `xytext` as offset in points
                ha=ha,
                va="center",  # Vertically center label
            )  # Horizontally align label differently for
            # positive and negative values.


def create_title(text: str, plot_type: str) -> str:
    """Create a title for a plot.

    Args:
        text (str):
            The title of the plot.
        plot_type (str):
            The type of plot.

    Returns:
        str:
            The title of the plot.
    """
    return "\n".join(
        [
            i
            for i in [
                text,
                plot_type,
            ]
            if i is not None
        ]
    )
