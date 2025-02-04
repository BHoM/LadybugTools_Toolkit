"""Color handling utilities"""

# pylint: disable=E0401
import base64
import colorsys
import copy
import io
from pathlib import Path
from typing import Any

import matplotlib.image as mimage
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from ladybug.color import Colorset
from matplotlib.colors import (
    LinearSegmentedColormap,
    cnames,
    colorConverter,
    is_color_like,
    rgb2hex,
    to_hex,
    to_rgb,
    to_rgba_array,
)
from matplotlib.tri import Triangulation
from PIL import Image
from python_toolkit.bhom.analytics import bhom_analytics

# pylint: enable=E0401


@bhom_analytics()
def animation(
    images: list[str | Path | Image.Image],
    output_gif: str | Path,
    ms_per_image: int = 333,
    transparency_idx: int = 0,
) -> Path:
    """Create an animated gif from a set of images.

    Args:
        images (list[str | Path | Image.Image]):
            A list of image files or PIL Image objects.
        output_gif (str | Path):
            The output gif file to be created.
        ms_per_image (int, optional):
            Number of milliseconds per image. Default is 333, for 3 images per second.
        transparency_idx (int, optional):
            The index of the color to be used as the transparent color. Default is 0.

    Returns:
        Path:
            The animated gif.

    """
    _images = []
    for i in images:
        if isinstance(i, (str, Path)):
            _images.append(Image.open(i))
        elif isinstance(i, Image.Image):
            _images.append(i)
        else:
            raise ValueError(
                f"images must be a list of strings, Paths or PIL Image objects - {i} is not valid."
            )

    # create white background
    background = Image.new("RGBA", _images[0].size, (255, 255, 255))

    _images = [Image.alpha_composite(background, i) for i in _images]

    _images[0].save(
        output_gif,
        save_all=True,
        append_images=_images[1:],
        optimize=False,
        duration=ms_per_image,
        loop=0,
        disposal=2,
        transparency=transparency_idx,
    )

    return output_gif


def relative_luminance(color: Any):
    """Calculate the relative luminance of a color according to W3C standards

    Args:
        color (Any):
            matplotlib color or sequence of matplotlib colors - Hex code,
            rgb-tuple, or html color name.

    Returns:
        float:
            Luminance value between 0 and 1.
    """
    rgb = colorConverter.to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lum = rgb.dot([0.2126, 0.7152, 0.0722])
    try:
        return lum.item()
    except ValueError:
        return lum


def contrasting_color(color: Any):
    """Calculate the contrasting color for a given color.

    Args:
        color (Any):
            matplotlib color or sequence of matplotlib colors - Hex code,
            rgb-tuple, or html color name.

    Returns:
        str:
            String code of the contrasting color.
    """
    return ".15" if relative_luminance(color) > 0.408 else "w"


def colormap_sequential(
    *colors: str | float | int | tuple, N: int = 256
) -> LinearSegmentedColormap:
    """
    Create a sequential colormap from a list of input colors.

    Args:
        *colors (str | float | int | tuple):
            A list of colors according to their hex-code, string name, character code or
            RGBA values.
        N (int, optional):
            The number of colors in the colormap. Defaults to 256.

    Returns:
        LinearSegmentedColormap:
            A matplotlib colormap.

    Examples:
    >> colormap_sequential(
        (0.89411764705, 0.01176470588, 0.01176470588),
        "darkorange",
        "#FFED00",
        "#008026",
        (36/255, 64/255, 142/255),
        "#732982"
    )
    """

    if len(colors) < 2:
        raise KeyError("Not enough colors input to create a colormap.")

    fixed_colors = []
    for color in colors:
        fixed_colors.append(to_hex(color))

    return LinearSegmentedColormap.from_list(
        name=f"{'_'.join(fixed_colors)}",
        colors=fixed_colors,
        N=N,
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


def annotate_imshow(
    im: mimage.AxesImage,
    data: list[float] = None,
    valfmt: str = "{x:.2f}",
    textcolors: tuple[str] = ("black", "white"),
    threshold: float = None,
    exclude_vals: list[float] = None,
    **text_kw,
) -> list[str]:
    """A function to annotate a heatmap.

    Args:
        im (AxesImage):
            The AxesImage to be labeled.
        data (list[float], optional):
            Data used to annotate. If None, the image's data is used. Defaults to None.
        valfmt (_type_, optional):
            The format of the annotations inside the heatmap. This should either use the string
            format method, e.g. "$ {x:.2f}", or be a `matplotlib.ticker.Formatter`.
            Defaults to "{x:.2f}".
        textcolors (tuple[str], optional):
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
        list[str]:
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


def lighten_color(color: str | tuple, amount: float = 0.5) -> tuple[float]:
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.

    Args:
        color (str):
            A color-like string.
        amount (float):
            The amount of lightening to apply.

    Returns:
        tuple[float]:
            An RGB value.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


@bhom_analytics()
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


@bhom_analytics()
def average_color(colors: Any, keep_alpha: bool = False, weights: list[float] = None) -> str:
    """Return the average color from a list of colors.

    Args:
        colors (Any):
            A list of colors.
        keep_alpha (bool, optional):
            If True, the alpha value of the color is kept. Defaults to False.
        weights (list[float], optional):
            A list of weights for each color. Defaults to None.

    Returns:
        color: str
            The average color in hex format.
    """

    if not isinstance(colors, (list, tuple)):
        raise ValueError("colors must be a list")

    for i in colors:
        if not is_color_like(i):
            raise ValueError(f"colors must be a list of valid colors - '{i}' is not valid.")

    if len(colors) == 1:
        return colors[0]

    return rgb2hex(
        np.average(to_rgba_array(colors), axis=0, weights=weights), keep_alpha=keep_alpha
    )


@bhom_analytics()
def base64_to_image(base64_string: str, image_path: Path) -> Path:
    """Convert a base64 encoded image into a file on disk.

    Arguments:
        base64_string (str):
            A base64 string encoding of an image file.
        image_path (Path):
            The location where the image should be stored.

    Returns:
        Path:
            The path to the image file.
    """

    # remove html pre-amble, if necessary
    if base64_string.startswith("data:image"):
        base64_string = base64_string.split(";")[-1]

    with open(Path(image_path), "wb") as fp:
        fp.write(base64.decodebytes(base64_string))

    return image_path


@bhom_analytics()
def image_to_base64(image_path: Path, html: bool = False) -> str:
    """Load an image file from disk and convert to base64 string.

    Arguments:
        image_path (Path):
            The file path for the image to be converted.
        html (bool, optional):
            Set to True to include the HTML preamble for a base64 encoded image. Default is False.

    Returns:
        str:
            A base64 string encoding of the input image file.
    """

    # convert path string to Path object
    image_path = Path(image_path).absolute()

    # ensure format is supported
    supported_formats = [".png", ".jpg", ".jpeg"]
    if image_path.suffix not in supported_formats:
        raise ValueError(
            f"'{image_path.suffix}' format not supported. Use one of {supported_formats}"
        )

    # load image and convert to base64 string
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")

    if html:
        content_type = f"data:image/{image_path.suffix.replace('.', '')}"
        content_encoding = "utf-8"
        return f"{content_type};charset={content_encoding};base64,{base64_string}"

    return base64_string


@bhom_analytics()
def figure_to_base64(figure: plt.Figure, html: bool = False, transparent: bool = True) -> str:
    """Convert a matplotlib figure object into a base64 string.

    Arguments:
        figure (Figure):
            A matplotlib figure object.
        html (bool, optional):
            Set to True to include the HTML preamble for a base64 encoded image. Default is False.

    Returns:
        str:
            A base64 string encoding of the input figure object.
    """

    buffer = io.BytesIO()
    figure.savefig(buffer, transparent=transparent)
    buffer.seek(0)
    base64_string = base64.b64encode(buffer.read()).decode("utf-8")

    if html:
        content_type = "data:image/png"
        content_encoding = "utf-8"
        return f"{content_type};charset={content_encoding};base64,{base64_string}"

    return base64_string


@bhom_analytics()
def figure_to_image(fig: plt.Figure) -> Image:
    """Convert a matplotlib Figure object into a PIL Image.

    Args:
        fig (Figure):
            A matplotlib Figure object.

    Returns:
        Image:
            A PIL Image.
    """

    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)

    return Image.fromarray(buf)


@bhom_analytics()
def tile_images(imgs: list[Path] | list[Image.Image], rows: int, cols: int) -> Image.Image:
    """Tile a set of images into a grid.

    Args:
        imgs (Union[list[Path], list[Image.Image]]):
            A list of images to tile.
        rows (int):
            The number of rows in the grid.
        cols (int):
            The number of columns in the grid.

    Returns:
        Image.Image:
            A PIL image of the tiled images.
    """

    imgs = np.array([Path(i) for i in np.array(imgs).flatten()])

    # open images if paths passed
    imgs = [Image.open(img) if isinstance(img, Path) else img for img in imgs]

    if len(imgs) != rows * cols:
        raise ValueError(f"The number of images given ({len(imgs)}) does not equal ({rows}*{cols})")

    # ensure each image has the same dimensions
    w, h = imgs[0].size
    for img in imgs:
        if img.size != (w, h):
            raise ValueError("All images must have the same dimensions")

    w, h = imgs[0].size
    grid = Image.new("RGBA", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
        img.close()

    return grid


@bhom_analytics()
def triangulation_area(triang: Triangulation) -> float:
    """Calculate the area of a matplotlib Triangulation.

    Args:
        triang (Triangulation):
            A matplotlib Triangulation object.

    Returns:
        float:
            The area of the Triangulation in the units given.
    """

    triangles = triang.triangles
    x, y = triang.x, triang.y
    a, _ = triangles.shape
    i = np.arange(a)
    area = np.sum(
        np.abs(
            0.5
            * (
                (x[triangles[i, 1]] - x[triangles[i, 0]])
                * (y[triangles[i, 2]] - y[triangles[i, 0]])
                - (x[triangles[i, 2]] - x[triangles[i, 0]])
                * (y[triangles[i, 1]] - y[triangles[i, 0]])
            )
        )
    )

    return area


@bhom_analytics()
def create_triangulation(
    x: list[float],
    y: list[float],
    alpha: float = None,
    max_iterations: int = 250,
    increment: float = 0.01,
) -> Triangulation:
    """Create a matplotlib Triangulation from a list of x and y coordinates, including a mask to
        remove elements with edges larger than alpha.

    Args:
        x (list[float]):
            A list of x coordinates.
        y (list[float]):
            A list of y coordinates.
        alpha (float, optional):
            A value to start alpha at.
            Defaults to None, with an estimate made for a suitable starting point.
        max_iterations (int, optional):
            The number of iterations to run to check against triangulation validity.
            Defaults to 250.
        increment (int, optional):
            The value by which to increment alpha by when searching for a valid triangulation.
            Defaults to 0.01.

    Returns:
        Triangulation:
            A matplotlib Triangulation object.
    """

    if alpha is None:
        # TODO - add method here to automatically determine appropriate alpha value
        alpha = 1.1

    if len(x) != len(y):
        raise ValueError("x and y must be the same length")

    # Triangulate X, Y locations
    triang = Triangulation(x, y)

    xtri = x[triang.triangles] - np.roll(x[triang.triangles], 1, axis=1)
    ytri = y[triang.triangles] - np.roll(y[triang.triangles], 1, axis=1)
    maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)

    # Iterate triangulation masking until a possible mask is found
    count = 0
    fig, ax = plt.subplots(1, 1)
    synthetic_values = range(len(x))
    success = False
    while not success:
        count += 1
        try:
            tr = copy.deepcopy(triang)
            tr.set_mask(maxi > alpha)
            ax.tricontour(tr, synthetic_values)
            success = True
        except ValueError:
            alpha += increment
        else:
            break
        if count > max_iterations:
            plt.close(fig)
            raise ValueError(f"Could not create a valid triangulation mask within {max_iterations}")
    plt.close(fig)
    triang.set_mask(maxi > alpha)
    return triang


@bhom_analytics()
def format_polar_plot(ax: plt.Axes, yticklabels: bool = True) -> plt.Axes:
    """Format a polar plot, to save on having to write this every time!"""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # format plot area
    ax.spines["polar"].set_visible(False)
    ax.grid(True, which="both", ls="--", zorder=0, alpha=0.3)
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    plt.setp(ax.get_yticklabels(), fontsize="small")
    ax.set_xticks(np.radians((0, 90, 180, 270)), minor=False)
    ax.set_xticklabels(("N", "E", "S", "W"), minor=False, **{"fontsize": "medium"})
    ax.set_xticks(
        np.radians((22.5, 45, 67.5, 112.5, 135, 157.5, 202.5, 225, 247.5, 292.5, 315, 337.5)),
        minor=True,
    )
    ax.set_xticklabels(
        (
            "NNE",
            "NE",
            "ENE",
            "ESE",
            "SE",
            "SSE",
            "SSW",
            "SW",
            "WSW",
            "WNW",
            "NW",
            "NNW",
        ),
        minor=True,
        **{"fontsize": "x-small"},
    )
    if not yticklabels:
        ax.set_yticklabels([])


PARULA_COLORMAP = LinearSegmentedColormap.from_list(
    name="parula",
    colors=[
        [0.2422, 0.1504, 0.6603],
        [0.2444, 0.1534, 0.6728],
        [0.2464, 0.1569, 0.6847],
        [0.2484, 0.1607, 0.6961],
        [0.2503, 0.1648, 0.7071],
        [0.2522, 0.1689, 0.7179],
        [0.254, 0.1732, 0.7286],
        [0.2558, 0.1773, 0.7393],
        [0.2576, 0.1814, 0.7501],
        [0.2594, 0.1854, 0.761],
        [0.2611, 0.1893, 0.7719],
        [0.2628, 0.1932, 0.7828],
        [0.2645, 0.1972, 0.7937],
        [0.2661, 0.2011, 0.8043],
        [0.2676, 0.2052, 0.8148],
        [0.2691, 0.2094, 0.8249],
        [0.2704, 0.2138, 0.8346],
        [0.2717, 0.2184, 0.8439],
        [0.2729, 0.2231, 0.8528],
        [0.274, 0.228, 0.8612],
        [0.2749, 0.233, 0.8692],
        [0.2758, 0.2382, 0.8767],
        [0.2766, 0.2435, 0.884],
        [0.2774, 0.2489, 0.8908],
        [0.2781, 0.2543, 0.8973],
        [0.2788, 0.2598, 0.9035],
        [0.2794, 0.2653, 0.9094],
        [0.2798, 0.2708, 0.915],
        [0.2802, 0.2764, 0.9204],
        [0.2806, 0.2819, 0.9255],
        [0.2809, 0.2875, 0.9305],
        [0.2811, 0.293, 0.9352],
        [0.2813, 0.2985, 0.9397],
        [0.2814, 0.304, 0.9441],
        [0.2814, 0.3095, 0.9483],
        [0.2813, 0.315, 0.9524],
        [0.2811, 0.3204, 0.9563],
        [0.2809, 0.3259, 0.96],
        [0.2807, 0.3313, 0.9636],
        [0.2803, 0.3367, 0.967],
        [0.2798, 0.3421, 0.9702],
        [0.2791, 0.3475, 0.9733],
        [0.2784, 0.3529, 0.9763],
        [0.2776, 0.3583, 0.9791],
        [0.2766, 0.3638, 0.9817],
        [0.2754, 0.3693, 0.984],
        [0.2741, 0.3748, 0.9862],
        [0.2726, 0.3804, 0.9881],
        [0.271, 0.386, 0.9898],
        [0.2691, 0.3916, 0.9912],
        [0.267, 0.3973, 0.9924],
        [0.2647, 0.403, 0.9935],
        [0.2621, 0.4088, 0.9946],
        [0.2591, 0.4145, 0.9955],
        [0.2556, 0.4203, 0.9965],
        [0.2517, 0.4261, 0.9974],
        [0.2473, 0.4319, 0.9983],
        [0.2424, 0.4378, 0.9991],
        [0.2369, 0.4437, 0.9996],
        [0.2311, 0.4497, 0.9995],
        [0.225, 0.4559, 0.9985],
        [0.2189, 0.462, 0.9968],
        [0.2128, 0.4682, 0.9948],
        [0.2066, 0.4743, 0.9926],
        [0.2006, 0.4803, 0.9906],
        [0.195, 0.4861, 0.9887],
        [0.1903, 0.4919, 0.9867],
        [0.1869, 0.4975, 0.9844],
        [0.1847, 0.503, 0.9819],
        [0.1831, 0.5084, 0.9793],
        [0.1818, 0.5138, 0.9766],
        [0.1806, 0.5191, 0.9738],
        [0.1795, 0.5244, 0.9709],
        [0.1785, 0.5296, 0.9677],
        [0.1778, 0.5349, 0.9641],
        [0.1773, 0.5401, 0.9602],
        [0.1768, 0.5452, 0.956],
        [0.1764, 0.5504, 0.9516],
        [0.1755, 0.5554, 0.9473],
        [0.174, 0.5605, 0.9432],
        [0.1716, 0.5655, 0.9393],
        [0.1686, 0.5705, 0.9357],
        [0.1649, 0.5755, 0.9323],
        [0.161, 0.5805, 0.9289],
        [0.1573, 0.5854, 0.9254],
        [0.154, 0.5902, 0.9218],
        [0.1513, 0.595, 0.9182],
        [0.1492, 0.5997, 0.9147],
        [0.1475, 0.6043, 0.9113],
        [0.1461, 0.6089, 0.908],
        [0.1446, 0.6135, 0.905],
        [0.1429, 0.618, 0.9022],
        [0.1408, 0.6226, 0.8998],
        [0.1383, 0.6272, 0.8975],
        [0.1354, 0.6317, 0.8953],
        [0.1321, 0.6363, 0.8932],
        [0.1288, 0.6408, 0.891],
        [0.1253, 0.6453, 0.8887],
        [0.1219, 0.6497, 0.8862],
        [0.1185, 0.6541, 0.8834],
        [0.1152, 0.6584, 0.8804],
        [0.1119, 0.6627, 0.877],
        [0.1085, 0.6669, 0.8734],
        [0.1048, 0.671, 0.8695],
        [0.1009, 0.675, 0.8653],
        [0.0964, 0.6789, 0.8609],
        [0.0914, 0.6828, 0.8562],
        [0.0855, 0.6865, 0.8513],
        [0.0789, 0.6902, 0.8462],
        [0.0713, 0.6938, 0.8409],
        [0.0628, 0.6972, 0.8355],
        [0.0535, 0.7006, 0.8299],
        [0.0433, 0.7039, 0.8242],
        [0.0328, 0.7071, 0.8183],
        [0.0234, 0.7103, 0.8124],
        [0.0155, 0.7133, 0.8064],
        [0.0091, 0.7163, 0.8003],
        [0.0046, 0.7192, 0.7941],
        [0.0019, 0.722, 0.7878],
        [0.0009, 0.7248, 0.7815],
        [0.0018, 0.7275, 0.7752],
        [0.0046, 0.7301, 0.7688],
        [0.0094, 0.7327, 0.7623],
        [0.0162, 0.7352, 0.7558],
        [0.0253, 0.7376, 0.7492],
        [0.0369, 0.74, 0.7426],
        [0.0504, 0.7423, 0.7359],
        [0.0638, 0.7446, 0.7292],
        [0.077, 0.7468, 0.7224],
        [0.0899, 0.7489, 0.7156],
        [0.1023, 0.751, 0.7088],
        [0.1141, 0.7531, 0.7019],
        [0.1252, 0.7552, 0.695],
        [0.1354, 0.7572, 0.6881],
        [0.1448, 0.7593, 0.6812],
        [0.1532, 0.7614, 0.6741],
        [0.1609, 0.7635, 0.6671],
        [0.1678, 0.7656, 0.6599],
        [0.1741, 0.7678, 0.6527],
        [0.1799, 0.7699, 0.6454],
        [0.1853, 0.7721, 0.6379],
        [0.1905, 0.7743, 0.6303],
        [0.1954, 0.7765, 0.6225],
        [0.2003, 0.7787, 0.6146],
        [0.2061, 0.7808, 0.6065],
        [0.2118, 0.7828, 0.5983],
        [0.2178, 0.7849, 0.5899],
        [0.2244, 0.7869, 0.5813],
        [0.2318, 0.7887, 0.5725],
        [0.2401, 0.7905, 0.5636],
        [0.2491, 0.7922, 0.5546],
        [0.2589, 0.7937, 0.5454],
        [0.2695, 0.7951, 0.536],
        [0.2809, 0.7964, 0.5266],
        [0.2929, 0.7975, 0.517],
        [0.3052, 0.7985, 0.5074],
        [0.3176, 0.7994, 0.4975],
        [0.3301, 0.8002, 0.4876],
        [0.3424, 0.8009, 0.4774],
        [0.3548, 0.8016, 0.4669],
        [0.3671, 0.8021, 0.4563],
        [0.3795, 0.8026, 0.4454],
        [0.3921, 0.8029, 0.4344],
        [0.405, 0.8031, 0.4233],
        [0.4184, 0.803, 0.4122],
        [0.4322, 0.8028, 0.4013],
        [0.4463, 0.8024, 0.3904],
        [0.4608, 0.8018, 0.3797],
        [0.4753, 0.8011, 0.3691],
        [0.4899, 0.8002, 0.3586],
        [0.5044, 0.7993, 0.348],
        [0.5187, 0.7982, 0.3374],
        [0.5329, 0.797, 0.3267],
        [0.547, 0.7957, 0.3159],
        [0.5609, 0.7943, 0.305],
        [0.5748, 0.7929, 0.2941],
        [0.5886, 0.7913, 0.2833],
        [0.6024, 0.7896, 0.2726],
        [0.6161, 0.7878, 0.2622],
        [0.6297, 0.7859, 0.2521],
        [0.6433, 0.7839, 0.2423],
        [0.6567, 0.7818, 0.2329],
        [0.6701, 0.7796, 0.2239],
        [0.6833, 0.7773, 0.2155],
        [0.6963, 0.775, 0.2075],
        [0.7091, 0.7727, 0.1998],
        [0.7218, 0.7703, 0.1924],
        [0.7344, 0.7679, 0.1852],
        [0.7468, 0.7654, 0.1782],
        [0.759, 0.7629, 0.1717],
        [0.771, 0.7604, 0.1658],
        [0.7829, 0.7579, 0.1608],
        [0.7945, 0.7554, 0.157],
        [0.806, 0.7529, 0.1546],
        [0.8172, 0.7505, 0.1535],
        [0.8281, 0.7481, 0.1536],
        [0.8389, 0.7457, 0.1546],
        [0.8495, 0.7435, 0.1564],
        [0.86, 0.7413, 0.1587],
        [0.8703, 0.7392, 0.1615],
        [0.8804, 0.7372, 0.165],
        [0.8903, 0.7353, 0.1695],
        [0.9, 0.7336, 0.1749],
        [0.9093, 0.7321, 0.1815],
        [0.9184, 0.7308, 0.189],
        [0.9272, 0.7298, 0.1973],
        [0.9357, 0.729, 0.2061],
        [0.944, 0.7285, 0.2151],
        [0.9523, 0.7284, 0.2237],
        [0.9606, 0.7285, 0.2312],
        [0.9689, 0.7292, 0.2373],
        [0.977, 0.7304, 0.2418],
        [0.9842, 0.733, 0.2446],
        [0.99, 0.7365, 0.2429],
        [0.9946, 0.7407, 0.2394],
        [0.9966, 0.7458, 0.2351],
        [0.9971, 0.7513, 0.2309],
        [0.9972, 0.7569, 0.2267],
        [0.9971, 0.7626, 0.2224],
        [0.9969, 0.7683, 0.2181],
        [0.9966, 0.774, 0.2138],
        [0.9962, 0.7798, 0.2095],
        [0.9957, 0.7856, 0.2053],
        [0.9949, 0.7915, 0.2012],
        [0.9938, 0.7974, 0.1974],
        [0.9923, 0.8034, 0.1939],
        [0.9906, 0.8095, 0.1906],
        [0.9885, 0.8156, 0.1875],
        [0.9861, 0.8218, 0.1846],
        [0.9835, 0.828, 0.1817],
        [0.9807, 0.8342, 0.1787],
        [0.9778, 0.8404, 0.1757],
        [0.9748, 0.8467, 0.1726],
        [0.972, 0.8529, 0.1695],
        [0.9694, 0.8591, 0.1665],
        [0.9671, 0.8654, 0.1636],
        [0.9651, 0.8716, 0.1608],
        [0.9634, 0.8778, 0.1582],
        [0.9619, 0.884, 0.1557],
        [0.9608, 0.8902, 0.1532],
        [0.9601, 0.8963, 0.1507],
        [0.9596, 0.9023, 0.148],
        [0.9595, 0.9084, 0.145],
        [0.9597, 0.9143, 0.1418],
        [0.9601, 0.9203, 0.1382],
        [0.9608, 0.9262, 0.1344],
        [0.9618, 0.932, 0.1304],
        [0.9629, 0.9379, 0.1261],
        [0.9642, 0.9437, 0.1216],
        [0.9657, 0.9494, 0.1168],
        [0.9674, 0.9552, 0.1116],
        [0.9692, 0.9609, 0.1061],
        [0.9711, 0.9667, 0.1001],
        [0.973, 0.9724, 0.0938],
        [0.9749, 0.9782, 0.0872],
        [0.9769, 0.9839, 0.0805],
    ],
)
