import base64
import colorsys
import copy
import io
from pathlib import Path
from typing import Any, List, Tuple, Union

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
    to_rgb,
    to_rgba_array,
)
from matplotlib.tri import Triangulation
from PIL import Image


def animation(
    image_files: List[Union[str, Path]],
    output_gif: Union[str, Path],
    ms_per_image: int = 333,
    transparency_idx: int = 0,
) -> Path:
    """Create an animated gif from a set of images.

    Args:
        image_files (List[Union[str, Path]]):
            A list of image files.
        output_gif (Union[str, Path]):
            The output gif file to be created.
        ms_per_image (int, optional):
            Number of milliseconds per image. Default is 333, for 3 images per second.
        transparency_idx (int, optional):
            The index of the color to be used as the transparent color. Default is 0.

    Returns:
        Path:
            The animated gif.

    """

    image_files = [Path(i) for i in image_files]

    images = [Image.open(i) for i in image_files]

    # create white background
    background = Image.new("RGBA", images[0].size, (255, 255, 255))

    images = [Image.alpha_composite(background, i) for i in images]

    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
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

    color : matplotlib color or sequence of matplotlib colors - Hex code,
    rgb-tuple, or html color name.
    Returns
    -------
    luminance : float(s) between 0 and 1
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
        color (Any): matplotlib color or sequence of matplotlib colors - Hex code,
        rgb-tuple, or html color name.
    Returns:
        str: String code of the contrasting color.
    """
    return ".15" if relative_luminance(color) > 0.408 else "w"


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


def lighten_color(color: Union[str, Tuple], amount: float = 0.5) -> Tuple[float]:
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.

    Args:
        color (str):
            A color-like string.
        amount (float):
            The amount of lightening to apply.

    Returns:
        Tuple[float]:
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


def add_bar_labels(ax: plt.Axes, orientation: str, threshold: float) -> None:
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes):
            The matplotlib object containing the axes of the plot to annotate.
        orientation (str):
            The orientation of the plot. Either "vertical" or "horizontal".
        threshold (float):
            The threshold value to use to determine whether to add a label.

    """

    for rect in ax.patches:
        x = rect.get_x() + (rect.get_width() / 2)
        y = rect.get_y() + (rect.get_height() / 2)
        if orientation == "vertical":
            value = rect.get_height()
        elif orientation == "horizontal":
            value = rect.get_width()
        else:
            raise ValueError("orientation must be either 'vertical' or 'horizontal'")
        if value > threshold:
            ax.annotate(
                f"{value:.0%}",
                (x, y),
                ha="center",
                va="center",
                c=contrasting_color(rect.get_facecolor()),
            )


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


def average_color(colors: Any, keep_alpha: bool = False) -> str:
    """Return the average color from a list of colors.

    Args:
        colors (Any):
            A list of colors.
        keep_alpha (bool, optional):
            If True, the alpha value of the color is kept. Defaults to False.

    Returns:
        color: str
            The average color in hex format.
    """

    if not isinstance(colors, (list, tuple)):
        raise ValueError("colors must be a list")

    for i in colors:
        if not is_color_like(i):
            raise ValueError(
                f"colors must be a list of valid colors - '{i}' is not valid."
            )

    if len(colors) == 1:
        return colors[0]

    return rgb2hex(to_rgba_array(colors).mean(axis=0), keep_alpha=keep_alpha)


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


def figure_to_base64(figure: plt.Figure, html: bool = False) -> str:
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
    figure.savefig(buffer)
    buffer.seek(0)
    base64_string = base64.b64encode(buffer.read()).decode("utf-8")

    if html:
        content_type = "data:image/png"
        content_encoding = "utf-8"
        return f"{content_type};charset={content_encoding};base64,{base64_string}"

    return base64_string


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


def tile_images(
    imgs: Union[List[Path], List[Image.Image]], rows: int, cols: int
) -> Image.Image:
    """Tile a set of images into a grid.

    Args:
        imgs (Union[List[Path], List[Image.Image]]):
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
        raise ValueError(
            f"The number of images given ({len(imgs)}) does not equal ({rows}*{cols})"
        )

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


def create_triangulation(
    x: List[float],
    y: List[float],
    alpha: float = None,
    max_iterations: int = 250,
    increment: float = 0.01,
) -> Triangulation:
    """Create a matplotlib Triangulation from a list of x and y coordinates, including a mask to
        remove elements with edges larger than alpha.

    Args:
        x (List[float]):
            A list of x coordinates.
        y (List[float]):
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
            raise ValueError(
                f"Could not create a valid triangulation mask within {max_iterations}"
            )
    plt.close(fig)
    triang.set_mask(maxi > alpha)
    return triang
