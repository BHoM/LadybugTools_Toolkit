from typing import List, Tuple

import numpy as np
from matplotlib import ticker as mticker
from matplotlib.image import AxesImage


def annotate_imshow(
    im: AxesImage,
    data: List[float] = None,
    valfmt: str = "{x:.2f}",
    textcolors: Tuple[str] = ("black", "white"),
    threshold: float = None,
    exclude_vals: List[float] = None,
    **textkw,
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

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(ha="center", va="center")
    kw.update(textkw)

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
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

    return texts
