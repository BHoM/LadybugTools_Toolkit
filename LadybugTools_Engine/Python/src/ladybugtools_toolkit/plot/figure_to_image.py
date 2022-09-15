import numpy as np
from matplotlib.figure import Figure
from PIL import Image


from python_toolkit.bhom.analytics import analytics


@analytics
def figure_to_image(fig: Figure) -> Image:
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
