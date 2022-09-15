from pathlib import Path
from typing import List, Union

from PIL import Image


from python_toolkit.bhom.analytics import analytics


@analytics
def animation(
    image_files: List[Union[str, Path]],
    output_gif: Union[str, Path],
    ms_per_image: int = 333,
) -> Path:
    """Create an animated gif from a set of images.

    Args:
        image_files (List[Union[str, Path]]):
            A list of image files.
        ms_per_image (int, optional):
            NUmber of milliseconds per image. Default is 333, for 3 images per second.

    Returns:
        Path:
            The animated gif.

    """

    image_files = [Path(i) for i in image_files]

    images = [Image.open(i) for i in image_files]

    background = Image.new("RGBA", images[0].size, (255, 255, 255))

    images = [Image.alpha_composite(background, i) for i in images]

    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=ms_per_image,
        loop=0,
    )
