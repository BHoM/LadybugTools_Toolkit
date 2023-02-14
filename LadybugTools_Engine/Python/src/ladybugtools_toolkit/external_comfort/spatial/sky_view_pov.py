from typing import Union

from honeybee.model import Model
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW, AnalysisPeriod
from ladybug_geometry.geometry3d import Point3D, Vector3D
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, Colormap
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ...helpers import figure_to_image
from ...plot.fisheye_sky import fisheye_sky
from ...plot.skymatrix import skymatrix
from ...plot.sunpath import sunpath


def sky_view_pov(
    model: Model,
    sensor: Point3D,
    epw: EPW,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    cmap: Union[Colormap, str] = "viridis",
    norm: BoundaryNorm = None,
    data_collection: HourlyContinuousCollection = None,
    density: int = 1,
    show_sunpath: bool = True,
    show_skymatrix: bool = True,
    title: str = None,
    translate_sensor: Vector3D = None,
) -> Image:
    """Create a sky view with overlaid sun location information

    Args:
        model (Model):
            A honeybee model object.
        sensor (Point3D):
            A location from where the sky view will be rendered.
        epw (EPW):
            A ladybug EPW object.
        analysis_period (AnalysisPeriod, optional):
            THe analysis period over which the sky-view should be rendered.
            Defaults to AnalysisPeriod().
        cmap (Union[Colormap, str], optional):
            A colormap to apply to the sun-path when a data-collection is supplied.
            Defaults to "viridis".
        norm (BoundaryNorm, optional):
            If cmap applied is UTCI, provide the associated boundary-norm. Defaults to None.
        data_collection (HourlyContinuousCollection, optional):
            A collection of values to color the sun-path by. Defaults to None.
        density (int, optional):
            The sky-dome density. Defaults to 1 which is a Tregenza dome.
        show_sunpath (bool, optional):
            Hide the sunpath if set to False. Defaults to True.
        show_skymatrix (bool, optional):
            Hide the sky-dome if set to False. Defaults to True.
        title (str, optional):
            Add a title to the plot.. Defaults to None.
        translate_sensor (Vector3D):
            A translation to apply to the sensor.
    Returns:
        Image:
            A PIL image object ready to save.
    """

    # render the sky view Image
    sky_view_img = fisheye_sky(model, sensor, translate_sensor)

    # render the sunpath
    if show_sunpath:
        sunpath_img = ImageOps.mirror(
            figure_to_image(
                sunpath(
                    epw=epw,
                    analysis_period=analysis_period,
                    data_collection=data_collection,
                    cmap=cmap,
                    norm=norm,
                    show_title=False,
                    show_grid=False,
                    show_legend=False,
                )
            ).resize(sky_view_img.size)
        )
        plt.close("all")

    # render the sky matrix
    if show_skymatrix:
        skymatrix_img = ImageOps.mirror(
            figure_to_image(
                skymatrix(
                    epw=epw,
                    analysis_period=analysis_period,
                    density=density,
                    cmap="Spectral_r",
                    show_title=False,
                    show_colorbar=False,
                )
            ).resize(sky_view_img.size)
        )
        plt.close("all")

    if show_skymatrix and show_sunpath:
        combined_img = Image.alpha_composite(
            Image.alpha_composite(skymatrix_img, sunpath_img), sky_view_img
        )
    elif show_sunpath:
        combined_img = Image.alpha_composite(sunpath_img, sky_view_img)
    elif show_skymatrix:
        combined_img = Image.alpha_composite(skymatrix_img, sky_view_img)
    else:
        combined_img = sky_view_img

    if title is not None:
        draw = ImageDraw.Draw(combined_img)
        try:
            font = ImageFont.truetype("C:/Windows/fonts/segoeuil.ttf", 14)
            draw.text(
                (5, 5),
                title,
                (255, 255, 255),
                font=font,
            )
        except Exception:
            draw.text(
                (5, 5),
                title,
                (255, 255, 255),
            )

    return combined_img
