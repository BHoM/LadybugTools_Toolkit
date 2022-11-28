import copy

import numpy as np
from cv2 import IMREAD_ANYDEPTH, imread
from honeybee.model import Model
from honeybee_radiance.lightsource.sky import CertainIrradiance
from honeybee_radiance.modifier.material import Plastic
from honeybee_radiance.view import View
from ladybug_geometry.geometry3d import Point3D, Vector3D
from lbt_recipes.recipe import Recipe
from lbt_recipes.settings import RecipeSettings
from PIL import Image, ImageEnhance


def fisheye_sky(
    model: Model,
    sensor: Point3D,
    translate_sensor: Vector3D = None,
) -> Image:
    """Create a sky-facing fisheye image.

    Args:
        model (Model):
            The model in which to simulate the view.
        sensor (Point3D):
            The location from where to render the view.
        translate_sensor (Vector3D):
            A translation to apply to the sensor.

    Returns:
        Image:
            A PIL Image.
    """

    up_vector = Vector3D(0, 1, 0)
    view_direction = Vector3D(0, 0, 1)

    if translate_sensor is not None:
        sensor = sensor.move(translate_sensor)

    # copy model and rename
    model = copy.copy(model)
    model.identifier = "fisheye_sky"

    # modify materials to make plastic
    modifier = Plastic.from_single_reflectance("fisheye_sky", 0.1)
    for face in model.faces:
        face.properties.radiance.modifier = modifier

    view = View("fisheye_sky", sensor, view_direction, up_vector, "a", 360, 360)

    model.properties.radiance.views = [view]

    sky = CertainIrradiance(100000)

    recipe = Recipe("point-in-time-view")
    recipe.input_value_by_name("model", model)
    recipe.input_value_by_name("sky", sky)
    recipe.input_value_by_name("view-filter", "fisheye_sky")
    recipe.input_value_by_name("resolution", 1600)
    recipe_settings = RecipeSettings()
    project_folder = recipe.run(
        settings=recipe_settings,
        radiance_check=True,
        queenbee_path=r"C:\Program Files\ladybug_tools\python\Scripts\queenbee.exe",
    )
    result = recipe.output_value_by_name("results", project_folder)

    # load HDR image
    pth = result[0]
    img = imread(pth, flags=IMREAD_ANYDEPTH)

    # convert RGB values per pixel into single values per pixel
    normalised = np.interp(img, [img.min(), img.max()], [0, 255]).astype("uint8")
    img = Image.fromarray(normalised)

    # image brightness enhancer
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(10)

    # set bright values (sky) to transparent in pov img
    img = img.convert("RGBA")
    datas = img.getdata()
    new_data = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    img.putdata(new_data)
    # smooth image
    im1 = img.resize(
        (int(img.size[0] * 0.9), int(img.size[0] * 0.9)), Image.Resampling.BILINEAR
    )
    im1_as_array = np.array(im1)
    alpha = im1_as_array[:, :, 3]
    semi_transparent_indices = alpha < 255
    alpha[semi_transparent_indices] = 0
    im1_as_array[:, :, 3] = alpha
    img = Image.fromarray(im1_as_array, "RGBA")

    # img = img.filter(ImageFilter.SMOOTH_MORE)

    # crop out most of floor from fisheye view. typically we see 210deg, so if the image is 360, crop 40$ off of the image (20% from each edge)
    image_size = img.size[0]
    img = img.crop(
        (
            image_size * 0.2,
            image_size * 0.2,
            image_size - (image_size * 0.2),
            image_size - (image_size * 0.2),
        )
    )

    return img
