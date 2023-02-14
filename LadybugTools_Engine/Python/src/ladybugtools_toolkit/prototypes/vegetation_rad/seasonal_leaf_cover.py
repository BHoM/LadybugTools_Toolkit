"""
This module contains a prototype for simulating seasonal leaf cover. It
generates each month individually based on a face/modifier adjustment and
simulatin, and then joins results together so that the results are in a
single file.

Things to note:
- Inputs are a single Honeybee model, plus dictionary contianing teh modifiers to adjust - can be used to adjust differnt modifiers differently.
- This prototype is not intended to be used as a final product. It is a work in progress.
- The prototype is currently set up to work with a single sensor grid.
- The prototype is currently set up to produce a monthly pandas dataframe of sky-view reuslts.
- The prototype is currently set up to produce an annual pandas dataframe of direct and total irradiance results.
"""

import contextlib
import getpass
import io
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.model import Face, Model
from honeybee_radiance.modifier.material import Glass
from honeybee_radiance.sensorgrid import SensorGrid
from ladybug.wea import AnalysisPeriod, Wea
from ladybugtools_toolkit.external_comfort import QUEENBEE_PATH
from ladybugtools_toolkit.honeybee_extension.results import (
    load_ill_file,
    load_res,
    make_annual,
)
from lbt_recipes.recipe import Recipe, RecipeSettings
from tqdm import tqdm


def get_model_grid(model: Model) -> SensorGrid:
    """Get the sensor grid from a honeybee model.

    Args:
        model (Model):
            A honeybee model.

    Raises:
        ValueError: If the model has more than one sensor grid.

    Returns:
        SensorGrid:
            The sensor grid from the model.
    """

    if len(model.properties.radiance.sensor_grids) > 1:
        raise ValueError("Model has more than one sensor grid.")

    return model.properties.radiance.sensor_grids[0]


def get_model_faces_by_modifier(model: Model, modifier_name: str) -> List[Face]:
    """Return a list of faces within a model based on the requested modifier name.

    Args:
        model (Model):
            A honeybee model.
        modifier_name (str):
            The name of the modifier to search for.

    Returns:
        List[Face]:
            A list of faces with the requested modifier.
    """

    faces = [
        face
        for face in model.faces
        if face.properties.radiance.modifier.identifier == modifier_name
    ]

    if len(faces) == 0:
        raise ValueError(f"No faces with modifier {modifier_name} found.")

    return faces


def assign_glass_modifier_to_faces(faces: List[Face], modifier: Glass) -> None:
    """Assign a glass modifier to a list of faces.

    Args:
        faces (List[Face]):
            A list of honeybee faces.
        modifier (Glass):
            A honeybee glass modifier.

    Raises:
        ValueError: If the modifier is not a glass modifier.

    Returns:
        None:
            The faces are modified in place.
    """

    for face in faces:
        if not isinstance(face.properties.radiance.modifier, Glass):
            raise ValueError(f"{face} modifier is not glass, and cannot be adjusted.")
        face.properties.radiance.modifier = modifier


def modify_model_veg(model: Model, modifier_config: Dict[str, float]) -> Model:
    """Adjust the porosity/transmissivity of an object.

    Args:
        model (Model):
            A honeybee model.
        modifier_config (Dict[str, float]):
            A dictionary containing the modifier_name: modifier_transmissivity to apply.

    Returns:
        Model:
            A honeybee model with adjusted modifiers.
    """

    new_model = deepcopy(model)

    for modifier_name, veg_object_transmissivity in modifier_config.items():

        # create new modifier to apply to faces
        _modifier = Glass.from_single_transmissivity(
            identifier=(f"{modifier_name}_{veg_object_transmissivity}"),
            rgb_transmissivity=veg_object_transmissivity,
            refraction_index=1.45,
        )

        # get faces to modify
        faces = get_model_faces_by_modifier(
            model=new_model, modifier_name=modifier_name
        )

        # modify faces
        assign_glass_modifier_to_faces(faces=faces, modifier=_modifier)

    return new_model


def simulate_sky_view(model: Model) -> Path:
    """Simulate sky view for a model.

    Args:
        model (Model):
            A honeybee model.

    Returns:
        Path:
            The path to the sky view results file.
    """

    if len(model.properties.radiance.sensor_grids) > 1:
        raise ValueError("Model has more than one sensor grid.")

    recipe = Recipe("sky-view")
    recipe.input_value_by_name("model", model)
    recipe_settings = RecipeSettings()
    result = recipe.run(
        settings=recipe_settings,
        radiance_check=True,
        queenbee_path=QUEENBEE_PATH,
        silent=True,
    )

    grid_name = model.properties.radiance.sensor_grids[0].identifier
    return Path(result) / "sky_view" / "results" / f"{grid_name}.res"


def simulate_irradiance(
    model: Model, epw_file: Path, analysis_period: AnalysisPeriod
) -> List[Path]:

    if len(model.properties.radiance.sensor_grids) > 1:
        raise ValueError("Model has more than one sensor grid.")

    wea = Wea.from_epw_file(epw_file).filter_by_analysis_period(analysis_period)

    recipe = Recipe("annual-irradiance")
    recipe.input_value_by_name("model", model)
    recipe.input_value_by_name("wea", wea)
    recipe.input_value_by_name("output-type", "solar")
    recipe_settings = RecipeSettings()
    result = recipe.run(
        settings=recipe_settings,
        radiance_check=True,
        queenbee_path=QUEENBEE_PATH,
        silent=True,
    )

    # get result file/s and move & rename
    grid_name = model.properties.radiance.sensor_grids[0].identifier
    total_result_file = (
        Path(result) / "annual_irradiance" / "results" / "total" / f"{grid_name}.ill"
    )
    direct_result_file = (
        Path(result) / "annual_irradiance" / "results" / "direct" / f"{grid_name}.ill"
    )
    sun_up_hours_file = (
        Path(result) / "annual_irradiance" / "results" / "direct" / "sun-up-hours.txt"
    )

    return (
        total_result_file,
        direct_result_file,
        sun_up_hours_file,
    )


def simulate_monthly(
    model: Model, epw_file: Path, modifier_monthly_config: Dict[str, float]
) -> Path:

    # check config has 12 keys (1 for each month)
    if len(list(modifier_monthly_config.keys())) != 12:
        raise ValueError("Modifier config must have 12 keys (1 for each month).")
    if list(modifier_monthly_config.keys()) != list(range(1, 13, 1)):
        raise ValueError("Modifier config keys must be 1-12.")

    # run simulation for each key
    for month, modifier_config in tqdm(modifier_monthly_config.items()):
        _model = modify_model_veg(model, modifier_config)
        _analysis_period = AnalysisPeriod(st_month=month, end_month=month)
        _sky_view_result = simulate_sky_view(_model)
        _total_rad_result, _direct_rad_result, _sun_up_result = simulate_irradiance(
            _model, epw_file, _analysis_period
        )

        # copy directory back to simulation folder
        monthly_results = _total_rad_result.parents[3] / "monthly_results"
        monthly_results.mkdir(parents=True, exist_ok=True)

        # move results to tmp dir
        for res_file in [
            _sky_view_result,
            _total_rad_result,
            _direct_rad_result,
            _sun_up_result,
        ]:
            if res_file.suffix in [".res", ".txt"]:
                target_file = (
                    monthly_results / f"{res_file.stem}_{month:02d}{res_file.suffix}"
                )
            else:
                target_file = (
                    monthly_results
                    / f"{res_file.stem}_{res_file.parent.name}_{month:02d}{res_file.suffix}"
                )

            shutil.move(
                res_file.as_posix(),
                target_file.as_posix(),
            )

    return monthly_results


def combine_results(monthly_results: Path) -> List[Path]:
    """Find all res files in a directoyr, and load them after sorting to create a pandas dataframe."""

    # res file
    res_files = list(monthly_results.glob("*.res"))
    sky_view_df = load_res(res_files=res_files)
    sky_view_df.columns = range(1, 13, 1)

    # rad files
    sun_up_hours_files = list(monthly_results.glob("*sun-up-hours*.txt"))
    total_rad_files = list(monthly_results.glob("*total*.ill"))
    direct_rad_files = list(monthly_results.glob("*direct*.ill"))

    total_rad_df = make_annual(
        pd.concat(
            [
                load_ill_file(i, j).droplevel(0, axis=1)
                for i, j in list(zip(*[total_rad_files, sun_up_hours_files]))
            ],
            axis=0,
        )
    ).fillna(0)
    direct_rad_df = make_annual(
        pd.concat(
            [
                load_ill_file(i, j).droplevel(0, axis=1)
                for i, j in list(zip(*[direct_rad_files, sun_up_hours_files]))
            ],
            axis=0,
        )
    ).fillna(0)

    # change column names to strings to appease the parquet gods
    sky_view_df.columns = sky_view_df.columns.astype(str)
    total_rad_df.columns = total_rad_df.columns.astype(str)
    direct_rad_df.columns = direct_rad_df.columns.astype(str)

    sky_view_df.to_parquet(monthly_results.parent / "monthlyveg_sky_view.parquet")
    total_rad_df.to_parquet(monthly_results.parent / "monthlyveg_total_rad.parquet")
    direct_rad_df.to_parquet(monthly_results.parent / "monthlyveg_direct_rad.parquet")

    return sky_view_df, total_rad_df, direct_rad_df


def main():

    # load hbjson
    model = Model.from_hbjson(
        r"C:\Users\tgerrish\Documents\GitHub\BHoM\LadybugTools_Toolkit\LadybugTools_Engine\Python\src\ladybugtools_toolkit\prototypes\vegetation_rad\seasonal_leaf_cover.hbjson"
    )

    # set properties for a specific month
    epw_file = r"C:\Users\tgerrish\BuroHappold\Sustainability and Physics - epws\USA_OH_Toledo.725360_TMY2.epw"

    # create modifier config
    modifier_monthly_config = {
        1: {"veg_deciduous": 0.8},
        2: {"veg_deciduous": 0.8},
        3: {"veg_deciduous": 0.7},
        4: {"veg_deciduous": 0.5},
        5: {"veg_deciduous": 0.3},
        6: {"veg_deciduous": 0.2},
        7: {"veg_deciduous": 0.2},
        8: {"veg_deciduous": 0.25},
        9: {"veg_deciduous": 0.35},
        10: {"veg_deciduous": 0.4},
        11: {"veg_deciduous": 0.6},
        12: {"veg_deciduous": 0.7},
    }

    # run simulations!
    composites_folder = simulate_monthly(model, epw_file, modifier_monthly_config)

    # combine results
    combine_results(composites_folder)


if __name__ == "__main__":
    main()
