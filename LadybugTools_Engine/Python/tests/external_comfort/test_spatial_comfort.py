from __future__ import annotations

import getpass
import shutil
from pathlib import Path

import pandas as pd
import pytest
from honeybee.config import folders as hb_folders
from honeybee.model import Model
from ladybug.wea import AnalysisPeriod, Wea
from ladybugtools_toolkit.external_comfort.external_comfort import SimulationResult
from ladybugtools_toolkit.external_comfort.material import Materials
from ladybugtools_toolkit.external_comfort.spatial.spatial_comfort import SpatialComfort
from lbt_recipes.recipe import Recipe, RecipeSettings

from .. import (
    CFD_DIRECTORY,
    EPW_FILE,
    EXTERNAL_COMFORT_IDENTIFIER,
    MODEL_FILE,
    SPATIAL_COMFORT_DIRECTORY,
)

SPATIAL_COMFORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
CFD_LOCAL_DIRECTORY = SPATIAL_COMFORT_DIRECTORY / "cfd"
CFD_LOCAL_DIRECTORY.mkdir(parents=True, exist_ok=True)

GROUND_MATERIAL = Materials.LBT_AsphaltPavement.value
SHADE_MATERIAL = Materials.FABRIC.value

# copy cfd values into local directory for testing
for file in list(CFD_DIRECTORY.glob("**/*")):
    shutil.copy(file, CFD_LOCAL_DIRECTORY)


@pytest.mark.order(1)
def test_run_spatial_annual_irradiance() -> Path:
    """Run spatial annual-irradiance simulation and return working directory."""

    # set simulation directory
    hb_folders.default_simulation_folder = f"C:/Users/{getpass.getuser()}/simulation"

    ill_file = (
        SPATIAL_COMFORT_DIRECTORY
        / "annual_irradiance"
        / "results"
        / "total"
        / "pytest_SC.ill"
    )

    # load test model containing grids
    model: Model = Model.from_hbjson(MODEL_FILE.as_posix())

    # create wea object
    wea = Wea.from_epw_file(EPW_FILE)

    # run annual irradiance recipe
    recipe = Recipe("annual-irradiance")
    recipe.input_value_by_name("model", model)
    recipe.input_value_by_name("wea", wea)
    recipe.input_value_by_name("north", 0)
    recipe.input_value_by_name("timestep", 1)
    recipe.input_value_by_name("output-type", "solar")
    recipe.input_value_by_name("grid-filter", "pytest")
    recipe_settings = RecipeSettings()
    _ = recipe.run(
        settings=recipe_settings,
        radiance_check=True,
        queenbee_path=r"C:\Program Files\ladybug_tools\python\Scripts\queenbee.exe",
    )

    assert ill_file.exists()


@pytest.mark.order(2)
def test_run_spatial_sky_view() -> Path:
    """Run spatial sky-view simulation and return working directory."""

    # set simulation directory
    hb_folders.default_simulation_folder = f"C:/Users/{getpass.getuser()}/simulation"

    res_file_old = (
        SPATIAL_COMFORT_DIRECTORY
        / "sky_view"
        / "results"
        / "sky_view"
        / "pytest_SC.res"
    )
    res_file_new = SPATIAL_COMFORT_DIRECTORY / "sky_view" / "results" / "pytest_SC.res"

    # load test model containing grids
    model: Model = Model.from_hbjson(MODEL_FILE.as_posix())

    # run annual irradiance recipe
    recipe = Recipe("sky-view")
    recipe.input_value_by_name("model", model)
    recipe.input_value_by_name("grid-filter", "pytest")
    recipe_settings = RecipeSettings()
    _ = recipe.run(
        settings=recipe_settings,
        radiance_check=True,
        queenbee_path=r"C:\Program Files\ladybug_tools\python\Scripts\queenbee.exe",
    )

    assert res_file_old.exists() or res_file_new.exists()


@pytest.mark.order(3)
def test_spatial_comfort():
    """_"""

    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()

    assert isinstance(
        SpatialComfort(SPATIAL_COMFORT_DIRECTORY, sim_res), SpatialComfort
    )


@pytest.mark.order(4)
def test_spatial_comfort_processing():
    """_"""

    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()

    spatial_comfort = SpatialComfort(SPATIAL_COMFORT_DIRECTORY, sim_res)

    # remove existing files
    for fp in spatial_comfort.spatial_simulation_directory.glob("*.parquet"):
        fp.unlink()

    # with pytest.warns((pd.errors.PerformanceWarning)):
    assert isinstance(
        spatial_comfort.universal_thermal_climate_index_calculated, pd.DataFrame
    )


@pytest.mark.order(5)
def test_spatial_comfort_summary():
    """_"""

    sim_res = SimulationResult(
        EPW_FILE,
        GROUND_MATERIAL,
        SHADE_MATERIAL,
        EXTERNAL_COMFORT_IDENTIFIER,
    ).run()

    spatial_comfort = SpatialComfort(SPATIAL_COMFORT_DIRECTORY, sim_res)

    # remove analsyis periods fo testing
    spatial_comfort.analysis_periods = [
        AnalysisPeriod(st_month=3, end_month=3, st_day=21, end_day=21)
    ]
    spatial_comfort.run_all()
