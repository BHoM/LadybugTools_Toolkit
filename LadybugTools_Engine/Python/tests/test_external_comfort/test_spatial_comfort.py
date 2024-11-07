import shutil
from pathlib import Path

import pandas as pd
import pytest
from ladybug.wea import AnalysisPeriod, Wea
from ladybugtools_toolkit.external_comfort.spatial.spatial_comfort import \
    SpatialComfort
from ladybugtools_toolkit.honeybee_extension.simulation.radiance import \
    HoneybeeRadiance

from .. import (CFD_DIRECTORY, EPW_FILE, SPATIAL_COMFORT_DIRECTORY,
                SPATIAL_COMFORT_MODEL_OBJ)
from .test_simulate import TEST_SIMULATION_RESULT

SPATIAL_COMFORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
CFD_LOCAL_DIRECTORY = SPATIAL_COMFORT_DIRECTORY / "cfd"
CFD_LOCAL_DIRECTORY.mkdir(parents=True, exist_ok=True)


# copy cfd values into local directory for testing
for file in list(CFD_DIRECTORY.glob("**/*")):
    shutil.copy(file, CFD_LOCAL_DIRECTORY)


def test_run_spatial_annual_irradiance():
    """Run spatial annual-irradiance simulation and return working directory."""

    hr = HoneybeeRadiance(SPATIAL_COMFORT_MODEL_OBJ)
    wea = Wea.from_epw_file(EPW_FILE)
    with pytest.warns(UserWarning):
        hr.simulate_annual_irradiance(wea=wea)
    assert isinstance(hr.annual_irradiance(), pd.DataFrame)


def test_run_spatial_sky_view() -> Path:
    """Run spatial sky-view simulation and return working directory."""

    hr = HoneybeeRadiance(SPATIAL_COMFORT_MODEL_OBJ)
    hr.simulate_sky_view()
    assert isinstance(hr.sky_view(), pd.DataFrame)


def test_spatial_comfort():
    """_"""

    assert isinstance(
        SpatialComfort(SPATIAL_COMFORT_DIRECTORY, TEST_SIMULATION_RESULT),
        SpatialComfort,
    )


def test_spatial_comfort_processing():
    """_"""

    spatial_comfort = SpatialComfort(
        SPATIAL_COMFORT_DIRECTORY,
        TEST_SIMULATION_RESULT)

    # remove existing files
    for fp in spatial_comfort.spatial_simulation_directory.glob("*.parquet"):
        fp.unlink()

    # with pytest.warns((pd.errors.PerformanceWarning)):
    assert isinstance(
        spatial_comfort.universal_thermal_climate_index_calculated,
        pd.DataFrame)


def test_spatial_comfort_summary():
    """_"""

    spatial_comfort = SpatialComfort(
        SPATIAL_COMFORT_DIRECTORY,
        TEST_SIMULATION_RESULT)

    # remove analsyis periods fo testing
    spatial_comfort.analysis_periods = [
        AnalysisPeriod(st_month=3, end_month=3, st_day=21, end_day=21)
    ]
    spatial_comfort.run_all()
