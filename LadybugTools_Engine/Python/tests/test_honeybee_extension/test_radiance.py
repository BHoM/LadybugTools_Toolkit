import pytest
from pathlib import Path
from ladybugtools_toolkit.honeybee_extension.simulation.radiance import (
    radiance_parameters,
    HoneybeeRadiance,
)
from ladybug.futil import nukedir
from ladybugtools_toolkit.ladybug_extension.analysisperiod import (
    describe_analysis_period,
    AnalysisPeriod,
)

from .. import TEST_DAYLIGHT_MODEL, TEST_WEA

TEST_HB_RAD = HoneybeeRadiance(TEST_DAYLIGHT_MODEL)


def test_object_init():
    """_"""

    assert TEST_HB_RAD.model_name == "LBTBHoM_pytest_daylight"
    assert TEST_HB_RAD.output_directory.parent.name == "simulation"


def test_simulate_sky_view():
    """_"""

    nukedir(TEST_HB_RAD.output_directory / "sky_view", rmdir=True)

    with pytest.raises(ValueError):
        TEST_HB_RAD.sky_view()

    assert isinstance(TEST_HB_RAD.simulate_sky_view(), Path)

    assert TEST_HB_RAD.sky_view().shape == (25, 8)
    assert TEST_HB_RAD.sky_view().values.sum() == pytest.approx(658, rel=10)


def test_simulate_daylight_factor():
    """_"""

    nukedir(TEST_HB_RAD.output_directory / "daylight_factor", rmdir=True)

    with pytest.raises(ValueError):
        TEST_HB_RAD.daylight_factor()

    assert isinstance(TEST_HB_RAD.simulate_daylight_factor(), Path)

    assert TEST_HB_RAD.daylight_factor().shape == (25, 8)
    assert TEST_HB_RAD.daylight_factor().values.sum() == pytest.approx(658, rel=10)


def test_simulate_annual_daylight():
    """_"""

    nukedir(TEST_HB_RAD.output_directory / "annual_daylight", rmdir=True)

    with pytest.raises(ValueError):
        TEST_HB_RAD.annual_daylight()

    with pytest.warns(UserWarning):
        assert isinstance(TEST_HB_RAD.simulate_annual_daylight(wea=TEST_WEA), Path)

    assert TEST_HB_RAD.annual_daylight().shape == (4398, 200)
    assert TEST_HB_RAD.annual_daylight().values.sum() == pytest.approx(
        902620740.0, rel=10000
    )


def test_simulate_annual_irradiance():
    """_"""

    nukedir(TEST_HB_RAD.output_directory / "annual_irradiance", rmdir=True)

    with pytest.raises(ValueError):
        TEST_HB_RAD.annual_irradiance()

    with pytest.warns(UserWarning):
        assert isinstance(TEST_HB_RAD.simulate_annual_irradiance(wea=TEST_WEA), Path)

    assert TEST_HB_RAD.annual_irradiance().shape == (4398, 200)
    assert TEST_HB_RAD.annual_irradiance().values.sum() == pytest.approx(
        8228457, rel=1000
    )


def test_direct_sun_hours():
    """_"""
    ap = AnalysisPeriod(st_month=3)

    nukedir(
        TEST_HB_RAD.output_directory
        / f"direct_sun_hours_{describe_analysis_period(ap, save_path=True, include_timestep=True)}",
        rmdir=True,
    )

    with pytest.raises(ValueError):
        TEST_HB_RAD.direct_sun_hours()

    assert isinstance(
        TEST_HB_RAD.simulate_direct_sun_hours(wea=TEST_WEA, analysis_period=ap), Path
    )

    assert TEST_HB_RAD.direct_sun_hours().shape == (25, 8)
    assert TEST_HB_RAD.direct_sun_hours().values.sum() == 75444.0
    assert (
        TEST_HB_RAD.direct_sun_hours().columns.get_level_values(0)[0]
        == "direct_sun_hours_0301_1231_00_23_01"
    )


def test_radiance_parameters():
    """_"""

    assert (
        radiance_parameters(
            TEST_DAYLIGHT_MODEL, detail_dim=0.1, recipe_type="daylight-factor"
        )
        == "-aa 0.25 -ab 2 -ad 512 -ar 25 -as 128 -dc 0.25 -dj 0.0 -dp 64 -dr 0 -ds 0.5 -dt 0.5 -lr 4 -lw 0.05 -ss 0.0 -st 0.85"
    )

    with pytest.warns(UserWarning):
        assert (
            radiance_parameters(
                TEST_DAYLIGHT_MODEL, detail_dim=0.1, recipe_type="annual-daylight"
            )
            == "-ab 3 -ad 5000 -ar 10 -as 128 -c 1 -dc 0.25 -dp 64 -dr 0 -ds 0.5 -dt 0.5 -lr 4 -lw 2e-06 -ss 0.0 -st 0.85"
        )

    assert (
        radiance_parameters(
            TEST_DAYLIGHT_MODEL, detail_dim=0.1, recipe_type="sky-view", detail_level=1
        )
        == "-aa 0.2 -ab 3 -ad 2048 -ar 20 -as 2048 -dc 0.5 -dj 0.5 -dp 256 -dr 1 -ds 0.25 -dt 0.25 -lr 6 -lw 0.01 -ss 0.7 -st 0.5"
    )
