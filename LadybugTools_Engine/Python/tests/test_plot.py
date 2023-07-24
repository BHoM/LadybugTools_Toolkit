import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.epw import EPW
from ladybug_comfort.collection.utci import UTCI
from ladybugtools_toolkit.external_comfort.material import Materials
from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
from ladybugtools_toolkit.plot import diurnal, heatmap, sunpath, windrose
from ladybugtools_toolkit.plot._utci import (
    utci_comfort_band_comparison,
    utci_comparison_diurnal_day,
    utci_day_comfort_metrics,
    utci_distance_to_comfortable,
    utci_heatmap,
    utci_heatmap_difference,
    utci_heatmap_histogram,
    utci_histogram,
    utci_journey,
    utci_pie,
)
from ladybugtools_toolkit.plot.colormaps import colormap_sequential
from ladybugtools_toolkit.plot.spatial_heatmap import spatial_heatmap
from ladybugtools_toolkit.plot.utilities import (
    contrasting_color,
    create_triangulation,
    lighten_color,
    relative_luminance,
)

from . import EPW_FILE

EPW_OBJ = EPW(EPW_FILE)

GROUND_MATERIAL = Materials.LBT_AsphaltPavement.value
LB_UTCI_COLLECTION = UTCI(
    EPW_OBJ.dry_bulb_temperature,
    EPW_OBJ.relative_humidity,
    EPW_OBJ.dry_bulb_temperature,
    EPW_OBJ.wind_speed,
).universal_thermal_climate_index


def test_create_triangulation():
    """_"""
    # Test with valid input
    x, y = np.meshgrid(range(10), range(10))
    triang = create_triangulation(x.flatten(), y.flatten())
    assert len(triang.triangles) == 162

    # Test with invalid input
    x = [0, 1, 2, 3, 4, 5]
    y = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        create_triangulation(x, y)

    # Test with alpha value that is too small
    x, y = np.meshgrid(range(0, 100, 10), range(0, 100, 10))
    with pytest.raises(ValueError):
        create_triangulation(x, y, alpha=0.00001)


def test_relative_luminance():
    """_"""
    assert relative_luminance("#FFFFFF") == pytest.approx(1.0, rel=1e-7)
    assert relative_luminance("#000000") == pytest.approx(0.0, rel=1e-7)
    assert relative_luminance("#808080") == pytest.approx(0.215860500965604, rel=1e-7)


def test_contrasting_color():
    """_"""
    assert contrasting_color("#FFFFFF") == ".15"
    assert contrasting_color("#000000") == "w"
    assert contrasting_color("#808080") == "w"


def test_lighten_color():
    """_"""
    # Test lightening a named color
    assert lighten_color("g", 0.3) == (
        0.5500000000000002,
        0.9999999999999999,
        0.5500000000000002,
    )

    # Test lightening a hex color
    assert lighten_color("#F034A3", 0.6) == (
        0.9647058823529411,
        0.5223529411764707,
        0.783529411764706,
    )

    # Test lightening an RGB color
    assert lighten_color((0.3, 0.55, 0.1), 0.5) == (
        0.6365384615384615,
        0.8961538461538462,
        0.42884615384615377,
    )

    # Test lightening a color by 0
    assert lighten_color("g", 0) == (1.0, 1.0, 1.0)

    # Test lightening a color by 1
    assert lighten_color("g", 1) == (0.0, 0.5, 0.0)


def test_colormap_sequential():
    """_"""
    assert sum(colormap_sequential("red", "green", "blue")(0.25)) == pytest.approx(
        1.750003844675125, rel=0.01
    )


def test_spatial_heatmap():
    """_"""
    x = np.linspace(0, 100, 101)
    y = np.linspace(0, 100, 101)
    xx, yy = np.meshgrid(x, y)
    zz = (np.sin(xx) * np.cos(yy)).flatten()
    tri = create_triangulation(xx.flatten(), yy.flatten())
    assert isinstance(spatial_heatmap([tri], [zz], contours=[0]), plt.Figure)
    plt.close("all")


def test_sunpath():
    """_"""
    assert isinstance(
        sunpath(
            EPW_OBJ.location,
            analysis_period=AnalysisPeriod(
                st_month=3, end_month=4, st_hour=9, end_hour=18
            ),
            data_collection=EPW_OBJ.dry_bulb_temperature,
            cmap="inferno",
        ),
        plt.Axes,
    )
    plt.close("all")


def test_timeseries_diurnal():
    """_"""
    assert isinstance(
        diurnal(collection_to_series(EPW_OBJ.dry_bulb_temperature)), plt.Axes
    )
    assert isinstance(
        diurnal(collection_to_series(EPW_OBJ.dry_bulb_temperature), period="daily"),
        plt.Axes,
    )
    assert isinstance(
        diurnal(collection_to_series(EPW_OBJ.dry_bulb_temperature), period="weekly"),
        plt.Axes,
    )
    assert isinstance(
        diurnal(collection_to_series(EPW_OBJ.dry_bulb_temperature), period="monthly"),
        plt.Axes,
    )
    with pytest.raises(ValueError):
        diurnal(collection_to_series(EPW_OBJ.dry_bulb_temperature), period="decadely")
    plt.close("all")


def test_heatmap():
    """_"""
    assert isinstance(
        heatmap(collection_to_series(EPW_OBJ.dry_bulb_temperature)), plt.Axes
    )
    plt.close("all")


def test_utci_comparison_diurnal():
    """_"""
    assert isinstance(
        utci_comparison_diurnal_day([LB_UTCI_COLLECTION - 12, LB_UTCI_COLLECTION]),
        plt.Axes,
    )
    plt.close("all")


def test_utci_day_comfort_metrics():
    """_"""
    assert isinstance(
        utci_day_comfort_metrics(
            collection_to_series(LB_UTCI_COLLECTION),
            collection_to_series(EPW_OBJ.dry_bulb_temperature),
            collection_to_series(EPW_OBJ.dry_bulb_temperature).rename(
                "Mean Radiant Temperature (C)"
            ),
            collection_to_series(EPW_OBJ.relative_humidity),
            collection_to_series(EPW_OBJ.wind_speed),
            month=6,
            day=21,
        ),
        plt.Axes,
    )
    plt.close("all")


def test_utci_distance_to_comfortable():
    """_"""
    assert isinstance(utci_distance_to_comfortable(LB_UTCI_COLLECTION), plt.Axes)
    plt.close("all")


def test_utci_heatmap_difference():
    """_"""
    assert isinstance(
        utci_heatmap_difference(LB_UTCI_COLLECTION, LB_UTCI_COLLECTION - 3), plt.Axes
    )
    plt.close("all")


def test_utci_heatmap_histogram():
    """_"""
    assert isinstance(utci_heatmap_histogram(LB_UTCI_COLLECTION), plt.Figure)
    plt.close("all")


def test_utci_heatmap():
    """_"""
    assert isinstance(utci_heatmap(LB_UTCI_COLLECTION), plt.Axes)
    plt.close("all")


def test_utci_journey():
    """_"""
    assert isinstance(
        utci_journey(
            utci_values=(10, 30, 15, 0),
            ax=None,
            names=("A", "B", "C", "D"),
            curve=True,
            show_legend=True,
            title="Hey there",
            show_grid=True,
            ylim=(-10, 50),
        ),
        plt.Axes,
    )
    plt.close("all")


def test_utci_histogram():
    """_"""
    assert isinstance(utci_histogram(LB_UTCI_COLLECTION), plt.Axes)
    plt.close("all")


def test_utci_comfort_band_comparison():
    """_"""
    assert isinstance(
        utci_comfort_band_comparison([LB_UTCI_COLLECTION, LB_UTCI_COLLECTION + 1]),
        plt.Axes,
    )
    plt.close("all")


def test_windrose():
    """_"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert isinstance(
            windrose(
                wind_directions=EPW_OBJ.wind_direction.values,
                data=EPW_OBJ.wind_speed.values,
                ax=None,
                data_bins=[0, 90, 180, 270, 360],
                include_legend=False,
                include_percentages=True,
            ),
            plt.Axes,
        )
    plt.close("all")


# def test_fisheye_sky():
#     """_"""
#     model = create_model(
#         GROUND_MATERIAL.to_lbt(), GROUND_MATERIAL.to_lbt(), EXTERNAL_COMFORT_IDENTIFIER
#     )
#     img = fisheye_sky(model, Point3D(0, 0, 1.2))
#     assert isinstance(img, Image)
#     img.close()


# def test_skymatrix():
#     """_"""
#     assert isinstance(skymatrix(EPW_OBJ), plt.Figure)
#     plt.close("all")


def test_utci_pie():
    """_"""
    assert isinstance(utci_pie(LB_UTCI_COLLECTION), plt.Axes)
    plt.close("all")
