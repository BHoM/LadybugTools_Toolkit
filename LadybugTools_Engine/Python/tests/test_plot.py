import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from ladybug.analysisperiod import AnalysisPeriod
from ladybug_comfort.collection.utci import UTCI
from ladybugtools_toolkit.ladybug_extension.datacollection import \
    collection_to_series
from ladybugtools_toolkit.plot._degree_days import (cooling_degree_days,
                                                    degree_days,
                                                    heating_degree_days)
from ladybugtools_toolkit.plot._diurnal import diurnal, stacked_diurnals
from ladybugtools_toolkit.plot._heatmap import heatmap
from ladybugtools_toolkit.plot._psychrometric import psychrometric
from ladybugtools_toolkit.plot._radiant_cooling_potential import \
    radiant_cooling_potential
from ladybugtools_toolkit.plot._sunpath import sunpath
from ladybugtools_toolkit.plot._utci import (utci_comfort_band_comparison,
                                             utci_comparison_diurnal_day,
                                             utci_day_comfort_metrics,
                                             utci_heatmap_difference,
                                             utci_heatmap_histogram,
                                             utci_histogram, utci_journey,
                                             utci_pie)
from ladybugtools_toolkit.plot.colormaps import colormap_sequential
from ladybugtools_toolkit.plot.spatial_heatmap import spatial_heatmap
from ladybugtools_toolkit.plot.utilities import (contrasting_color,
                                                 create_triangulation,
                                                 lighten_color,
                                                 relative_luminance)

from . import EPW_OBJ

LB_UTCI_COLLECTION = UTCI(
    EPW_OBJ.dry_bulb_temperature,
    EPW_OBJ.relative_humidity,
    EPW_OBJ.dry_bulb_temperature,
    EPW_OBJ.wind_speed,
).universal_thermal_climate_index


def test_cooling_degree_days():
    """_"""
    assert isinstance(cooling_degree_days(epw=EPW_OBJ), plt.Axes)


def test_heating_degree_days():
    """_"""
    assert isinstance(heating_degree_days(epw=EPW_OBJ), plt.Axes)


def test_degree_days():
    """_"""
    assert isinstance(degree_days(epw=EPW_OBJ), plt.Figure)


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
    assert relative_luminance("#808080") == pytest.approx(
        0.215860500965604, rel=1e-7)


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
    assert sum(
        colormap_sequential(
            "red",
            "green",
            "blue")(0.25)) == pytest.approx(
        1.750003844675125,
        rel=0.01)


def test_spatial_heatmap():
    """_"""
    x = np.linspace(0, 100, 101)
    y = np.linspace(0, 100, 101)
    xx, yy = np.meshgrid(x, y)
    zz = (np.sin(xx) * np.cos(yy)).flatten()
    tri = create_triangulation(xx.flatten(), yy.flatten())
    assert isinstance(spatial_heatmap([tri], [zz], contours=[0]), plt.Figure)
    plt.close("all")


def test_radiant_cooling_potential():
    """_"""
    # create a pandas series of dew point temperature
    dpt = collection_to_series(EPW_OBJ.dew_point_temperature)

    # call the function
    ax = radiant_cooling_potential(dpt)

    # check that the returned object is a matplotlib axes
    assert isinstance(ax, plt.Axes)
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
    assert isinstance(
        sunpath(
            EPW_OBJ.location,
            analysis_period=AnalysisPeriod(
                st_month=3, end_month=4, st_hour=9, end_hour=18
            ),
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
        diurnal(
            collection_to_series(
                EPW_OBJ.dry_bulb_temperature),
            period="daily"),
        plt.Axes,
    )
    assert isinstance(
        diurnal(
            collection_to_series(
                EPW_OBJ.dry_bulb_temperature),
            period="weekly"),
        plt.Axes,
    )
    assert isinstance(
        diurnal(
            collection_to_series(
                EPW_OBJ.dry_bulb_temperature),
            period="monthly"),
        plt.Axes,
    )
    with pytest.raises(ValueError):
        diurnal(
            collection_to_series(
                EPW_OBJ.dry_bulb_temperature),
            period="decadely")
        diurnal(
            collection_to_series(
                EPW_OBJ.dry_bulb_temperature).reset_index(
                drop=True),
            period="monthly",
        )
        diurnal(
            collection_to_series(EPW_OBJ.dry_bulb_temperature),
            period="monthly",
            minmax_range=[0.95, 0.05],
        )
        diurnal(
            collection_to_series(EPW_OBJ.dry_bulb_temperature),
            period="monthly",
            quantile_range=[0.95, 0.05],
        )
    plt.close("all")

    assert isinstance(
        stacked_diurnals(
            datasets=[
                collection_to_series(EPW_OBJ.dry_bulb_temperature),
                collection_to_series(EPW_OBJ.relative_humidity),
            ]
        ),
        plt.Figure,
    )


def test_heatmap():
    """_"""
    assert isinstance(
        heatmap(collection_to_series(EPW_OBJ.dry_bulb_temperature)), plt.Axes
    )
    plt.close("all")

    mask = np.random.random(8760) > 0.5
    assert isinstance(
        heatmap(
            collection_to_series(
                EPW_OBJ.dry_bulb_temperature),
            mask=mask),
        plt.Axes)
    plt.close("all")

    mask_bad = np.random.random(10) > 0.5
    with pytest.raises(ValueError):
        heatmap(
            collection_to_series(
                EPW_OBJ.dry_bulb_temperature),
            mask=mask_bad)
    plt.close("all")

    assert isinstance(
        heatmap(
            pd.Series(
                np.random.random(21000),
                index=pd.date_range("2000-01-01", periods=21000, freq="h"),
            )
        ),
        plt.Axes,
    )
    plt.close("all")


def test_utci_comparison_diurnal():
    """_"""
    assert isinstance(utci_comparison_diurnal_day(
        [LB_UTCI_COLLECTION - 12, LB_UTCI_COLLECTION]), plt.Axes, )
    plt.close("all")


def test_utci_day_comfort_metrics():
    """_"""
    assert isinstance(
        utci_day_comfort_metrics(
            collection_to_series(LB_UTCI_COLLECTION),
            collection_to_series(EPW_OBJ.dry_bulb_temperature),
            collection_to_series(
                EPW_OBJ.dry_bulb_temperature, "Mean Radiant Temperature (C)"
            ),
            collection_to_series(EPW_OBJ.relative_humidity),
            collection_to_series(EPW_OBJ.wind_speed),
            month=6,
            day=21,
        ),
        plt.Axes,
    )
    plt.close("all")


def test_utci_heatmap_difference():
    """_"""
    assert isinstance(
        utci_heatmap_difference(
            LB_UTCI_COLLECTION,
            LB_UTCI_COLLECTION - 3),
        plt.Axes)
    plt.close("all")


def test_utci_heatmap_histogram():
    """_"""
    assert isinstance(utci_heatmap_histogram(LB_UTCI_COLLECTION), plt.Figure)
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
    assert isinstance(utci_comfort_band_comparison(
        [LB_UTCI_COLLECTION, LB_UTCI_COLLECTION + 1]), plt.Axes, )
    plt.close("all")


def test_psychrometric():
    """_"""
    assert isinstance(psychrometric(epw=EPW_OBJ), plt.Figure)
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
