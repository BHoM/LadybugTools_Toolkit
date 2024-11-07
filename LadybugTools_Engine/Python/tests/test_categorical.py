import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from ladybugtools_toolkit.categorical.categories import (
    UTCI_DEFAULT_CATEGORIES, Categorical, CategoricalComfort, ComfortClass)
from ladybugtools_toolkit.ladybug_extension.datacollection import \
    collection_to_series
from matplotlib.legend import Legend

from . import EPW_OBJ

TEST_TIMESERIES_DATA = collection_to_series(EPW_OBJ.dry_bulb_temperature)

TEST_CATEGORICAL_UNBOUNDED = Categorical(
    bins=[-np.inf, 0, np.inf],
    colors=["blue", "red"],
)
TEST_CATEGORICAL_BOUNDED = Categorical(
    bins=[-100, 0, 100],
    colors=["blue", "red"],
)


def test_categorical():
    """_"""
    assert isinstance(Categorical(bins=[0, 1, 2]), Categorical)

    with pytest.raises(ValueError):
        Categorical(bins=[0, 1, 2], colors=["blue", "red", "green"])

    assert isinstance(repr(TEST_CATEGORICAL_UNBOUNDED), str)
    assert isinstance(str(TEST_CATEGORICAL_UNBOUNDED), str)


def test_generic_categorical_properties():
    """_"""
    assert TEST_CATEGORICAL_UNBOUNDED.name == "GenericCategories"
    assert TEST_CATEGORICAL_UNBOUNDED.bin_names == [
        "(-inf, 0.0]", "(0.0, inf]"]
    assert TEST_CATEGORICAL_UNBOUNDED.colors == ("#0000ffff", "#ff0000ff")
    assert TEST_CATEGORICAL_UNBOUNDED.bins == [-np.inf, 0, np.inf]
    assert TEST_CATEGORICAL_UNBOUNDED.bins_finite == (0,)
    assert TEST_CATEGORICAL_UNBOUNDED.bin_names_detailed == [
        "(-inf, 0.0] (-inf to 0.0)",
        "(0.0, inf] (0.0 to inf)",
    ]
    assert TEST_CATEGORICAL_UNBOUNDED.descriptions == [
        "-inf to 0.0", "0.0 to inf"]
    assert len(TEST_CATEGORICAL_BOUNDED.lb_colors) == 2


def test_categorical_computed_properties():
    """_"""
    assert TEST_CATEGORICAL_UNBOUNDED.color_from_bin_name(
        "(-inf, 0.0]") == "#0000ffff"

    with pytest.raises(ValueError):
        TEST_CATEGORICAL_UNBOUNDED.get_color(value=3)
    assert TEST_CATEGORICAL_BOUNDED.get_color(value=3) == "#ff0000ff"
    assert isinstance(
        TEST_CATEGORICAL_BOUNDED.interval_from_bin_name("(0, 100]"), pd.Interval)

    assert isinstance(TEST_CATEGORICAL_BOUNDED.categorise([3]), pd.Categorical)
    assert (
        TEST_CATEGORICAL_BOUNDED.summarise([-5, -2, 0, 10])
        == '"(-100, 0]" occurs 3 times (75.0%).\n"(0, 100]" occurs 1 times (25.0%).'
    )
    assert isinstance(TEST_CATEGORICAL_BOUNDED.value_counts(
        [-5, -2, 0, 10]), pd.Series)

    with pytest.raises(ValueError):
        TEST_CATEGORICAL_BOUNDED.get_color(-1000)


def test_categorical_comfort():
    """_"""
    assert len(UTCI_DEFAULT_CATEGORIES.comfort_classes) == 10
    assert isinstance(UTCI_DEFAULT_CATEGORIES.comfort_classes[0], ComfortClass)
    assert len(UTCI_DEFAULT_CATEGORIES.simplify().comfort_classes) == 3

    with pytest.raises(ValueError):
        CategoricalComfort(
            bins=(-np.inf, 9, 26, np.inf),
            bin_names=("Too cold", "Comfortable", "Too hot"),
            colors=("blue", "yellow", "red"),
            name="example",
            comfort_classes=[
                ComfortClass.TOO_COLD,
                ComfortClass.TOO_HOT,
            ],
        )
        CategoricalComfort(
            bins=(-np.inf, 9, 26, np.inf),
            bin_names=("Too cold", "Comfortable", "Too hot"),
            colors=("blue", "yellow", "red"),
            name="example",
        )
        CategoricalComfort(
            bins=(-np.inf, 9, 26, np.inf),
            bin_names=("Too cold", "Comfortable", "Too hot"),
            colors=("blue", "yellow", "red"),
            name="example",
            comfort_classes=[
                ComfortClass.TOO_COLD,
                ComfortClass.TOO_COLD,
                ComfortClass.TOO_HOT,
            ],
        )


def test_from():
    """_"""
    assert isinstance(
        Categorical.from_cmap(
            bins=[
                0,
                1,
                2],
            cmap=plt.get_cmap("inferno")),
        Categorical)
    assert isinstance(Categorical.from_cmap(
        bins=[-np.inf, 1, 2], cmap=plt.get_cmap("inferno")), Categorical, )
    assert isinstance(
        Categorical.from_cmap(
            bins=[
                0,
                1,
                np.inf],
            cmap=plt.get_cmap("inferno")),
        Categorical,
    )


def test_legend():
    """_"""
    assert isinstance(TEST_CATEGORICAL_UNBOUNDED.create_legend(), Legend)
    plt.close("all")


def test_timeseries_summary_monthly():
    """_"""
    s = TEST_CATEGORICAL_BOUNDED.timeseries_summary_monthly(
        TEST_TIMESERIES_DATA)
    assert isinstance(s, pd.DataFrame)
    assert s.shape == (12, 2)
    assert s.sum().sum() == 8760
    assert s.quantile(0.8).mean() == pytest.approx(392.7, abs=0.01)

    with pytest.raises(ValueError):
        TEST_CATEGORICAL_BOUNDED.timeseries_summary_monthly(
            "not_a_timeseries_data")
        TEST_CATEGORICAL_BOUNDED.timeseries_summary_monthly(
            TEST_TIMESERIES_DATA.reset_index(drop=True)
        )


def test_annual_heatmap():
    """_"""
    assert isinstance(
        TEST_CATEGORICAL_BOUNDED.annual_heatmap(TEST_TIMESERIES_DATA), plt.Axes
    )
    plt.close("all")


def test_annual_monthly_histogram():
    """_"""
    assert isinstance(TEST_CATEGORICAL_BOUNDED.annual_monthly_histogram(
        TEST_TIMESERIES_DATA), plt.Axes, )
    plt.close("all")

    assert isinstance(
        TEST_CATEGORICAL_BOUNDED.annual_monthly_histogram(
            TEST_TIMESERIES_DATA, show_labels=True
        ),
        plt.Axes,
    )
    plt.close("all")

    assert isinstance(
        TEST_CATEGORICAL_BOUNDED.annual_monthly_histogram(
            TEST_TIMESERIES_DATA, show_legend=True
        ),
        plt.Axes,
    )
    plt.close("all")
