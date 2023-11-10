import pytest
from ladybug.epw import EPW
from ladybug_comfort.collection.utci import UTCI
from ladybugtools_toolkit.categorical.categories import UTCI_DEFAULT_CATEGORIES
from ladybugtools_toolkit.external_comfort.utci import (
    compare_monthly_utci,
    distance_to_comfortable,
    feasible_utci_limits,
    shade_benefit_category,
    utci,
)

from .. import EPW_FILE

EPW_OBJ = EPW(EPW_FILE)

LB_UTCI_COLLECTION = UTCI(
    EPW_OBJ.dry_bulb_temperature,
    EPW_OBJ.relative_humidity,
    EPW_OBJ.dry_bulb_temperature,
    EPW_OBJ.wind_speed,
).universal_thermal_climate_index


def test_utci():
    """_"""
    assert utci(
        EPW_OBJ.dry_bulb_temperature.values,
        EPW_OBJ.relative_humidity.values,
        EPW_OBJ.dry_bulb_temperature.values,
        EPW_OBJ.wind_speed.values,
    ).mean() == pytest.approx(LB_UTCI_COLLECTION.average, rel=2)


def test_compare_monthly_utci():
    """_"""
    # Test with default categories and no simplification
    a = compare_monthly_utci(
        [LB_UTCI_COLLECTION, LB_UTCI_COLLECTION + 5],
        utci_categories=UTCI_DEFAULT_CATEGORIES,
        identifiers=["test1", "test2"],
        density=True,
    )
    assert a.shape == (12, 20)
    assert a.sum().sum() == 24

    # test with use of comfort class categories
    a = compare_monthly_utci(
        [LB_UTCI_COLLECTION, LB_UTCI_COLLECTION + 5],
        utci_categories=UTCI_DEFAULT_CATEGORIES,
        identifiers=["test1", "test2"],
        density=True,
    )
    assert a.shape == (12, 20)


def test_shade_benefit_category():
    """_"""
    assert (
        shade_benefit_category(
            LB_UTCI_COLLECTION, LB_UTCI_COLLECTION - 5
        ).value_counts()["Comfortable without shade"]
        == 3009
    )
    assert (
        shade_benefit_category(
            LB_UTCI_COLLECTION, LB_UTCI_COLLECTION - 5, comfort_limits=(5, -10)
        ).value_counts()["Comfortable without shade"]
        == 3923
    )


def test_distance_to_comfortable():
    """_"""
    assert distance_to_comfortable(LB_UTCI_COLLECTION).average == pytest.approx(
        5.228092141408169, rel=0.01
    )


def test_feasible_utci_limits():
    """_"""
    a = feasible_utci_limits(
        EPW_OBJ, as_dataframe=True, include_additional_moisture=0.1
    )
    assert a.shape == (8760, 2)
    assert a.sum().sum() == pytest.approx(154055.81027975344, rel=0.0001)
