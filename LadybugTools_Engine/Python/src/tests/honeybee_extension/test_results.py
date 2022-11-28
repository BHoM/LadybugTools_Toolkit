import pytest

from ladybugtools_toolkit.honeybee_extension.results import (
    load_ill,
    load_pts,
    load_res,
    load_sql,
    make_annual,
)

from ...tests import ILL_FILE, PTS_FILE, RES_FILE, SQL_FILE


def test_load_ill():
    """_"""
    assert load_ill([ILL_FILE]).sum().sum() == pytest.approx(5779077.539738757, rel=1)


def test_load_sql():
    """_"""
    assert load_sql([SQL_FILE]).sum().sum() == pytest.approx(13418.630423320064, rel=1)


def test_load_res():
    """_"""
    assert load_res([RES_FILE]).sum().sum() == pytest.approx(657.2946374, rel=1)


def test_load_pts():
    """_"""
    assert load_pts([PTS_FILE]).sum().sum() == pytest.approx(180.0000011920929, rel=1)


def test_make_annual():
    """_"""
    assert make_annual(load_ill([ILL_FILE])).shape == (8760, 100)
