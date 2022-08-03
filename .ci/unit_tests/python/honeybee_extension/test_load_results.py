import pytest
from ladybugtools_toolkit.honeybee_extension.results.load_ill import load_ill
from ladybugtools_toolkit.honeybee_extension.results.load_pts import load_pts
from ladybugtools_toolkit.honeybee_extension.results.load_res import load_res
from ladybugtools_toolkit.honeybee_extension.results.load_sql import load_sql
from ladybugtools_toolkit.honeybee_extension.results.make_annual import make_annual

from .. import ILL_FILE, PTS_FILE, RES_FILE, SQL_FILE


def test_load_ill():
    assert load_ill([ILL_FILE]).sum().sum() == pytest.approx(5779077.539738757, rel=1)


def test_load_sql():
    assert load_sql([SQL_FILE]).sum().sum() == pytest.approx(13418.630423320064, rel=1)


def test_load_res():
    assert load_res([RES_FILE]).sum().sum() == pytest.approx(657.2946374, rel=1)


def test_load_pts():
    assert load_pts([PTS_FILE]).sum().sum() == pytest.approx(180.0000011920929, rel=1)


def test_make_annual():
    assert make_annual(load_ill([ILL_FILE])).shape == (8760, 100)
