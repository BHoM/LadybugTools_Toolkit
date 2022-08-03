from datetime import datetime, timedelta

import pytest
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybug.header import AnalysisPeriod, Header
from ladybugtools_toolkit.ladybug_extension.header.from_string import from_string
from ladybugtools_toolkit.ladybug_extension.header.to_string import to_string

header_good = Header(
    data_type=UniversalThermalClimateIndex(),
    unit="C",
    analysis_period=AnalysisPeriod(),
)
header_str_good = "Universal Thermal Climate Index (C)"

header_str_bad = "Something wrong"


def test_from_string_good():
    assert from_string(header_str_good) == header_good


def test_from_string_bad():
    with pytest.raises(ValueError):
        from_string(header_str_bad)


def test_to_string():
    assert to_string(header_good) == header_str_good
