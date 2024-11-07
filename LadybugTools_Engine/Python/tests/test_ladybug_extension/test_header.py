import pytest
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybug.header import AnalysisPeriod, Header
from ladybugtools_toolkit.ladybug_extension.header import (header_from_string,
                                                           header_to_string)

header_good = Header(
    data_type=UniversalThermalClimateIndex(),
    unit="C",
    analysis_period=AnalysisPeriod(),
)
header_str_good = "Universal Thermal Climate Index (C)"

header_str_bad = "Something wrong"


def test_from_string_good():
    """_"""
    assert header_from_string(header_str_good) == header_good


def test_from_string_bad():
    """_"""
    with pytest.raises(ValueError):
        header_from_string(header_str_bad)


def test_to_string():
    """_"""
    assert header_to_string(header_good) == header_str_good
