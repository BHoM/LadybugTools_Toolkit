import sys
sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype import TYPESDICT
from ladybug.datatype.generic import GenericType
from ladybug.header import Header


def to_string(header: Header) -> str:
    """Convert a Ladybug header object into a string."""
    header_str = header.to_csv_strings()
    return f"{header.data_type} ({header.unit})"


def from_string(string: str) -> Header:
    """Convert a string into a Ladybug header object."""

    try:
        str_elements = string.split(" ")
        _unit = str_elements[-1].replace("(", "").replace(")", "")
        _data_type = " ".join(str_elements[:-1])
    except AttributeError:
        raise ValueError(
            'The input series must have a name in the format "variable (unit)".'
        )
    try:
        _data_type = TYPESDICT[_data_type.replace(" ", "")]()
    except KeyError:
        _data_type = GenericType(name=_data_type, unit=_unit)

    return Header(data_type=_data_type, unit=_unit, analysis_period=AnalysisPeriod())

if __name__ == "__main__":
    pass