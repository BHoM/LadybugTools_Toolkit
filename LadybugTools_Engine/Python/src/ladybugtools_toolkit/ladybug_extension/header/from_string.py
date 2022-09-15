from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype import TYPESDICT
from ladybug.datatype.generic import GenericType
from ladybug.header import Header


from python_toolkit.bhom.analytics import analytics


@analytics
def from_string(string: str) -> Header:
    """Convert a string into a Ladybug header object.

    Args:
        string (str):
            A Ladybug header string.

    Returns:
        Header:
            A Ladybug header object."""

    str_elements = string.split(" ")

    if (len(str_elements) < 2) or ("(" not in string) or (")" not in string):
        raise ValueError(
            "The string to be converted into a LB Header must be in the format 'variable (unit)'"
        )

    str_elements = string.split(" ")
    unit = str_elements[-1].replace("(", "").replace(")", "")
    data_type = " ".join(str_elements[:-1])

    try:
        data_type = TYPESDICT[data_type.replace(" ", "")]()
    except KeyError:
        data_type = GenericType(name=data_type, unit=unit)

    return Header(data_type=data_type, unit=unit, analysis_period=AnalysisPeriod())
