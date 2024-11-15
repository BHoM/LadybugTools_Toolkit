"""Methods for manipulating Ladybug header objects."""

import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype import TYPESDICT
from ladybug.datatype.generic import GenericType
from ladybug.header import Header
from python_toolkit.bhom.analytics import bhom_analytics


def header_to_string(header: Header) -> str:
    """Convert a Ladybug header object into a string.

    Args:
        header (Header):
            A Ladybug header object.

    Returns:
        str:
            A Ladybug header string."""

    return f"{header.data_type} ({header.unit})"


@bhom_analytics()
def header_to_multiindex(header: Header) -> pd.MultiIndex:
    """Convert a Ladybug header object into a Pandas MultiIndex. Used for creating CSV headers for
        reloading by Ladybug.

    Args:
        header (Header):
            A Ladybug header object.

    Returns:
        pd.MultiIndex:
            A Pandas MultiIndex object."""

    meta = header.to_csv_strings(True)
    names = [
        "datatype",
        "unit",
    ] + [i.split(":")[0].strip() for i in meta[2:]]
    values = [str(header.data_type), header.unit] + [i.split(":")[-1].strip() for i in meta[2:]]
    return pd.MultiIndex.from_arrays([[i] for i in values], names=names)


def header_from_string(string: str, is_leap_year: bool = False) -> Header:
    """Convert a string into a Ladybug header object.

    Args:
        string (str):
            A Ladybug header string.
        is_leap_year (bool, optional):
            A boolean to indicate whether the header is for a leap year. Default is False.

    Returns:
        Header:
            A Ladybug header object."""

    if "(" not in string or ")" not in string or string.count("(") > 1 or string.count(")") > 1:
        raise ValueError(
            "The string to be converted into a LB Header must be in the format 'variable (unit)'"
        )

    data_type, unit = string.split("(")
    data_type = data_type.strip()
    unit = unit.replace(")", "").strip()

    try:
        data_type = TYPESDICT[data_type.replace(" ", "")]()
    except KeyError:
        data_type = GenericType(name=data_type, unit=unit)

    return Header(
        data_type=data_type,
        unit=unit,
        analysis_period=AnalysisPeriod(is_leap_year=is_leap_year),
    )


@bhom_analytics()
def header_from_multiindex(multiindex: pd.MultiIndex) -> Header:
    """Convert a Pandas MultiIndex into a Ladybug header object.

    Args:
        multiindex (pd.MultiIndex):
            A Pandas MultiIndex object.

    Returns:
        Header:
            A Ladybug header object."""

    # ensure that the multi-index only contains a single "column"
    if len(multiindex) > 1:
        raise ValueError(
            "The multi-index passed contains more than one column and cannot be interpreted as a ",
            'single "Header" object.',
        )

    # get the datatype and value
    datatype = TYPESDICT[multiindex.values[0][0].replace(" ", "")]
    unit = multiindex.values[0][1]

    # get the metadata
    metadata = {}
    for n, (k, v) in enumerate(list(zip(*[multiindex.names, multiindex.values[0]]))):
        if n <= 1:
            continue
        metadata[k] = v

    # construct the header
    header = Header(
        data_type=datatype,
        unit=unit,
        analysis_period=AnalysisPeriod(),
        metadata=metadata,
    )

    return header


def combine_headers(headers: list[Header]) -> Header:
    """Combine like-headers into a single header, retaining metadata.

    Args:
        headers (list[Header]):
            A list of Ladybug headers.

    Returns:
        Header:
            A Ladybug header object.
    """

    # ensure each header shares the same dtype, unit and analysis period
    if not all([h.data_type == headers[0].data_type for h in headers]):
        raise ValueError("All headers must share the same data type.")
    if not all([h.unit == headers[0].unit for h in headers]):
        raise ValueError("All headers must share the same unit.")
    if not all([h.analysis_period == headers[0].analysis_period for h in headers]):
        raise ValueError("All headers must share the same analysis period.")

    # create a new header
    header = Header(
        data_type=headers[0].data_type,
        unit=headers[0].unit,
        analysis_period=headers[0].analysis_period,
        metadata={},
    )
    # combine metadata and add to resultant collection
    old_metadata = [c.metadata for c in headers]
    for n, dd in enumerate(old_metadata):
        for k, v in dd.items():
            if k in header.metadata:
                header.metadata[k] += f", {n}: {v}"
            else:
                header.metadata[k] = f"{n}: v"

    return header
