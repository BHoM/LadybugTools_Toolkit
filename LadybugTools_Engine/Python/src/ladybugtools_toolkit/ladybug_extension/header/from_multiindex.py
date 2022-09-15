import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype import TYPESDICT
from ladybug.header import Header


from python_toolkit.bhom.analytics import analytics


@analytics
def from_multiindex(multiindex: pd.MultiIndex) -> Header:
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
