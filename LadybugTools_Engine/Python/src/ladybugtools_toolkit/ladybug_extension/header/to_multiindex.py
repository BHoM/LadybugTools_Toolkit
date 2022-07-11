import pandas as pd
from ladybug.header import Header


def to_multiindex(header: Header) -> pd.MultiIndex:
    """Convert a Ladybug header object into a Pandas MultiIndex.
    Used for creating CSV headers for reloading by Ladybug.

    Args:
        header (Header): A Ladybug header object.

    Returns:
        pd.MultiIndex: A Pandas MultiIndex object."""

    meta = header.to_csv_strings(True)
    names = [
        "datatype",
        "unit",
    ] + [i.split(":")[0].strip() for i in meta[2:]]
    values = [str(header.data_type), header.unit] + [
        i.split(":")[-1].strip() for i in meta[2:]
    ]
    return pd.MultiIndex.from_arrays([[i] for i in values], names=names)
