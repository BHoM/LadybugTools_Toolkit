import numpy as np

from ladybug._datacollectionbase import BaseCollection


from python_toolkit.bhom.analytics import analytics


@analytics
def to_array(collection: BaseCollection) -> np.ndarray:
    """Convert a Ladybug BaseCollection-like object into a numpy array.

    Args:
        collection: A Ladybug BaseCollection-like object.

    Returns:
        np.ndarray: A numpy array.
    """

    return np.array(collection.values)
