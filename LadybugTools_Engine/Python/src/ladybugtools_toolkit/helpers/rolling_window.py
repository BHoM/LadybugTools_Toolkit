from typing import Any, List

import numpy as np


def rolling_window(array: List[Any], window: int):
    """Throwaway function here to roll a window along a list.

    Args:
        array (List[Any]):
            A 1D list of some kind.
        window (int):
            The size of the window to apply to the list.

    Returns:
        List[List[Any]]:
            The resulting, "windowed" list.
    """
    a: np.ndarray = np.array(array)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
