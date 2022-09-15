from pathlib import Path
from typing import List, Union

import pandas as pd

from ladybugtools_toolkit.honeybee_extension.results.load_files import load_files
from ladybugtools_toolkit.honeybee_extension.results.load_pts_file import load_pts_file


from python_toolkit.bhom.analytics import analytics


@analytics
def load_pts(pts_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    """Load a single Radiance .pts file, or list of Radiance .pts files and return a combined DataFrame with the data.

    Args:
        pts_files (Union[str, Path, List[Union[str, Path]]]): A single .pts file, or a list of .pts files.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the input .pts files.
    """
    return load_files(load_pts_file, pts_files)
