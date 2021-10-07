from ladybug.datacollection import HourlyContinuousCollection

import sys
sys.path.append(r"C:\ProgramData\BHoM\Extensions")
from LadybugTools.get_location_str import get_location_str


def get_title_str(collection: HourlyContinuousCollection) -> str:
    """Generate a plot title string from a data collections metadata.

    Args:
        collection (HourlyContinuousCollection): A ladybug data collection.

    Returns:
        str: A title string.
    """
    return "\n".join(
        [
            get_location_str(collection),
            f"{collection.header.data_type.ToString()} ({collection.header.unit})",
        ]
    )
