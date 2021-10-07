from ladybug.epw import EPW
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datacollectionimmutable import HourlyContinuousCollectionImmutable


def get_location_str(lb_obj) -> str:
    """Generate a location string describing where a data collection or EPW is from.

    Args:
        lb_obj (EPW or HourlyContinuousCollection): A ladybug EPW or data collection.

    Returns:
        str: A descriptive location string.
    """
    try:
        city = lb_obj.header.metadata["city"].strip()
        country = lb_obj.header.metadata["country"].strip()
        source = lb_obj.header.metadata["source"].strip()
        return f"{city} ({country}) {source}"
    except Exception:
        try:
            city = lb_obj.location.city.strip()
            country = lb_obj.location.country.strip()
            source = lb_obj.metadata["source"].strip()
            return f"{city} ({country}) {source}"
        except:
            return "No location data available"
