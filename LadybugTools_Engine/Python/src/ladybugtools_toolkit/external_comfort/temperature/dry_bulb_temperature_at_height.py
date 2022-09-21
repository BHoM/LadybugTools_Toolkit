from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.temperature.temperature_at_height import (
    temperature_at_height,
)


from ladybugtools_toolkit import analytics


@analytics
def dry_bulb_temperature_at_height(
    epw: EPW, target_height: float
) -> HourlyContinuousCollection:
    """Translate DBT values from an EPW into

    Args:
        dry_bulb_temperature_collection (HourlyContinuousCollection): _description_
        target_height (float): _description_

    Returns:
        HourlyContinuousCollection: _description_
    """
    dbt_collection = epw.dry_bulb_temperature.__copy__()
    dbt_collection.values = [
        temperature_at_height(i, 10, target_height)
        for i in epw.dry_bulb_temperature.values
    ]
    return dbt_collection
