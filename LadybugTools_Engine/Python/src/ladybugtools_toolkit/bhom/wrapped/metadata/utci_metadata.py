from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import \
    UniversalThermalClimateIndex as LB_UniversalThermalClimateIndex
from ladybugtools_toolkit.ladybug_extension.datacollection import \
    collection_to_series


def utci_metadata(
    utci_collection: HourlyContinuousCollection,
    comfort_lower: float = 9,
    comfort_higher: float = 26,
    use_start_hour: int = 7,
    use_end_hour: int = 23,
) -> dict:
    """Returns a dictionary of useful metadata for the given collection dependant on the given comfortable range.

    Args:
        utci_collection (HourlyContinuousCollection):
            utci headered ladybug hourly collection

        comfort_lower (float):
            lower value for the comfortable temperature range, where temperatures exclusively below this are too cold.

        comfort_higher (float):
            higher value for the comfortable temperature range, where temperatures above and equal to this are too hot.

        use_start_hour (int):
            start hour to filter usage time, inclusive

        use_end_hour (int):
            end hour to filter usage time, exclusive

    Returns:
        dict:
            dictionary containing comfortable, hot and cold ratios, structured as follows:
            {
                'comfortable_ratio': ratio_of_comfortable_hours,
                'hot_ratio': ratio_of_hot_hours,
                'cold_ratio': ratio_of_cold_hours,
                'daytime_comfortable': daytime_comfortable_ratio,
                'daytime_hot': daytime_hot_ratio,
                'daytime_cold': daytime_cold_ratio
            }
    """
    if not isinstance(
        utci_collection.header.data_type, LB_UniversalThermalClimateIndex
    ):
        raise ValueError("Input collection is not a UTCI collection.")

    if not comfort_lower < comfort_higher:
        raise ValueError(
            f"The lower comfort temperature {comfort_lower}, must be less than the higher comfort temperature {comfort_higher}."
        )

    series = collection_to_series(utci_collection)

    daytime = series.loc[(series.index.hour >= use_start_hour) & (
        series.index.hour < use_end_hour)]

    comfortable_ratio = (
        (series >= comfort_lower) & (series < comfort_higher)
    ).sum() / len(series)
    hot_ratio = (series >= comfort_higher).sum() / len(series)
    cold_ratio = (series < comfort_lower).sum() / len(series)

    day_comfortable = (
        (daytime >= comfort_lower) & (daytime < comfort_higher)
    ).sum() / len(daytime)
    day_hot = (daytime >= comfort_higher).sum() / len(daytime)
    day_cold = (daytime < comfort_lower).sum() / len(daytime)

    return {
        "comfortable_ratio": comfortable_ratio,
        "hot_ratio": hot_ratio,
        "cold_ratio": cold_ratio,
        "daytime_comfortable": day_comfortable,
        "daytime_hot": day_hot,
        "daytime_cold": day_cold,
    }
