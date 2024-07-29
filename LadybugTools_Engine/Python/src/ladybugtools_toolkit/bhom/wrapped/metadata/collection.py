from ladybug.datacollection import BaseCollection
from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series

def collection_metadata(collection: BaseCollection) -> dict:
    """Returns a dictionary containing useful metadata about the series.
    
    Args:
        collection (BaseCollection):
            ladybug data collection object
    
    Returns:
        dict:
            A dictionary containing metadata about the collection, structured:
            {
                "lowest": lowest,
                "lowest_index": lowest_index,
                "highest": highest,
                "highest_index": highest_index,
                "median": median,
                "mean": mean,
                "month_means": [month_means],
                "month_diurnal_ranges": [month_diurnal_ranges],
            }
            where month_means is a list of means indexed by month, and month_ranges is a list of diurnal month ranges as tuples: (min, max).
    """
    
    series = collection_to_series(collection)
    lowest = series.min()
    highest = series.max()
    lowest_index = series.idxmin()
    highest_index = series.idxmax()
    median = series.quantile(0.5)
    mean = series.mean()

    series_diurnal_mean = series.groupby([series.index.month, series.index.hour]).mean()
    series_diurnal_max = series_diurnal_mean.groupby(
        series_diurnal_mean.index.get_level_values(0)
    ).max()
    series_diurnal_min = series_diurnal_mean.groupby(
        series_diurnal_mean.index.get_level_values(0)
    ).min()
    
    month_means = []
    month_ranges = []
    for month in range(12):
        month_series = series[series.index.month == month + 1]
        month_means.append(month_series.mean())
        month_ranges.append((series_diurnal_min.iloc[month], series_diurnal_max.iloc[month]))
    
    return {
        "lowest": lowest,
        "lowest_index": lowest_index,
        "highest": highest,
        "highest_index": highest_index,
        "median": median,
        "mean": mean,
        "month_means": month_means,
        "month_diurnal_ranges": month_ranges,
        }