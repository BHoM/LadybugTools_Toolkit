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
    
    month_means = []
    for month in range(12):
        month_series = series[series.index.month == month + 1]
        month_means.append(month_series.mean())
    
    return {
        "lowest": lowest,
        "lowest_index": lowest_index,
        "highest": highest,
        "highest_index": highest_index,
        "median": median,
        "mean": mean,
        "month_means": month_means,
        }