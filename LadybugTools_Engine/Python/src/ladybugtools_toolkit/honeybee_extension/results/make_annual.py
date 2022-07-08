import pandas as pd


def make_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataFrame with partial annual data to a DataFrame with annual data.

    Args:
        df (pd.DataFrame): A DataFrame with partially annually indexed data.

    Returns:
        pd.DataFrame: A DataFrame with annually indexed data.
    """
    assert (
        df.index.year.min() == df.index.year.max()
    ), "The DataFrame must be indexed with annual data within a single year."

    year = df.index[0].year
    freq = f"{(df.index[1] - df.index[0]).total_seconds():0.0f}S"
    minutes_of_hour = df.index.minute.unique()
    df2 = pd.DataFrame(
        index=pd.date_range(
            f"{year}-01-01 00:{minutes_of_hour.min()}:00",
            f"{year + 1}-01-01 00:{minutes_of_hour.min()}:00",
            freq=freq,
        )[:-1]
    )
    df_reindexed = pd.concat([df2, df], axis=1)
    df_reindexed.columns = pd.MultiIndex.from_tuples(df_reindexed.columns)
    return df_reindexed
