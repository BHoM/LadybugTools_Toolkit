import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compare_histogram(
    serieses: list(pd.Series),
    bins: list(float) | list(int),
    names: list(str) = None,
    **kwargs
    ) -> plt.Axes:
    if names is None:
        names = [series.name for series in serieses]
    elif len(names) != len(serieses):
        raise ValueError(f"The series count ({len(serieses)}) must match the count of names ({len(names)}).")
    
    if bins is None:
        bins = np.linspace(df.values.min(), df.values.max(), 31)
    elif len(bins) <= 1:
        bins = np.linspace(df.values.min(), df.values.max(), 31)

    df = pd.concat(serieses, axis=1, keys=names)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.hist(df.values, bins=bins, label = df.columns, density=False)

    return ax

def compare_line(
    serieses: list(pd.Series),
    names: list(str) = None,
    **kwargs
    ) -> plt.Axes:
    if names is None:
        names = [series.name for series in serieses]
    elif len(names) != len(serieses):
        raise ValueError(f"The series count ({len(serieses)}) must match the count of names ({len(names)}).")
    
    df = pd.concat(serieses, axis=1, keys=names)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    df.plot(ax=ax)
    return ax
    