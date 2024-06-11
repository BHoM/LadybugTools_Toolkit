import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def histogram(
    series: pd.Series,
    bins: list(float) | list(int) = None,
    **kwargs,
    ):
    if bins is None:
        bins = np.linspace(series.values.min(), series.values.max(), 31)
    elif len(bins) <= 1:
        bins = np.linspace(series.values.min(), series.values.max(), 31)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.hist(series.values, bins=bins, label = series.name, density=False)

    legend = kwargs.pop("legend")

    if legend:
        ax.legend()

    return ax