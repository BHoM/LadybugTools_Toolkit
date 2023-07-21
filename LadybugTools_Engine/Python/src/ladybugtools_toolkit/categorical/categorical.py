from dataclasses import dataclass, field
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from ladybug.legend import Color
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    ListedColormap,
    is_color_like,
    to_hex,
    to_rgba,
)
from matplotlib.legend import Legend

from ..helpers import rolling_window


@dataclass(init=True, repr=True)
class Categorical:
    """A class to hold categorical data.

    Args:
        bins (Tuple[float], optional):
            The bin edges for the categories. These are righ-inclusive, with the exception of the first bin which is
            also left-inclusive.
        bin_names (Tuple[str], optional):
            The names of the categories.
        colors (Tuple[Union[str, Tuple]], optional):
            The colors of the categories.
        name (str, optional):
            The name of the categories.
    """

    bins: Tuple[float] = field(default_factory=tuple, repr=False)
    bin_names: Tuple[str] = field(default_factory=tuple)
    colors: Tuple[str] = field(default_factory=tuple, repr=True)
    name: str = field(default="GenericCategories")

    def __post_init__(self):
        if len(self.colors) == 0:
            self.colors = [to_hex("k")] * (len(self.bins) - 1)

        for color in self.colors:
            if not is_color_like(color):
                raise ValueError(f"{color} is not a valid color.")
        self.colors = tuple([to_hex(i, keep_alpha=True) for i in self.colors])

        if len(self.bin_names) == 0:
            self.bin_names = [str(i) for i in self.interval_index]

        if len(set([len(self.bin_names), len(self.colors), (len(self.bins) - 1)])) > 1:
            raise ValueError(
                f"The number of colors ({len(self.colors)}) and bin names ({len(self.bin_names)}) must be one less than the number of bins ({len(self.bins)} - 1)."
            )

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"

    def __str__(self):
        return repr(self)

    def __iter__(self):
        for idx in self.interval_index:
            yield idx

    def __len__(self):
        return len(self.interval_index)

    def __getitem__(self, item):
        return self.interval_index[item]

    @classmethod
    def from_cmap(
        cls,
        bins: Tuple[float],
        cmap: Colormap,
        bin_names: Tuple[str] = (),
        name: str = "",
    ):
        """Create a categorical from a colormap.

        Args:
            bins (Tuple[float]):
                The bin edges for the categories.
            cmap (Colormap):
                The colormap to use.
            bin_names (Tuple[str], optional):
                The names for each of the bins. Defaults to () which names each bin using its boundaries.
            name (str, optional):
                The name for this categories object. Defaults to "" which uses the colormap name.

        Returns:
            Categories: The resulting categories object.
        """
        if name == "":
            name = f"GenericCategories_{cmap.name}"

        # get the midpoints for the bins
        groups = rolling_window(bins, 2)
        _bins = [np.mean(i) for i in groups]

        if np.isinf(_bins[0]):
            _bins[0] = bins[1]

        if np.isinf(_bins[-1]):
            _bins[-1] = bins[-2]

        normalised = np.interp(_bins, [min(bins), max(bins)], [0, 1])
        colors = [cmap(i) for i in normalised]

        return cls(bins=bins, bin_names=bin_names, colors=colors, name=name)

    @property
    def interval_index(self) -> pd.IntervalIndex:
        """The pandas interval index form of the bins.

        Returns:
            pd.IntervalIndex:
                The pandas interval index form of the bins.
        """
        return pd.IntervalIndex.from_breaks(self.bins)

    @property
    def descriptions(self) -> List[str]:
        """The descriptions of the categories.

        Returns:
            List[str]:
                The descriptions of the categories.
        """
        return [f"{i.left}<x<={i.right}" for i in self]

    @property
    def cmap(self) -> Colormap:
        """The matplotlib colormap.

        Returns:
            Colormap:
                The matplotlib colormap.
        """

        # get upper/lower bound colors if ends are inf

        if np.isinf(self.bins[0]) and np.isinf(self.bins[-1]):
            colors = self.colors[1:-1]
            # above_color = self.colors[-1]
            # below_color = self.colors[0]
        elif np.isinf(self.bins[0]) and not np.isinf(self.bins[-1]):
            colors = self.colors[1:]
            # above_color = self.colors[-1]
            # below_color = self.colors[0]
        elif not np.isinf(self.bins[0]) and np.isinf(self.bins[-1]):
            colors = self.colors[:-1]
            # above_color = self.colors[-1]
            # below_color = self.colors[0]
        else:
            colors = self.colors
            # above_color = self.colors[-1]
            # below_color = self.colors[0]

        cmap = ListedColormap(colors=colors, name=self.name)
        cmap.set_over(self.colors[-1])
        cmap.set_under(self.colors[0])
        return cmap

    @property
    def norm(self) -> BoundaryNorm:
        """Return the boundary-norm associate with this comfort metric.

        Returns:
            BoundaryNorm:
                The boundary norm.
        """
        boundaries = np.array(self.bins)
        if np.isinf(self.bins[0]) and np.isinf(self.bins[-1]):
            boundaries = boundaries[1:-1]
        elif np.isinf(self.bins[0]) and not np.isinf(self.bins[-1]):
            boundaries = boundaries[1:]
        elif not np.isinf(self.bins[0]) and np.isinf(self.bins[-1]):
            boundaries = boundaries[:-1]
        else:
            pass
        return BoundaryNorm(boundaries=boundaries, ncolors=self.cmap.N)

    @property
    def lb_colors(self) -> Tuple[Color]:
        """The ladybug colors.

        Returns:
            Tuple[Color]:
                The ladybug color objects.
        """
        return tuple(
            Color(*i)
            for i in (np.array([to_rgba(color) for color in self.colors]) * 255).astype(
                int
            )
        )

    def get_color(self, value: Union[float, int], as_array: bool = False) -> str:
        """Return the color associated with the categorised value.

        Args:
            value (Union[float, int]):
                The value to get the color for.
            as_array (bool, optional):
                Whether to return the color as an array or a hex string.

        Returns:
            str:
                The color as a hex string.
        """
        if value <= self.bins[0] or value > self.bins[-1]:
            raise ValueError(
                f"The input value/s are outside the range of the categories ({min(self).left} < x <= {max(self).right})."
            )
        color = self.cmap(self.norm(value))
        if not as_array:
            return to_hex(color)
        return color

    def categorise(self, data: Any) -> pd.Categorical:
        """Categorise the data.

        Args:
            data (Any):
                The data to categorise.

        Returns:
            pd.Categorical:
                The categorised data.
        """
        categorical = pd.cut(
            data, self.bins, labels=self.bin_names, include_lowest=True
        )
        if categorical.isna().any():
            raise ValueError(
                f"The input value/s are outside the range of the categories ({self.bins[0]} <= x <= {self.bins[-1]})."
            )
        return categorical

    def value_counts(
        self,
        data: Any,
        density: bool = False,
    ) -> pd.Series:
        """Return a table summary of the categories.

        Args:
            data (Any):
                The data to categorise/bin.
            density (bool, optional):
                Whether to normalise the data to a percentage.

        Returns:
            pd.Series:
                The number of counts within each categorical bin.
        """
        result = self.categorise(data).value_counts().sort_index()
        if density:
            return result / len(data)
        return result

    def timeseries_summary_monthly(
        self, series: pd.Series, density: bool = False
    ) -> pd.DataFrame:
        """Return a table summary of the categories.

        Args:
            series (pd.Series):
                The time-indexed dataset to summarise.
            density (bool, optional):
                Whether to normalise the data to a percentage.

        Returns:
            pd.DataFrame:
                A table summary of the categories over each month
        """
        if not isinstance(series, pd.Series):
            raise ValueError("The series must be a pandas series.")

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("The series must have a time series.")

        counts = (
            (
                self.categorise(series)
                .groupby(series.index.month)
                .value_counts()
                .unstack()
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        counts.index.name = "Month"
        if density:
            return counts.div(counts.sum(axis=1), axis=0)
        return counts

    def summarise(
        self,
        data: Any,
    ) -> str:
        """Return a text summary of the categories across all values in dataset given.

        Args:
            data (Any):
                The dataset to summarise.

        Returns:
            str:
                A text summary of the occurneces within each of the categories.
        """
        result = self.value_counts(data)
        result_density = result / result.sum()

        statements = []
        for desc, (idx, val) in list(zip(*[self.descriptions, result.iteritems()])):
            statements.append(
                f'"{desc}" occurs {val} times ({result_density[idx]:0.1%}*{len(data)}).'
            )
        return "\n".join(statements)

    def create_legend(
        self, ax: plt.Axes = None, verbose: bool = True, **kwargs
    ) -> Legend:
        """Create a legend for this categoical.

        Args:
            ax (plt.Axes, optional):
                The axes to add the legend to.
            verbose (bool, optional):
                Whether to use the verbose descriptions or the interval index.
            **kwargs:
                Additional keyword arguments to pass to the legend.

        Returns:
            Legend:
                The legend.

        """

        if ax is None:
            ax = plt.gca()

        handles = []
        labels = []
        for color, description, iidx in list(
            zip(*[self.colors, self.descriptions, self.interval_index])
        ):
            handles.append(
                patches.Patch(
                    facecolor=color,
                    edgecolor=None,
                )
            )
            labels.append(description if verbose else iidx)
        lgd = ax.legend(handles=handles, labels=labels, **kwargs)
        return lgd
