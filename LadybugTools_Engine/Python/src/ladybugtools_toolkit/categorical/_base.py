from calendar import month_abbr
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Union

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from ladybug.legend import Color
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

from ..ladybug_extension.datacollection import (
    HourlyContinuousCollection,
    collection_from_series,
    collection_to_series,
)


@dataclass(init=True, repr=True, unsafe_hash=True)
class CategoryBase:
    """Base class for categories."""

    low_limit: float = field(init=True, repr=False)
    high_limit: float = field(init=True, repr=False)
    name: str = field(init=True, repr=True)
    color: Union[str, List[float]] = field(init=True, repr=False)

    def __post_init__(self):
        # check that color is a string or a list of floats
        assert isinstance(
            self.color, (str, list)
        ), "The color must be a string or a list of floats."

        # check that the limits are sorted
        assert (
            self.low_limit < self.high_limit
        ), "The low limit must be less than the high limit."

        # convert the color to hex if it is a list of floats
        assert is_color_like(
            self.color
        ), "The color must be a valid color string or a list of floats."
        self.color = to_hex(self.color, keep_alpha=True)

    def __str__(self) -> str:
        """Return the description of the category."""
        return self.name

    @property
    def range(self) -> Tuple[float, float]:
        """Return the limits of the category."""
        return self.low_limit, self.high_limit

    @property
    def mid_point(self) -> float:
        """Return the mid point of the category."""
        return (self.low_limit + self.high_limit) / 2

    @property
    def description(self) -> str:
        """Return the description of the category."""
        if np.isinf(self.low_limit):
            return f"{self.name} (≤{self.high_limit})"

        if np.isinf(self.high_limit):
            return f"{self.name} (>{self.low_limit})"

        return f"{self.name} ({self.low_limit}<x≤{self.high_limit})"

    def lb_color(self) -> Color:
        """Return the ladybug color of the category."""
        r, g, b, a = np.array(to_rgba(self.color)) * 255
        return Color(r=r, g=g, b=b, a=a)


@dataclass(init=True, repr=True)
class CategoriesBase:
    """Base class for categorical binning of data.

    Args:
        categories List[Category]:
            A list of categories.
        below Category (optional):
            A category to use for values below the first category.
        above Category (optional):
            A category to use for values above the last category.
    """

    categories: List[CategoryBase] = field(init=True, repr=True)
    below: CategoryBase = field(init=True, repr=False, default=None)
    above: CategoryBase = field(init=True, repr=False, default=None)

    def __post_init__(self):
        """Validation checks."""

        # check that the categories are sorted
        for i, category in enumerate(self.categories):
            if i == 0:
                continue
            assert (
                category.low_limit > self.categories[i - 1].low_limit
            ), "The categories must be sorted by increasing low limit."

        # check that the categories are adjacent
        for i, category in enumerate(self.categories):
            if i == 0:
                continue
            assert (
                category.low_limit == self.categories[i - 1].high_limit
            ), "Adjacent categories must share an edge."

        # check that if categories is only 1-long, below and above are not None
        if len(self.categories) == 1:
            assert (
                self.below is not None
            ), "If there is only one category, below must be supplied."
            assert (
                self.above is not None
            ), "If there is only one category, above must be supplied."

        if self.below is not None:
            # check "below" category low-limit is -inf
            if self.below.low_limit != -np.inf:
                raise ValueError(
                    'The lower limit of the "below" category must be -inf if a "below" category is supplied.'
                )
            # check that below category upper limit is less than or equal to the first category lower limit
            if self.below.high_limit != self.categories[0].low_limit:
                raise ValueError(
                    "The upper limit of the 'below' category must be equal to the lower limit of the first category."
                )

        if self.above is not None:
            # check "above" category high-limit is inf
            if self.above.high_limit != np.inf:
                raise ValueError(
                    'The upper limit of the "above" category must be inf if an "above" category is supplied.'
                )
            # check that above category lower limit is greater than or equal to the last category upper limit
            if self.above.low_limit != self.categories[-1].high_limit:
                raise ValueError(
                    "The lower limit of the 'above' category must be equal to the upper limit of the last category."
                )

    @classmethod
    def from_bins(
        cls,
        names: List[str],
        bins: List[float],
        colors: List[str],
        left_closed: bool = False,
        right_closed: bool = False,
    ) -> "CategoriesBase":
        """Create a Categorical object from a list of bins.

        Args:
            names List[str]:
                A list of names for the categories.
            bins List[float]:
                A list of bin edges. Should be sorted, unique and one longer than names.
            colors List[str]:
                A list of colors for the categories. Should be the same length as names.
            left_closed bool (optional):
                If True, the first category is considered the "below" category.
            right_closed bool (optional):
                If True, the last category is considered the "above" category.
        """

        # check that there is at least 2-values in bins
        assert (
            len(bins) >= 2
        ), "There must be at least two values when defining a single bin."

        # check that if bins is only 2-long, left_closed and right_closed are both False
        if len(bins) == 2:
            assert (
                not left_closed
            ), "If there is only one bin (defined by two values), left_closed must be False."
            assert (
                not right_closed
            ), "If there is only one bin (defined by two values), right_closed must be False."

        if (left_closed or right_closed) and len(names) == 2:
            raise ValueError("Cannot have a closed end with only two categories.")

        # check input lengths align
        assert (
            len(names) == len(bins) - 1
        ), "The length of names must be one less than the length of bins."
        assert len(names) == len(
            colors
        ), "The length of names must equal the length of colors."

        # check input bin edges are all sorted
        assert np.all(np.diff(bins) > 0), "The bins must be sorted."

        # check input bin edges are all unique
        assert len(np.unique(bins)) == len(bins), "The bins must be unique."

        # create object
        categories = []
        for i, name in enumerate(names):
            categories.append(
                CategoryBase(
                    name=name,
                    low_limit=bins[i],
                    high_limit=bins[i + 1],
                    color=colors[i],
                )
            )

        # return the generic case, without above/below categories
        if not left_closed and not right_closed:
            return cls(categories=categories)

        # return the case with a below category only
        if left_closed and not right_closed:
            return cls(categories=categories[1:], below=categories[0])

        # return the case with a below category only
        if right_closed and not left_closed:
            return cls(categories=categories[:-1], below=categories[-1])

        # return the case with both above and below categories
        return cls(
            categories=categories[1:-1],
            below=categories[0],
            above=categories[-1],
        )

    def categories_with_limits(self) -> List[CategoryBase]:
        """Return the categories, including any upper/lower limits where included."""
        _categories = self.categories
        if self.below is not None:
            _categories = [self.below] + _categories
        if self.above is not None:
            _categories = _categories + [self.above]
        return _categories

    @property
    def colors(self) -> List[Any]:
        """Return the colors of the categories, including any upper/lower limits where included."""
        return [i.color for i in self.categories_with_limits()]

    def _bin_edges(self) -> List[float]:
        """Return the limits of the categories."""
        return np.unique(
            np.array([i.range for i in self.categories_with_limits()]).flatten()
        ).tolist()

    @property
    def names(self) -> List[str]:
        """Return the names of the categories."""
        return [i.name for i in self.categories_with_limits()]

    @property
    def descriptions(self) -> List[str]:
        """Return the descriptions of the categories."""
        return [i.description for i in self.categories_with_limits()]

    @property
    def cmap(self) -> Colormap:
        """Return the colormap associated with this categorical."""
        cmap = ListedColormap(colors=[i.color for i in self.categories])
        if self.below is not None:
            cmap.set_under(self.below.color)
        if self.above is not None:
            cmap.set_over(self.above.color)

        return cmap

    def min(self) -> float:
        """Return the minimum value of the categories."""
        return self._bin_edges()[0]

    def max(self) -> float:
        """Return the maximum value of the categories."""
        return self._bin_edges()[-1]

    @property
    def boundarynorm(self) -> BoundaryNorm:
        """Return the boundary-norm associate with this comfort metric."""
        return BoundaryNorm(self._bin_edges()[1:-1], self.cmap.N)

    def categorise(self, value: Any) -> pd.Categorical:
        """Return the category that the value falls into."""
        if isinstance(value, (int, float)):
            return pd.cut(
                np.atleast_1d(value),
                bins=self._bin_edges(),
                labels=self.names,  # could be made a category object, but easier to use strings
                include_lowest=True,
                right=True,
            )[0]
        if isinstance(value, pd.Series):
            return pd.cut(
                value,
                bins=self._bin_edges(),
                labels=self.names,
                include_lowest=True,
                right=True,
            )
        if isinstance(value, pd.DataFrame):
            return value.apply(
                pd.cut,
                bins=self._bin_edges(),
                labels=self.names,
                include_lowest=True,
                right=True,
            )
        if isinstance(value, HourlyContinuousCollection):
            d = collection_to_series(value)
            dd = pd.cut(
                d,
                bins=self._bin_edges(),
                labels=self.names,
                include_lowest=True,
                right=True,
            )
            dd.name = "Categorical (unitless)"
            return collection_from_series(dd)
        if isinstance(value, (np.ndarray, tuple, list)):
            categories = np.empty_like(value, dtype=object)
            value = np.atleast_1d(value)
            # check that value dtype is not object
            if value.dtype == object:
                raise ValueError(
                    "Categorisation can only be applied to non-ragged arrays"
                )
            for category in self.categories:
                categories[
                    (category.low_limit <= value) & (value < category.high_limit)
                ] = category.name
            return categories
        raise TypeError(
            f"The value {value} is not a valid type. It must be a number, a pandas Series, DataFrame, or a numpy array."
        )

    def timeseries_summary_categorical(
        self,
        value: Union[pd.Series, HourlyContinuousCollection],
        mask: List[bool] = None,
    ) -> pd.Categorical:
        """Return a table summary of the categories.

        Args:
            value Union[pd.Series, HourlyContinuousCollection]:
                The time-indexed dataset to summarise.
            mask List[bool] (optional):
                A mask to apply to the data before summarising.

        Returns:
            pd.Categorical:
                A categorical series with the same index as the input.
        """

        if isinstance(value, HourlyContinuousCollection):
            assert (
                value.is_continuous and len(value) >= 8760
            ), "The collection given is not continuous, or annual-hourly"
            series = collection_to_series(value)
        elif isinstance(value, pd.Series):
            assert isinstance(
                value.index, pd.DatetimeIndex
            ), "The index must be a DatetimeIndex"
            series = value
        else:
            raise TypeError(f"{type(value)} cannot be passed to this function.")

        if mask is not None:
            assert len(mask) == len(
                series
            ), "The mask must be the same length as the data passed"
            series = series[mask]
        else:
            mask = [True] * len(series)

        categorical = self.categorise(series)
        return categorical

    def timeseries_summary_valuecounts(
        self,
        value: Union[pd.Series, HourlyContinuousCollection],
        mask: List[bool] = None,
    ) -> pd.Series:
        """Return a table summary of the categories.

        Args:
            value Union[pd.Series, HourlyContinuousCollection]:
                The time-indexed dataset to summarise.
            mask List[bool] (optional):
                A mask to apply to the data before summarising.

        Returns:
            pd.Series:
                A series value counts of the categories.
        """
        categorical = self.timeseries_summary_categorical(value, mask=mask)
        return categorical.value_counts()

    def timeseries_summary_text(
        self,
        value: Union[pd.Series, HourlyContinuousCollection],
        mask: List[bool] = None,
        mask_name: str = None,
    ) -> str:
        """Return a text summary of the categories.

        Args:
            value Union[pd.Series, HourlyContinuousCollection]:
                The time-indexed dataset to summarise.
            mask List[bool] (optional):
                A mask to apply to the data before summarising.
            mask_name str (optional):
                A name to give the mask in the text summary.

        Returns:
            str:
                A text summary of the categories.
        """
        categorical = self.timeseries_summary_categorical(value, mask=mask)
        total_number_of_hours = len(categorical)

        statements = []
        if mask_name is not None:
            statements.append(f"For {mask_name}, over {total_number_of_hours} hours: ")
        for idx, val in categorical.value_counts().iteritems():
            statements.append(
                f'The "{idx}" category is achieved for {val} hours ({val/total_number_of_hours:0.1%}).'
            )
        return "\n".join(statements)

    def timeseries_summary_monthly(
        self,
        value: Union[pd.Series, HourlyContinuousCollection],
        mask: List[bool] = None,
        mask_name: str = None,
        density: bool = False,
    ) -> pd.DataFrame:
        """Return a table summary of the categories.

        Args:
            value Union[pd.Series, HourlyContinuousCollection]:
                The time-indexed dataset to summarise.
            mask List[bool] (optional):
                A mask to apply to the data before summarising.
            mask_name str (optional):
                A name to give the mask in the summary.

        Returns:
            pd.DataFrame:
                A table summary of the categories over each month
        """

        categorical = self.timeseries_summary_categorical(value, mask=mask)
        d = categorical.groupby(categorical.index.month).value_counts().unstack()
        d.index = [month_abbr[i] for i in d.index]
        d.index.name = mask_name
        if density:
            d = d.div(d.sum(axis=1), axis=0)
        return d

    def create_legend(self, ax: plt.Axes = None, **kwargs) -> Legend:
        """Create a legend for the categories."""

        if ax is None:
            ax = plt.gca()

        include_values = kwargs.pop("include_values", False)

        handles = []
        labels = []
        for cat in self.categories_with_limits():
            handles.append(
                mpatches.Patch(
                    facecolor=cat.color,
                    edgecolor=None,
                )
            )
            labels.append(cat.description if include_values else cat.name)
        return ax.legend(handles=handles, labels=labels, **kwargs)
