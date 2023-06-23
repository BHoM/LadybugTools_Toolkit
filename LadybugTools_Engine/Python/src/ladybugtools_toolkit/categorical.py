from calendar import month_abbr
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from ladybug.legend import Color
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    ListedColormap,
    is_color_like,
    to_hex,
    to_rgba,
)

from .ladybug_extension.analysis_period import AnalysisPeriod, describe_analysis_period
from .ladybug_extension.datacollection import (
    HourlyContinuousCollection,
    collection_from_series,
    collection_to_series,
)


@dataclass(init=True, repr=True, unsafe_hash=True)
class Category:
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
            return f"{self.name} (<{self.high_limit}°C UTCI)"

        if np.isinf(self.high_limit):
            return f"{self.name} (>{self.low_limit}°C UTCI)"

        return f"{self.name} ({self.low_limit}°C < x < {self.high_limit}°C UTCI)"

    def lb_color(self) -> Color:
        """Return the ladybug color of the category."""
        r, g, b, a = np.array(to_rgba(self.color)) * 255
        return Color(r=r, g=g, b=b, a=a)


@dataclass(init=True, repr=True)
class Categories:
    """ "Base class for categorical binning of data."""

    categories: List[Category] = field(init=True, repr=True)

    def __post_init__(self):
        # check that the categories are sorted
        for i, category in enumerate(self.categories):
            if i == 0:
                continue
            assert (
                category.low_limit > self.categories[i - 1].low_limit
            ), "The categories must be sorted by increasing low limit."

    @classmethod
    def from_bins(
        cls, names: List[str], bins: List[float], colors: List[str]
    ) -> "Categories":
        """Create a Categorical object from a list of bins."""
        assert len(names) >= 2, "There must be at least two categories."
        assert (
            len(names) == len(bins) - 1
        ), "The length of names must be one less than the length of bins."
        assert np.all(np.diff(bins) > 0), "The bins must be sorted."
        assert len(np.unique(bins)) == len(bins), "The bins must be unique."
        assert len(names) == len(
            colors
        ), "The length of names must equal the length of colors."

        categories = []
        for i, name in enumerate(names):
            categories.append(
                Category(
                    name=name,
                    low_limit=bins[i],
                    high_limit=bins[i + 1],
                    color=colors[i],
                )
            )
        return cls(categories=categories)

    @property
    def colors(self) -> List[Any]:
        """Return the colors of the categories."""
        return [i.color for i in self.categories]

    def _bin_edges(self) -> List[float]:
        """Return the limits of the categories."""
        return np.unique(
            np.array([i.range for i in self.categories]).flatten()
        ).tolist()

    @property
    def names(self) -> List[str]:
        """Return the names of the categories."""
        return [i.name for i in self.categories]

    @property
    def cmap(self) -> Colormap:
        """Return the colormap associated with this categorical."""
        cmap = ListedColormap(
            colors=self.colors,
            # name="Categorical",
        )
        cmap.set_under(self.categories[0].color)
        cmap.set_over(self.categories[-1].color)

        return cmap

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
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
    ) -> str:
        """Return a table summary of the categories."""

        if isinstance(value, HourlyContinuousCollection):
            assert (
                value.is_continuous and len(value) >= 8760
            ), "The collection given is not continuous, or annual-hourly"
            series = collection_to_series(
                value.filter_by_analysis_period(analysis_period)
            )
        elif isinstance(value, pd.Series):
            col = collection_from_series(value).filter_by_analysis_period(
                analysis_period
            )
            series = series = collection_to_series(col)
        else:
            raise TypeError(f"{type(value)} cannot be passed to this function.")

        categorical = self.categorise(series)
        return categorical

    def timeseries_summary_series(
        self,
        value: Union[pd.Series, HourlyContinuousCollection],
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
    ) -> pd.Series:
        """Return a table summary of the categories."""
        categorical = self.timeseries_summary_categorical(
            value, analysis_period=analysis_period
        )
        return categorical.value_counts()

    def timeseries_summary_text(
        self,
        value: Union[pd.Series, HourlyContinuousCollection],
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
    ) -> str:
        """Return a text summary of the categories."""
        categorical = self.timeseries_summary_categorical(
            value, analysis_period=analysis_period
        )
        total_number_of_hours = len(categorical)
        statements = [
            f"For {describe_analysis_period(analysis_period, include_timestep=False)}, accounting for {total_number_of_hours} hours"
        ]
        for idx, val in categorical.value_counts().iteritems():
            statements.append(
                f'The "{idx}" category is achieved for {val} hours ({val/total_number_of_hours:0.1%}).'
            )
        return "\n".join(statements)

    def timeseries_summary_monthly(
        self,
        value: Union[pd.Series, HourlyContinuousCollection],
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        density: bool = False,
    ) -> pd.DataFrame:
        """Return a table summary of the categories."""

        categorical = self.timeseries_summary_categorical(
            value, analysis_period=analysis_period
        )
        d = categorical.groupby(categorical.index.month).value_counts().unstack()
        d.index = [month_abbr[i] for i in d.index]
        d.index.name = analysis_period
        if density:
            d = d.div(d.sum(axis=1), axis=0)
        return d
