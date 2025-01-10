"""Methods to help with the handling of different ladybug data types used thrgouhout BHoM workflows"""

# TODO - make color-like datatype more flexible and usable (no repeating the str Z tuple Z float Z int Z etc. for each method)
# TODO - absratc inheritance!

import abc
import calendar
import inspect
import textwrap
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from honeybee.config import folders as hb_folders
from ladybug.datatype import TYPESDICT
from ladybug.datatype.base import DataTypeBase
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybug.epw import EPW
from ladybugtools_toolkit.categorical.categories import (
    ACTUAL_SENSATION_VOTE_CATEGORIES,
    APPARENT_TEMPERATURE_CATEGORIES,
    BEAUFORT_CATEGORIES,
    CLO_VALUE_CATEGORIES,
    DISCOMFORT_INDEX_CATEGORIES,
    HEAT_INDEX_CATEGORIES,
    HUMIDEX_CATEGORIES,
    METABOLIC_RATE_CATEGORIES,
    PET_CATEGORIES,
    SET_CATEGORIES,
    THERMAL_SENSATION_CATEGORIES,
    UTCI_DEFAULT_CATEGORIES,
    WBGT_CATEGORIES,
    WIND_CHILL_CATEGORIES,
    Categorical,
)
from ladybugtools_toolkit.helpers import (
    CONSOLE_LOGGER,
    average_color,
    color_to_format,
    sanitise_string,
)
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
    hex2color,
    rgb2hex,
    to_hex,
    to_rgb,
    to_rgba,
)
from matplotlib.legend import Legend

TYPESDICT: dict[str, DataTypeBase]


@dataclass
class DtypeConfig(abc.ABC):
    """Base class for configuring a datatype and its associated properties.

    Attributes:
        name (str):
            The full name of the datatype.
        abbreviation (str):
            The abbreviation of the datatype.
        unit (str):
            The unit of the datatype.
        boundaries (list[float]):
            The boundaries of the datatype.
            For continuous datatypes, this should be a list of two values.
            For categorical datatypes, this should be a list of n+1 values, where n is the number
            of categories and each rolling window is the boundary of each category.
    """

    name: str
    abbreviation: str
    unit: str
    boundaries: list[float]

    # def __repr__(self):
    #     """Get the string representation of this object."""
    #     return f"{self.__class__.__name__}({self.name})"

    def __post_init__(self): ...

    def min(self) -> float:
        """Get the minimum value this variable can be."""
        return min(self.boundaries)

    def max(self) -> float:
        """Get the maximum value this variable can be."""
        return max(self.boundaries)

    def boundary_pairs(self) -> list[tuple[float, float]]:
        return np.lib.stride_tricks.sliding_window_view(self.boundaries, 2).tolist()

    @abc.abstractmethod
    def norm(self) -> BoundaryNorm: ...

    @abc.abstractmethod
    def get_colors(self) -> list[str] | list[int] | list[float]: ...

    def get_average_color(self, return_type: str = "hex") -> str | list[int] | list[float]:
        """Return the average color of this variable."""
        return color_to_format(average_color(self.get_colors()), return_type=return_type)

    @abc.abstractmethod
    def get_color(self) -> str | list[int] | list[float]: ...

    @abc.abstractmethod
    def get_colormap(self) -> Colormap: ...

    @classmethod
    @abc.abstractmethod
    def from_lb_dtype(cls) -> "DtypeConfig": ...

    @abc.abstractmethod
    def plotly_heatmap(self, data: pd.Series) -> go.Figure: ...

    def to_unit(self, data: Any, unit: str) -> "CategoricalDtype":
        """Convert this object to a different unit.

        Args:
            data (Any):
                The data to convert.
            unit (str):
                The unit to convert this object to.

        Returns:
            CategoricalDtype:
                The object with the new unit.
        """
        # TODO - implement unit conversions between datatypes here, including logic for invalid conversions.
        raise NotImplementedError("This method has not been implemented yet.")


@dataclass
class CategoricalDtype(DtypeConfig):
    """Class for configuring a categorical datatype and its associated properties.

    The number of colors and categories must match, and be equal to the number of boundaries - 1.

    Attributes:
        colors (list[str | list[int] | list[float]]):
            The colors of the datatype.
        categories (list[str]):
            The categories of the datatype.
    """

    colors: list[str | list[int] | list[float]]
    categories: list[str]

    def __post_init__(self):
        super().__post_init__()

        # ensure the number of colors and categories match
        if len(self.colors) != len(self.categories):
            raise ValueError(
                f"The number of colors ({len(self.colors)}) must be equal to the number of categories ({len(self.categories)})."
            )

        # ensure the number of boundaries is one more than the number of colors
        if len(self.boundaries) != len(self.colors) + 1:
            raise ValueError(
                f"The number of boundaries ({len(self.boundaries)}) must be one more than the number of colors ({len(self.colors)})."
            )

        # ensure boundaries are in increasing order
        if not all(
            self.boundaries[i] < self.boundaries[i + 1] for i in range(len(self.boundaries) - 1)
        ):
            raise ValueError("Boundaries must be in increasing order.")

    @classmethod
    def from_lb_dtype(
        cls,
        dtype: DataTypeBase,
        boundaries: list[float],
        colors: list[str | list[int] | list[float]],
        categories: list[str],
    ) -> "CategoricalDtype":
        """Create this object from a Ladybug datatype.

        Args:
            dtype (DataTypeBase):
                The Ladybug datatype to create this object from.
            boundaries (list[float]):
                The boundaries of the datatype.
            colors (list[str | list[int] | list[float]]):
                The colors of the datatype.
            categories (list[str]):
                The categories of the datatype.

        Returns:
            CategoricalDtype:
                The categorical datatype object.
        """

        if not isinstance(dtype, DataTypeBase):
            raise ValueError("dtype must be a Ladybug datatype.")

        if min(boundaries) != dtype.min:
            raise ValueError(f"Boundaries must start at {dtype.min}.")
        if max(boundaries) != dtype.max:
            raise ValueError(f"Boundaries must end at {dtype.max}.")

        obj = cls(
            name=dtype.name,
            abbreviation=dtype.abbreviation,
            unit=dtype.si_units[0],
            boundaries=boundaries,
            colors=colors,
            categories=categories,
        )
        obj.lb_dtype = dtype
        return obj

    def norm(self) -> BoundaryNorm:
        return BoundaryNorm(boundaries=self.boundaries, ncolors=len(self.colors), clip=True)

    def get_colors(self, return_type: str = "hex") -> list[str]:
        """Get the colors of this variable.

        Args:
            return_type (str, optional):
                The type of color to return. Default is "hex".

        Returns:
            list[str]:
                The colors of the variable.
        """
        return [color_to_format(i, return_type) for i in self.colors]

    def get_colormap(self) -> ListedColormap:
        return ListedColormap(colors=self.get_colors(), name=self.name)

    def get_categories(
        self, include_ranges: bool = False, include_units: bool = False
    ) -> list[str]:
        """Return the categories of this data type, with or without the ranges and units.

        Args:
            include_ranges (bool, optional):
                Include the ranges of the categories. Default is False.
            include_units (bool, optional):
                Include the units of the categories. Default is False.

        Returns:
            list[str]:
                The categories of this data
        """
        categories = []
        for category, (bnd_low, bnd_high) in zip(
            *[self.categories, np.lib.stride_tricks.sliding_window_view(self.boundaries, 2)]
        ):
            temp = category
            if include_ranges:
                if include_units:
                    temp += f" ({bnd_low}{self.unit} to {bnd_high}{self.unit})"
                else:
                    temp += f" ({bnd_low} to {bnd_high})"
            categories.append(temp)
        return categories

    def get_legend_handles_labels(
        self, include_ranges: bool = False, include_units: bool = False
    ) -> tuple[list[patches.Patch], list[str]]:
        """Create the handles and labels that can be used to create a legend object, from this object.

        Args:
            include_ranges (bool, optional):
                Include the ranges of the categories. Default is False.
            include_units (bool, optional):
                Include the units of the categories. Default is False.
        Returns:
            tuple[list[patches.Patch], list[str]]:
                A tuple containing the handles and labels for the legend.
        """

        handles = []
        for color in self.get_colors(return_type="hex"):
            handles.append(
                patches.Patch(
                    facecolor=color,
                    edgecolor=None,
                )
            )
        return handles, self.get_categories(
            include_ranges=include_ranges, include_units=include_units
        )

    def get_legend(
        self, include_ranges: bool = False, include_units: bool = False, **kwargs
    ) -> Legend:
        """Get the legend for this variable.

        Args:
            include_ranges (bool, optional):
                Include the ranges of the categories. Default is False.
            include_units (bool, optional):
                Include the units of the categories. Default is False.
            **kwargs:
                Additional keyword arguments to pass to the legend.

        Returns:
            Legend:
                The legend for this variable.
        """

        handles, labels = self.get_legend_handles_labels(
            include_ranges=include_ranges, include_units=include_units
        )

        return plt.legend(
            handles=handles,
            labels=labels,
            **kwargs,
        )

    def get_category_index(self, value: float) -> int:
        """Return the index of the value within the categories of the variable.

        Args:
            value (float):
                The value of the variable.

        Returns:
            int:
                The index of the value within the categories of the variable.
        """
        if value < self.min():
            return 0

        if value >= self.max():
            return -1

        for n, (low, high) in enumerate(
            np.lib.stride_tricks.sliding_window_view(self.boundaries, 2)
        ):
            if low <= value < high:
                return n

        raise ValueError(
            f"Value is not within the boundaries of {self} ({self.min} to {self.max})."
        )

    def get_category(
        self, value: float, include_ranges: bool = False, include_units: bool = False
    ) -> str:
        """Get the category of this variable at the given value.

        Args:
            value (float):
                The value of the variable.
            include_ranges (bool, optional):
                Include the ranges of the categories. Default is False.
            include_units (bool, optional):
                Include the units of the categories. Default is False.

        Returns:
            str:
                The category of the variable at the given value.
        """
        categories = self.get_categories(include_ranges=include_ranges, include_units=include_units)
        return categories[self.get_category_index(value=value)]

    def get_color(
        self, value: float, interpolate_color: bool = False, return_type: str = "hex"
    ) -> str | list[int] | list[float]:
        """Get the color of this variable at a given value.

        Args:
            value (float):
                The value of the variable.
            interpolate_color (bool, optional):
                Interpolate the color for a categorical variable.
                Default is False.
            return_type (str, optional):
                The type of color to return.

        Returns:
            str | list[int] | list[float]:
                The color of the variable at the given value.
        """

        idx = self.get_category_index(value=value)
        cmap = self.get_colormap()

        if not interpolate_color:
            return color_to_format(cmap.colors[idx], return_type)

        if interpolate_color:
            if idx == -1:
                idx = len(self.categories) - 1
            # if index is 0 or -1, get the first/last color
            if idx == 0 or idx == len(self.categories) - 1:
                color = cmap.colors[idx]
            else:
                # get the color above and below the value
                color_below = cmap.colors[idx]
                color_above = cmap.colors[idx + 1]
                # create a temporary colormap with just the two colors
                temp_cmap = LinearSegmentedColormap.from_list(
                    colors=[color_below, color_above], name="temp"
                )
                # get the value above and below the value
                val_below = self.norm.boundaries[idx]
                val_above = self.norm.boundaries[idx + 1]
                # interpolate the color
                color = temp_cmap(np.interp(value, [val_below, val_above], [0, 1]))
            return color_to_format(color, return_type)

        # get the index of the value
        idx = self.norm(value)
        # get the color
        color = self.colors[idx]
        return color_to_format(color, "hex")

    def _plotly_colorscale(self) -> list[tuple[float, str]]:
        """Get the Plotly colorscale for this variable, and the associated categorical colormap."""
        colors = self.get_colors()
        plotly_colors = []
        for color, (low, high) in zip(
            *[
                colors,
                np.lib.stride_tricks.sliding_window_view(
                    np.linspace(0, 1, len(colors) + 1).round(4), 2
                ),
            ]
        ):
            plotly_colors.append([float(low), color])
            plotly_colors.append([float(high), color])
        return plotly_colors

    def plotly_heatmap(self, data: pd.Series) -> go.Figure:
        """Create a Plotly heatmap for this variable."""

        # ensure the data is a series
        if not isinstance(data, pd.Series):
            raise ValueError("Data must be a pandas Series.")

        # ensure data has a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index.")

        # ensure data values are numeric
        if not pd.api.types.is_numeric_dtype(data):
            raise ValueError("Data values must be numeric.")

        # get the values, categories, and index mappings for the data
        categories = data.map(self.get_category).values
        indices = data.map(self.get_category_index).values
        plotly_colors = self._plotly_colorscale()

        # generate the heatmap trace
        heatmap = go.Heatmap(
            x=data.index.date,
            y=data.index.time,
            z=indices,
            colorscale=plotly_colors,
            zmin=0,
            zmax=len(self.categories) - 1,
            customdata=np.stack((categories, data.values), axis=-1),
            hovertemplate=(
                "<b>Date:</b> %{x|%b %d}<br>"
                + "<b>Time:</b> %{y}<br>"
                + "<b>Category:</b> %{customdata[0]}<br>"
                + f"<b>Value:</b> %{{customdata[1]}}{self.unit}<br>"
            ),
            name="",
            colorbar=dict(title=self.unit),
        )

        # plot the figure
        fig = go.Figure(data=[heatmap])

        # # add gridlines
        # lines = {"a": 500, "c": 700, "a": 900, "b": 950}
        # for xx in data.index.date[
        #     (data.index.day == 1)
        #     & (data.index.hour == 0)
        #     & (data.index.minute == 0)
        #     & (data.index.second == 0)
        # ]:
        #     fig.add_shape(
        #         type="line",
        #         x0=xx - timedelta(hours=12),
        #         x1=xx - timedelta(hours=12),
        #         y0=data.index.time.min(),  # - timedelta(minutes=30),
        #         y1=data.index.time.max(),  # + timedelta(minutes=30),
        #         line=dict(color="black", width=1),
        #     )

        # customise the hovertemplate
        fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
            )
        )

        # format the colorbar
        fig["data"][0]["colorbar"] = dict(
            title=dict(
                text=self.abbreviation,
                font=dict(weight="bold"),
            ),
            titleside="top",
            tickmode="array",
            tickvals=np.linspace(0.5, len(self.categories) - 1.5, len(self.categories)),
            ticktext=[
                textwrap.fill(i, 23).replace("\n", "<br>")
                for i in self.get_categories(include_ranges=False, include_units=False)
            ],
        )

        # format x-axis tick label placement, dependent on zoom
        fig.update_layout(
            xaxis_tickformatstops=[
                dict(dtickrange=[86400000, "M1"], value="%b %e"),  # <1 month e.g. "Jan 10"
                dict(dtickrange=["M1", "M12"], value="%b"),  # 1-12 months e.g. "Jan"
                dict(dtickrange=["M12", None], value="%b %Y"),  # >12 months e.g. "Jan 2017"
            ],
            xaxis_ticklabelmode="instant",  # do not centre the tick label
            xaxis_ticklabelposition="outside left",  # tick at start of time period
        )

        # format y-axis tick label placement, dependent on zoom
        fig.update_layout(
            yaxis_ticklabelmode="instant",  # do not centre the tick label
            yaxis_ticklabelposition="outside left",  # tick at start of time period
        )

        # # add range slider and filter buttons
        # fig.update_xaxes(
        #     rangeslider_visible=True,
        #     rangeselector=dict(
        #         buttons=list(
        #             [
        #                 dict(label="Week", step="day", count=7),
        #                 dict(label="Month", step="month", count=1),
        #                 dict(label="All", step="all"),
        #             ]
        #         )
        #     ),
        # )

        # format y-axis
        fig.update_yaxes(title=dict(text="Time of day", font=dict(weight="bold")))

        # add title to plot
        fig.update_layout(
            title=dict(text=f"{self.name}<br>{data.name}", font=dict(weight="bold")),
        )

        # config options - used where fig.show is used
        config = {
            "modeBarButtonsToRemove": ["zoom", "pan", "select"],
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",  # one of png, svg, jpeg, webp
                "filename": sanitise_string(f"{self.abbreviation}_{data.name}"),
                "height": 500,
                "width": 1200,
                "scale": 3,  # Multiply title/legend/axis/canvas sizes by this factor
            },
        }

        return fig, config


# @dataclass
# class ContinuousDtype(DtypeConfig):
#     """Class for configuring a continuous datatype and its associated properties."""

#     colormap: str | Colormap

#     def __repr__(self) -> str:
#         return super().__repr__()

#     def __post_init__(self):
#         super().__post_init__()
#         if isinstance(self.colormap, str):
#             self.colormap = plt.get_cmap(self.colormap)
#         if not isinstance(self.colormap, Colormap):
#             raise ValueError("colormap if not an acceptable dtype (str | Colormap).")
#         self.colormap.name = self.name
#         if len(self.boundaries) != 2:
#             raise ValueError("boundaries must have two values.")
#         if self.boundaries[0] >= self.boundaries[1]:
#             raise ValueError("boundaries must be in increasing order.")

#     @property
#     def min(self) -> float:
#         return self.boundaries[0]

#     @property
#     def max(self) -> float:
#         return self.boundaries[-1]

#     @property
#     def norm(self) -> Normalize:
#         return Normalize(vmin=self.min, vmax=self.max, clip=True)

#     def get_colormap(self) -> Colormap:
#         return self.colormap

#     def get_colors(
#         self, return_type: str = "hex", intervals: int = 11
#     ) -> list[str] | list[tuple[int, int, int, int]] | list[tuple[float, float, float, float]]:
#         """Get the colors of this variable.

#         Args:
#             return_type (str, optional):
#                 The type of color to return. Default is "hex".
#             intervals (int, optional):
#                 The number of intervals to use for the colors. Default is 11.

#         Returns:
#             list[str]:
#                 The colors of the variable.
#         """
#         cmap = self.get_colormap()
#         return [
#             color_to_format(color=cmap(i), return_type=return_type)
#             for i in np.linspace(0, 1, intervals)
#         ]

#     def get_color(
#         self, value: float, return_type: str = "hex"
#     ) -> str | list[tuple[int, int, int, int]] | list[tuple[float, float, float, float]]:
#         """Get the color of this variable at a given value.

#         Args:
#             value (float):
#                 The value of the variable.
#             return_type (str, optional):
#                 The type of color to return.

#         Returns:
#             str | list[tuple[int, int, int, int]] | list[tuple[float, float, float, float]]:
#                 The color of the variable at the given value.
#         """

#         color = self.get_colormap()(self.norm(value))
#         return color_to_format(color, return_type)

#     def get_colorbar(self, **kwargs) -> plt.Axes:
#         """Get the colorbar for this variable."""

#         return plt.colorbar(
#             mappable=plt.cm.ScalarMappable(norm=self.norm, cmap=self.colormap),
#             **kwargs,
#         )

#     def plotly_time_indexed_heatmap(self, data: pd.Series) -> go.Figure:
#         """Create a Plotly heatmap for this variable."""

#         # ensure the data is a series
#         if not isinstance(data, pd.Series):
#             raise ValueError("Data must be a pandas Series.")

#         # ensure data has a datetime index
#         if not isinstance(data.index, pd.DatetimeIndex):
#             raise ValueError("Data must have a datetime index.")

#         # ensure data values are numeric
#         if not pd.api.types.is_numeric_dtype(data):
#             raise ValueError("Data values must be numeric.")

#         fig = go.Figure(
#             data=go.Heatmap(
#                 y=data.index.time,
#                 x=data.index.date,
#                 z=data.values,
#                 zmin=np.nan_to_num(self.norm.vmin),
#                 zmax=np.nan_to_num(self.norm.vmax),
#                 colorscale=self.get_colors(return_type="plotly", intervals=101),
#                 connectgaps=False,
#                 name=data.name,
#                 colorbar=dict(
#                     title=self.unit,
#                 ),
#             )
#         )
#         fig.update_layout(
#             template="plotly_white",
#             title=str(self),
#         )
#         fig.update_yaxes(title_text="Hour")
#         fig.update_xaxes(title_text="Day")
#         return fig


# @dataclass
# class EnvironmentVariable:
#     """A class containing colour information for an environmental variable.

#     Attributes:
#         name (str):
#             The full name of the environment variable.
#         unit (str):
#             The unit of the environment variable.
#         abbreviation (str):
#             The abbreviation of the environment variable.
#         colormap (Colormap | str, Optional):
#             The colormap to use for the variable.
#         norm (Normalize, Optional):
#             The normalization to use for the variables colormap.
#             If categories are provided, this should be a BoundaryNorm.
#         categories (list[str], Optional):
#             The categories for the variable, if it is categorical.
#     """

#     name: str
#     unit: str
#     abbreviation: str = None
#     colormap: Colormap | str = None
#     norm: Normalize = None
#     categories: list[str] = None

#     def __repr__(self):
#         return f"EnvironmentVariable({self.name})"

#     def __post_init__(self):
#         if self.colormap is None:
#             self.colormap = plt.get_cmap("viridis")
#         elif isinstance(self.colormap, str):
#             self.colormap = plt.get_cmap(self.colormap)
#         if not isinstance(self.colormap, Colormap):
#             raise ValueError("colormap must be a string or a matplotlib colormap.")

#         # override colormaps name with the name of the variable
#         self.colormap.name = self.name

#         # set the abbreviation to the name if not provided
#         if self.abbreviation is None:
#             self.abbreviation = self.name

#         # set the normalization to the default if not provided
#         if self.norm is None:
#             self.norm = Normalize(vmin=-np.inf, vmax=np.inf, clip=True)

#         # check if categories are provided, and validate if they are
#         if self.categories is not None:
#             if not isinstance(self.norm, BoundaryNorm):
#                 raise ValueError("When providing categories, norm must be a BoundaryNorm.")
#             else:
#                 if len(self.norm.boundaries) - 1 != len(self.categories):
#                     raise ValueError(
#                         f"The number of categories ({len(self.categories)}) must be equal to the number of bins ({len(self.norm.boundaries)})."
#                     )

#     @classmethod
#     def from_lb_dtype(
#         cls,
#         lb_dtype: DataTypeBase,
#         colormap: Colormap | str,
#         norm: Normalize = None,
#         categories: list[str] = None,
#     ) -> "EnvironmentVariable":
#         """Create an EnvironmentVariable from a Ladybug datatype."""

#         # check if bounds are within the ladybug datatype's allowable range
#         if norm is None:
#             norm = Normalize(vmin=lb_dtype.min, vmax=lb_dtype.max, clip=True)
#         if norm.vmin < lb_dtype.min or norm.vmax > lb_dtype.max:
#             raise ValueError(
#                 f"norm bounds ({norm.vmin} to {norm.vmax}) must be within the ladybug datatype's bounds ({lb_dtype.min} to {lb_dtype.max})."
#             )

#         obj = cls(
#             name=lb_dtype.name,
#             abbreviation=lb_dtype.abbreviation,
#             unit=lb_dtype.si_units[0],
#             norm=norm,
#             colormap=colormap,
#             categories=categories,
#         )
#         # add ladybug datatype to the object as an attribute
#         setattr(obj, "lb_dtype", lb_dtype)
#         return obj

#     @classmethod
#     def from_bins(
#         cls,
#         name: str,
#         unit: str,
#         bins: list[float],
#         colors: list[str | tuple[int, int, int]],
#         abbreviation: str = None,
#         categories: list[str] = None,
#     ) -> "EnvironmentVariable":
#         """Create an EnvironmentVariable from a list of bins and colors."""

#         if len(bins) != len(colors) + 1:
#             raise ValueError("The number of bins must be one more than the number of colors.")
#         if not all(bins[i] < bins[i + 1] for i in range(len(bins) - 1)):
#             raise ValueError("Bins must be in increasing order.")

#         # create boundaries
#         norm = BoundaryNorm(boundaries=bins, ncolors=len(colors), extend="both", clip=True)

#         # create a segmented colormap, with colors changing at the thresholds of the bins
#         cmap = ListedColormap(colors=[to_rgb(i) for i in colors], name=name)

#         # # create a colormap that maps the normalized values to the colors
#         # colormap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

#         return cls(
#             name=name,
#             abbreviation=abbreviation,
#             unit=unit,
#             norm=norm,
#             colormap=cmap,
#             categories=categories,
#         )

#     def _is_categorical(self) -> bool:
#         """Check if this variable is categorical."""
#         return isinstance(self.norm, BoundaryNorm)

#     def _possible_intervals(self) -> int:
#         """Get the possible number of intervals for this variable."""
#         if self._is_categorical():
#             return len(self.categories)
#         return np.inf

#     def _min(self) -> float:
#         """Get the minimum value of this variable."""
#         return self.norm.vmin

#     def _max(self) -> float:
#         """Get the maximum value of this variable."""
#         return self.norm.vmax

#     def _interval_values(self, n_intervals: int = 11) -> list[float]:
#         """Get the values at the intervals of this variable."""
#         if self._is_categorical():
#             if self._possible_intervals() != n_intervals:
#                 CONSOLE_LOGGER.warning(
#                     f"The number of intervals requested ({n_intervals}) does not match the number of bins ({self._possible_intervals()})."
#                 )
#             return ((self.norm.boundaries[1:] + self.norm.boundaries[:-1]) / 2).tolist()
#         else:
#             return np.linspace(self.norm.vmin, self.norm.vmax, n_intervals).tolist()

#     def average_color(
#         self, return_type: str = "hex"
#     ) -> str | tuple[int, int, int] | tuple[float, float, float]:
#         """Get the average color of this variable, using the average color from the colormap.

#         Args:
#             return_type (str, optional):
#                 The type of color to return.

#         Returns:
#             str | tuple[int, int, int] | tuple[float, float, float]:
#                 The average color of the variable.
#         """

#         color_hex = average_color(colors=[self.colormap(i) for i in np.linspace(0, 1, 100)])

#         return color_to_format(color_hex, return_type)

#     def get_colors(
#         self, n_intervals: int = 11, return_type: str = "hex"
#     ) -> list[tuple[int, int, int]] | list[tuple[float, float, float]] | tuple[str]:
#         """Get the colors of this variable.

#         Args:
#             n (int, optional):
#                 The number of colors to return. Is norm is a BoundaryNorm, then
#                 this will only return the colours at the midpoint of those boundaries.
#             return_type (str, optional):
#                 The type of color to return.

#         Returns:
#             list[tuple[int, int, int]] | list[tuple[float, float, float]] | tuple[str]:
#                 The colors of the variable.
#         """

#         values = self._interval_values(n_intervals=n_intervals)
#         return [self.get_value_color(value=i, return_type=return_type) for i in values]

#     def plotly_colorscale(self, n_intervals: int = 101) -> list[str] | list[tuple[float, str]]:
#         """Get the Plotly colorscale for this variable, from the associated colormap.

#         Args:
#             n_intervals (int, optional):
#                 The number of intervals to use for the colorscale.

#         Returns:
#             list[str] | list[tuple[float, str]]:
#                 The colorscale for the variable, in a format Plotly can understand.
#         """
#         colors = self.get_colors(n_intervals=n_intervals, return_type="hex")

#         plotly_colors = []
#         if self._is_categorical():
#             boundaries = np.nan_to_num(self.norm.boundaries)
#             for color, (low, high) in zip(
#                 *[colors, np.lib.stride_tricks.sliding_window_view(boundaries, 2)]
#             ):
#                 plotly_colors.append(
#                     [
#                         float(np.interp(low, [min(boundaries), max(boundaries)], [0, 1])),
#                         color_to_format(color, "plotly"),
#                     ]
#                 )
#                 plotly_colors.append(
#                     [
#                         float(np.interp(high, [min(boundaries), max(boundaries)], [0, 1])),
#                         color_to_format(color, "plotly"),
#                     ]
#                 )
#             return plotly_colors
#         else:
#             for color in colors:
#                 plotly_colors.append(color_to_format(color, "plotly"))

#         return plotly_colors

#     def _get_value_index(self, value: float) -> int:
#         """Return the index of the value within the categories of the variable.

#         Args:
#             value (float):
#                 The value of the variable.

#         Returns:
#             int:
#                 The index of the value within the categories of the variable.
#         """
#         if not self._is_categorical():
#             raise ValueError(f"{self} is not categorical.")

#         if value < self.norm.boundaries[0]:
#             return 0

#         # if value greater than the maximum value, return the last category
#         if value >= self.norm.boundaries[-1]:
#             return -1

#         # iterate pairs of boundaries and categories
#         for n, (low, high) in enumerate(
#             np.lib.stride_tricks.sliding_window_view(self.norm.boundaries, 2)
#         ):
#             if low <= value < high:
#                 return n

#         raise ValueError(
#             f"Value is not within the boundaries of {self} ({self._min()} to {self._max()})."
#         )

#     def get_value_category(self, value: float) -> str:
#         """Get the category of this variable at the given value.

#         Args:
#             value (float):
#                 The value of the variable.

#         Returns:
#             str:
#                 The category of the variable at the given value.
#         """
#         return self.categories[self._get_value_index(value=value)]

#     def get_value_color(
#         self, value: float, return_type: str = "hex", interpolate_categorical: bool = False
#     ) -> str | tuple[int, int, int] | tuple[float, float, float]:
#         """Get the color of this variable at a given value.

#         Args:
#             value (float):
#                 The value of the variable.
#             return_type (str):
#                 The type of color to return.
#             interpolate_categorical (bool, optional):
#                 Interpolate the color for a categorical variable.
#                 If the variable is not categorical, this will be ignored.
#                 Default is False.

#         Returns:
#             str | tuple[int, int, int] | tuple[float, float, float]:
#                 The color of the variable at the given value.
#         """

#         # handle non-categorical variables first
#         if not self._is_categorical():
#             color = self.colormap(self.norm(value))
#             return color_to_format(color, return_type)

#         if not interpolate_categorical:
#             # get the index of the value
#             idx = self._get_value_index(value=value)
#             # get the color
#             color = self.colormap.colors[idx]
#             # return the color at the index
#             return color_to_format(color, return_type)

#         if interpolate_categorical:
#             # get the index of the value
#             idx = self._get_value_index(value=value)
#             if idx == -1:
#                 idx = len(self.categories) - 1
#             # if index is 0 or -1, get the first/last color
#             if idx == 0 or idx == len(self.categories) - 1:
#                 color = self.colormap.colors[idx]
#             else:
#                 # get the color above and below the value
#                 color_below = self.colormap.colors[idx]
#                 color_above = self.colormap.colors[idx + 1]
#                 # create a temporary colormap with just the two colors
#                 temp_cmap = LinearSegmentedColormap.from_list(
#                     colors=[color_below, color_above], name="temp"
#                 )
#                 # get the value above and below the value
#                 val_below = self.norm.boundaries[idx]
#                 val_above = self.norm.boundaries[idx + 1]
#                 # interpolate the color
#                 color = temp_cmap(np.interp(value, [val_below, val_above], [0, 1]))
#             return color_to_format(color, return_type)

#     def get_legend_handles_labels(
#         self, include_values_in_labels: bool = False
#     ) -> tuple[list[patches.Patch], list[str]]:
#         """Get the legend for this variable."""

#         if not self._is_categorical():
#             raise ValueError(
#                 "Legend handles and labels can only be created for categorical variables."
#             )

#         labels = []
#         for category, (bnd_low, bnd_high) in zip(
#             *[self.categories, np.lib.stride_tricks.sliding_window_view(self.norm.boundaries, 2)]
#         ):
#             if include_values_in_labels:
#                 labels.append(f"{category} ({bnd_low} to {bnd_high})")
#             else:
#                 labels.append(category)
#         handles = []
#         for color in self.get_colors(return_type="hex"):
#             handles.append(
#                 patches.Patch(
#                     facecolor=color,
#                     edgecolor=None,
#                 )
#             )
#         return handles, labels

#     def get_legend(self, include_values_in_labels: bool = False, **kwargs) -> Legend:
#         """Get the legend for this variable."""

#         if not self._is_categorical():
#             raise ValueError(
#                 "Legend handles and labels can only be created for categorical variables. Try using self.get_colorbar instead."
#             )

#         handles, labels = self.get_legend_handles_labels(
#             include_values_in_labels=include_values_in_labels
#         )

#         return plt.legend(
#             handles=handles,
#             labels=labels,
#             **kwargs,
#         )

#     def get_colorbar(self, **kwargs) -> plt.Axes:
#         """Get the colorbar for this variable."""

#         if self._is_categorical():
#             raise ValueError(
#                 "Colorbars can only be created for non-categorical variables. Try using self.get_legend instead."
#             )

#         return plt.colorbar(
#             mappable=plt.cm.ScalarMappable(norm=self.norm, cmap=self.colormap),
#             **kwargs,
#         )

#     # region: PLOT METHODS

#     def plotly_time_indexed_heatmap(self, data: pd.Series) -> go.Figure:
#         """Create a Plotly heatmap for this variable."""

#         # ensure the data is a series
#         if not isinstance(data, pd.Series):
#             raise ValueError("Data must be a pandas Series.")
#         # ensure data has a datetime index
#         if not isinstance(data.index, pd.DatetimeIndex):
#             raise ValueError("Data must have a datetime index.")
#         # ensure data values are numeric
#         if not pd.api.types.is_numeric_dtype(data):
#             raise ValueError("Data values must be numeric.")

#         fig = go.Figure(
#             data=go.Heatmap(
#                 y=data.index.time,
#                 x=data.index.date,
#                 z=data.values,
#                 zmin=np.nan_to_num(self.norm.vmin),
#                 zmax=np.nan_to_num(self.norm.vmax),
#                 colorscale=self.plotly_colorscale(),
#                 connectgaps=False,
#                 hoverongaps=False,
#                 hovertemplate=(
#                     "<b>"
#                     + self.name
#                     + ": %{z:.2f} "
#                     + self.unit
#                     + "</b><br>"
#                     + "Month: %{x}<br>"
#                     + "Day: %{y}<br>"
#                     + "Time: %{y}:00<br>"
#                 ),
#                 name="",
#                 colorbar=dict(
#                     title=self.unit,
#                     # cmin=np.nan_to_num(self.norm.vmin),
#                     # cmax=np.nan_to_num(self.norm.vmax),
#                 ),
#             )
#         )
#         fig.update_layout(
#             template="plotly_white",
#             title=str(self),
#             # xaxis_nticks=53,
#             # yaxis_nticks=13,
#             # yaxis=dict(range=(1, 24)),
#             # xaxis=dict(range=(1, 365)),
#         )
#         fig.update_yaxes(title_text="Hour")
#         fig.update_xaxes(title_text="Day")
#         return fig

#     # endregion: PLOT METHODS


class EnvironmentVariables(Enum):

    UNIVERSAL_THERMAL_CLIMATE_INDEX = CategoricalDtype.from_lb_dtype(
        dtype=UniversalThermalClimateIndex(),
        boundaries=[-273.15, -40, -27, -13, 0, 9, 26, 32, 38, 46, np.inf],
        colors=[
            "#0d104bff",
            "#262972ff",
            "#3452a4ff",
            "#3c65afff",
            "#37bcedff",
            "#2eb349ff",
            "#f38322ff",
            "#c31f25ff",
            "#7f1416ff",
            "#580002ff",
        ],
        categories=[
            "Extreme cold stress",
            "Very strong cold stress",
            "Strong cold stress",
            "Moderate cold stress",
            "Slight cold stress",
            "No thermal stress",
            "Moderate heat stress",
            "Strong heat stress",
            "Very strong heat stress",
            "Extreme heat stress",
        ],
    )


#     # https://en.wikipedia.org/wiki/Wet-bulb_globe_temperature
#     WET_BULB_GLOBE_TEMPERATURE = EnvironmentVariable.from_lb_dtype(
#         lb_dtype=TYPESDICT["WetBulbGlobeTemperature"](),
#         colormap=ListedColormap(
#             colors=["#c1c1c1", "#32cd32", "#ffff00", "#ffa500", "#ff0000", "#000000"]
#         ),
#         norm=BoundaryNorm(
#             boundaries=[
#                 TYPESDICT["WetBulbGlobeTemperature"]().min,
#                 23,
#                 25,
#                 28,
#                 30,
#                 33,
#                 TYPESDICT["WetBulbGlobeTemperature"]().max,
#             ],
#             ncolors=6,
#         ),
#         categories=[
#             "Any activity",
#             "Very heavy activity",
#             "Heavy activity",
#             "Moderate activity",
#             "Light activity",
#             "Resting only",
#         ],
#     )

#     AIR_TEMPERATURE = EnvironmentVariable.from_lb_dtype(
#         lb_dtype=TYPESDICT["AirTemperature"](),
#         colormap="Spectral_r",
#         norm=Normalize(vmin=-10, vmax=50),
#     )


#     AIR_SPEED = EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["AirSpeed"](), colormap="YlGnBu", bounds=(0, 30)
#     )

#     AEROSOL_OPTICAL_DEPTH = EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["AerosolOpticalDepth"](), colormap="coolwarm", bounds=(0, 1)
#     )
#     ATMOSPHERIC_STATION_PRESSURE = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["AtmosphericStationPressure"](), colormap="coolwarm", bounds=(95000, 11000)
#         ),
#     )
#     CEILING_HEIGHT = EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["CeilingHeight"](), colormap="binary", bounds=(0, 99998)
#     )
#     DEW_POINT_TEMPERATURE = EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["DewPointTemperature"](), colormap="summer", bounds=(-50, 50)
#     )
#     DIFFUSE_HORIZONTAL_ILLUMINANCE = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["DiffuseHorizontalIlluminance"](), colormap="cividis", bounds=(0, 150000)
#         ),
#     )
#     DIFFUSE_HORIZONTAL_IRRADIANCE = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["DiffuseHorizontalIrradiance"](), colormap="YlOrRd", bounds=(0, 1500)
#         ),
#     )
#     DIFFUSE_HORIZONTAL_RADIATION = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["DiffuseHorizontalRadiation"](), colormap="YlOrRd", bounds=(0, 1500)
#         ),
#     )
#     DIRECT_HORIZONTAL_IRRADIANCE = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["DirectHorizontalIrradiance"](), colormap="YlOrRd", bounds=(0, 1500)
#         ),
#     )
#     DIRECT_HORIZONTAL_RADIATION = EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["DirectHorizontalRadiation"](), colormap="YlOrRd", bounds=(0, 1500)
#     )
#     DIRECT_NORMAL_ILLUMINANCE = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["DirectNormalIlluminance"](), colormap="cividis", bounds=(0, 150000)
#         ),
#     )
#     DIRECT_NORMAL_IRRADIANCE = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["DirectNormalIrradiance"](), colormap="YlOrRd", bounds=(0, 1500)
#         ),
#     )
#     DIRECT_NORMAL_RADIATION = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["DirectNormalRadiation"](), colormap="YlOrRd", bounds=(0, 1500)
#         ),
#     )
#     DRY_BULB_TEMPERATURE = EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["DryBulbTemperature"](), colormap="Spectral_r", bounds=(-10, 50)
#     )
#     EFFECTIVE_RADIANT_FIELD = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["EffectiveRadiantField"](), colormap="RdBu_r", bounds=(-1000, 1000)
#         ),
#     )
#     ENTHALPY = EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["Enthalpy"](), colormap="cool", bounds=(0, 100)
#     )
#     EXTRATERRESTRIAL_DIRECT_NORMAL_RADIATION = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["ExtraterrestrialDirectNormalRadiation"](),
#             colormap="YlOrRd",
#             bounds=(0, 1500),
#         ),
#     )
#     EXTRATERRESTRIAL_HORIZONTAL_RADIATION = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["ExtraterrestrialHorizontalRadiation"](), colormap="YlOrRd", bounds=(0, 1500)
#         ),
#     )
#     GLOBAL_HORIZONTAL_ILLUMINANCE = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["GlobalHorizontalIlluminance"](), colormap="cividis", bounds=(0, 150000)
#         ),
#     )
#     GLOBAL_HORIZONTAL_IRRADIANCE = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["GlobalHorizontalIrradiance"](), colormap="YlOrRd", bounds=(0, 1500)
#         ),
#     )
#     GLOBAL_HORIZONTAL_RADIATION = EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["GlobalHorizontalRadiation"](), colormap="YlOrRd", bounds=(0, 1500)
#     )
#     GROUND_TEMPERATURE = EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["GroundTemperature"](), colormap="Wistia", bounds=(-10, 60)
#     )
#     HEAT_INDEX_TEMPERATURE = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["HeatIndexTemperature"](), colormap="rainbow", bounds=(-30, 50)
#         ),
#     )
#     HORIZONTAL_INFRARED_RADIATION_INTENSITY = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["HorizontalInfraredRadiationIntensity"](),
#             colormap="Oranges",
#             bounds=(100, 600),
#         ),
#     )
#     HUMIDITY_RATIO = EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["HumidityRatio"](), colormap="GnBu", bounds=(0, 0.01)
#     )
#     ILLUMINANCE = EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["Illuminance"](), colormap="cividis", bounds=(0, 150000)
#     )
#     IRRADIANCE = EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["Irradiance"](), colormap="YlOrRd", bounds=(0, 1500)
#     )
#     LIQUID_PRECIPITATION_DEPTH = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["LiquidPrecipitationDepth"](), colormap="Blues", bounds=(0, 100)
#         ),
#     )
#     LIQUID_PRECIPITATION_QUANTITY = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["LiquidPrecipitationQuantity"](), colormap="Blues", bounds=(0, 100)
#         ),
#     )
#     LUMINANCE = EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["Luminance"](), colormap="cividis", bounds=(0, 600)
#     )
#     MEAN_RADIANT_TEMPERATURE = (
#         EnvironmentVariable.from_lb_dtype(
#             TYPESDICT["MeanRadiantTemperature"](), colormap="YlOrBr", bounds=(-10, 100)
#         ),
#     )
# OPAQUE_SKY_COVER = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["OpaqueSkyCover"](), colormap="Greys", bounds=(0, 10)
# )
# OPERATIVE_TEMPERATURE = (
#     EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["OperativeTemperature"](), colormap="Spectral_r", bounds=(-10, 50)
#     ),
# )
# PHYSIOLOGICAL_EQUIVALENT_TEMPERATURE = (
#     EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["PhysiologicalEquivalentTemperature"](), colormap="viridis", bounds=None
#     ),
# )
# PRECIPITABLE_WATER = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["PrecipitableWater"](), colormap="Blues", bounds=(0, 100)
# )
# PREDICTED_MEAN_VOTE = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["PredictedMeanVote"](), colormap="viridis", bounds=None
# )
# PRESSURE = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["Pressure"](), colormap="viridis", bounds=None
# )
# PREVAILING_OUTDOOR_TEMPERATURE = (
#     EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["PrevailingOutdoorTemperature"](), colormap="viridis", bounds=None
#     ),
# )
# RADIANT_TEMPERATURE = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["RadiantTemperature"](), colormap="viridis", bounds=None
# )
# RADIATION = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["Radiation"](), colormap="viridis", bounds=None
# )
# RELATIVE_HUMIDITY = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["RelativeHumidity"](), colormap="PuBuGn", bounds=(0, 100)
# )
# SKIN_TEMPERATURE = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["SkinTemperature"](), colormap="viridis", bounds=None
# )
# SKY_TEMPERATURE = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["SkyTemperature"](), colormap="RdBu_r", bounds=(-50, 50)
# )
# SNOW_DEPTH = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["SnowDepth"](), colormap="PuBu", bounds=(0, 100)
# )
# SPEED = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["Speed"](), colormap="YlGnBu", bounds=(0, 100)
# )
# STANDARD_EFFECTIVE_TEMPERATURE = (
#     EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["StandardEffectiveTemperature"](), colormap="viridis", bounds=None
#     ),
# )
# TEMPERATURE = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["Temperature"](), colormap="viridis", bounds=None
# )
# THERMAL_COMFORT = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["ThermalComfort"](), colormap="viridis", bounds=None
# )
# THERMAL_CONDITION = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["ThermalCondition"](), colormap="viridis", bounds=None
# )
# THERMAL_CONDITION_ELEVEN_POINT = (
#     EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["ThermalConditionElevenPoint"](), colormap="viridis", bounds=None
#     ),
# )
# THERMAL_CONDITION_FIVE_POINT = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["ThermalConditionFivePoint"](), colormap="viridis", bounds=None
# )
# THERMAL_CONDITION_NINE_POINT = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["ThermalConditionNinePoint"](), colormap="viridis", bounds=None
# )
# THERMAL_CONDITION_SEVEN_POINT = (
#     EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["ThermalConditionSevenPoint"](), colormap="viridis", bounds=None
#     ),
# )
# TOTAL_SKY_COVER = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["TotalSkyCover"](), colormap="viridis", bounds=None
# )

# UTCI_CATEGORY = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["UTCICategory"](), colormap="viridis", bounds=None
# )
# VISIBILITY = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["Visibility"](), colormap="viridis", bounds=None
# )

# WET_BULB_TEMPERATURE = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["WetBulbTemperature"](), colormap="viridis", bounds=None
# )
# WIND_CHILL_TEMPERATURE = (
#     EnvironmentVariable.from_lb_dtype(
#         TYPESDICT["WindChillTemperature"](), colormap="viridis", bounds=None
#     ),
# )
# WIND_DIRECTION = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["WindDirection"](), colormap="viridis", bounds=None
# )
# WIND_SPEED = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["WindSpeed"](), colormap="viridis", bounds=None
# )
# ZENITH_LUMINANCE = EnvironmentVariable.from_lb_dtype(
#     TYPESDICT["ZenithLuminance"](), colormap="viridis", bounds=None
# )


# a mapping between downstream variables and their upstream dependencies
DEPENDENCIES = {
    "actual_sensation_vote": [
        "air_temperature",
        "solar_radiation",
        "air_velocity",
        "relative_humidity",
    ],
    "apparent_temperature": [
        "air_temperature",
        "air_velocity",
        "relative_humidity",
    ],
    "discomfort_index": [
        "air_temperature",
        "relative_humidity",
    ],
    "heat_index": [
        "air_temperature",
        "relative_humidity",
    ],
    "humidex": [
        "air_temperature",
        "relative_humidity",
    ],
    "physiologic_equivalent_temperature": [
        "air_temperature",
        "mean_radiant_temperature",
        "air_velocity",
        "relative_humidity",
        "metabolic_rate",
        "clo_value",
    ],
    "standard_effective_temperature": [
        "air_temperature",
        "mean_radiant_temperature",
        "air_velocity",
        "relative_humidity",
        "metabolic_rate",
        "clo_value",
    ],
    "thermal_sensation": [
        "air_temperature",
        "air_velocity",
        "relative_humidity",
        "solar_radiation",
    ],
    "universal_thermal_climate_index": [
        "air_temperature",
        "mean_radiant_temperature",
        "air_velocity",
        "relative_humidity",
    ],
    "wet_bulb_globe_temperature": [
        "air_temperature",
        "mean_radiant_temperature",
        "air_velocity",
        "relative_humidity",
    ],
    "windchill_temp": [
        "air_temperature",
        "air_velocity",
    ],
}

VARIABLE_ACRONYM = Enum(
    "Variable",
    {
        "air_velocity": "vair",
        "air_temperature": "Tair",
        "relative_humidity": "RH",
        "solar_radiation": "Esolar",
        "mean_radiant_temperature": "MRT",
        "clo_value": "CLO",
        "metabolic_rate": "MET",
        "month_of_year": "moy",
        "hour_of_day": "hod",
        "actual_sensation_vote": "ASV",
        "apparent_temperature": "AT",
        "discomfort_index": "DT",
        "heat_index": "HIT",
        "humidex": "HDX",
        "physiologic_equivalent_temperature": "PET",
        "standard_effective_temperature": "SET",
        "thermal_sensation": "TS",
        "universal_thermal_climate_index": "UTCI",
        "wet_bulb_globe_temperature": "WBGT",
        "windchill_temp": "WCT",
    },
)


if __name__ == "__main__":
    
    from ladybug_comfort.collection.utci import UTCI
    from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series

    # create a categorical bar chart for the Universal Thermal Climate Index
    epw = EPW(r"C:\Users\tgerrish\OneDrive - Buro Happold\Weather files\epws\MRT_NOUADHIBOU_614150_IW2.epw")
    utci_col = UTCI.from_epw(epw=epw).universal_thermal_climate_index
    utci_s = collection_to_series(utci_col)

    # convert into table array
    utci_table = utci_s.

    # create 
    fig = go.Figure()

    # create plotly stacked bar, with each column a date and each 

