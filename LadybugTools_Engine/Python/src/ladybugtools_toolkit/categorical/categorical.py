"""Categorical objects for grouping data into bins."""

# pylint: disable=W0212,E0401
import calendar
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties

# pylint: enable=E0401
import numpy as np
import pandas as pd
from ladybug.legend import Color
from ladybug.location import Location
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, to_hex, to_rgba
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from mpl_toolkits.axes_grid1 import make_axes_locatable
from python_toolkit.bhom.analytics import bhom_analytics
from python_toolkit.plot.heatmap import heatmap
from python_toolkit.plot.timeseries import timeseries


from ..helpers import rolling_window, sunrise_sunset, validate_timeseries
from ..plot.utilities import contrasting_color


@dataclass(init=True, repr=True)
class Categorical:
    """A class to hold categorical data.

    Args:
        bins (tuple[float], optional):
            The bin edges for the categories. These are right-inclusive, with
            the exception of the first bin which is also left-inclusive.
        bin_names (tuple[str], optional):
            The names of the categories.
        colors (tuple[str | tuple], optional):
            The colors of the categories.
        name (str, optional):
            The name of the categories.
    """

    bins: tuple[float] = field(default_factory=tuple, repr=False)
    bin_names: tuple[str] = field(default_factory=tuple)
    colors: tuple[str] = field(default_factory=tuple, repr=True)
    name: str = field(default="GenericCategories")

    def __post_init__(self):
        # ensure colors are valid
        if len(self.colors) == 0:
            cycle = tuple(plt.rcParams["axes.prop_cycle"].by_key()["color"])
            while len(cycle) < len(self.bins):
                cycle += cycle
            self.colors = cycle[: len(self.bins) - 1]
        self.colors = tuple(to_hex(i, keep_alpha=True) for i in self.colors)

        # ensure bin names are valid
        if len(self.bin_names) == 0:
            self.bin_names = [str(i) for i in self.interval_index]

        # ensure the number of bins, colors and bin names are consistent
        if len(set([len(self.bin_names), len(self.colors), (len(self.bins) - 1)])) > 1:
            raise ValueError(
                f"The number of colors ({len(self.colors)}) and bin names "
                f"({len(self.bin_names)}) must be one less than the number of "
                f"bins ({len(self.bins) - 1})."
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
        bins: tuple[float],
        cmap: Colormap,
        bin_names: tuple[str] = (),
        name: str = "",
    ):
        """Create a categorical from a colormap.

        Args:
            bins (tuple[float]):
                The bin edges for the categories.
            cmap (Colormap):
                The colormap to use.
            bin_names (tuple[str], optional):
                The names for each of the bins. Defaults to () which names
                each bin using its boundaries.
            name (str, optional):
                The name for this categories object. Defaults to "" which
                uses the colormap name.

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

        _bins_finite = [x for x in bins if not (np.isinf(x))]

        normalised = np.interp(_bins, [min(_bins_finite), max(_bins_finite)], [0, 1])
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
    def descriptions(self) -> list[str]:
        """The descriptions of the categories.

        Returns:
            List[str]:
                The descriptions of the categories.
        """
        return [f"{i.left} to {i.right}" for i in self]

    @property
    def bin_names_detailed(self) -> list[str]:
        """The detailed bin names."""
        return [f"{nom} ({desc})" for desc, nom in list(zip(*[self.descriptions, self.bin_names]))]

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
        elif np.isinf(self.bins[0]) and not np.isinf(self.bins[-1]):
            colors = self.colors[1:]
        elif not np.isinf(self.bins[0]) and np.isinf(self.bins[-1]):
            colors = self.colors[:-1]
        else:
            colors = self.colors
        colors = [to_rgba(i) for i in colors]

        cmap = ListedColormap(colors=colors, name=self.name)
        cmap.set_over(self.colors[-1])
        cmap.set_under(self.colors[0])
        return cmap

    @property
    def norm(self) -> BoundaryNorm:
        """Return the boundary-norm associate with this categorical.

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
        if len(boundaries) == 1:
            raise ValueError(
                "The current Categorical object has unbounded edges and cannot be used to create a BoundaryNorm."
            )
        return BoundaryNorm(boundaries=boundaries, ncolors=self.cmap.N)

    @property
    def bins_finite(self) -> tuple[float]:
        """The finite bins excluding any which are infinite.

        Returns:
            Tuple[float]:
                The finite bins.
        """
        return tuple(i for i in self.bins if not np.isinf(i))

    @property
    def lb_colors(self) -> tuple[Color]:
        """The ladybug colors.

        Returns:
            tuple[Color]:
                The ladybug color objects.
        """
        return tuple(
            Color(*i)
            for i in (np.array([to_rgba(color) for color in self.colors]) * 255).astype(int)
        )

    @property
    def _bin_name_interval(self) -> dict[str, pd.Interval]:
        """The bin name to interval dictionary.

        Returns:
            dict[str, pd.Interval]:
                The bin name to interval dictionary.
        """
        return dict(zip(self.bin_names, self.interval_index))

    @property
    def _detailed_bin_name_interval(self) -> dict[str, pd.Interval]:
        """The bin name to interval dictionary.

        Returns:
            dict[str, pd.Interval]:
                The bin name to interval dictionary.
        """
        return dict(zip(self.bin_names_detailed, self.interval_index))

    @property
    def _interval_bin_name(self) -> dict[pd.Interval, str]:
        """The interval to bin name dictionary.

        Returns:
            dict[pd.Interval, str]:
                The interval to bin name dictionary.
        """
        return dict(zip(self.interval_index, self.bin_names))

    @bhom_analytics()
    def interval_from_bin_name(self, bin_name: str) -> pd.Interval:
        """Return the interval from the bin name.

        Args:
            bin_name (str):
                The bin name.

        Returns:
            pd.Interval:
                The interval associated with the bin name.
        """
        try:
            return self._bin_name_interval[bin_name]
        except KeyError:
            return self._detailed_bin_name_interval[bin_name]

    @bhom_analytics()
    def bin_name_from_interval(self, interval: pd.Interval) -> str:
        """Return the bin name from the interval.

        Args:
            interval (pd.Interval):
                The interval.

        Returns:
            str:
                The bin name associated with the interval.
        """
        return self._interval_bin_name[interval]

    @bhom_analytics()
    def color_from_bin_name(self, bin_name: str) -> str:
        """Return the color from the bin name.

        Args:
            bin_name (str):
                The bin name.

        Returns:
            str:
                The color associated with the bin name.
        """
        return dict(zip(self.bin_names, self.colors))[bin_name]

    @bhom_analytics()
    def get_color(self, value: float | int, as_array: bool = False) -> str:
        """Return the color associated with the categorised value.

        Args:
            value (float | int):
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
            return to_hex(color, keep_alpha=True)
        return color

    @bhom_analytics()
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
            data, self.bins, labels=self.bin_names, include_lowest=True, right=True
        )
        if categorical.isna().any():
            raise ValueError(
                f"The input value/s are outside the range of the categories ({self.bins[0]} <= x < {self.bins[-1]})."
            )
        return categorical

    @bhom_analytics()
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
        result = self.categorise(data).value_counts()[list(self.bin_names)]
        if density:
            return result / len(data)
        return result

    @bhom_analytics()
    def timeseries_summary_monthly(self, series: pd.Series, density: bool = False) -> pd.DataFrame:
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
            self.categorise(series).groupby(series.index.month).value_counts().unstack()
        ).sort_index(axis=0)
        counts.columns.name = None
        counts.index.name = "Month"
        if density:
            return counts.div(counts.sum(axis=1), axis=0)
        return counts

    @bhom_analytics()
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
        for desc, (idx, val) in list(zip(*[self.bin_names, result.items()])):
            statements.append(f'"{desc}" occurs {val} times ({result_density[idx]:0.1%}).')
        return "\n".join(statements)

    @bhom_analytics()
    def create_legend_handles_labels(
        self, label_type: str = "value"
    ) -> tuple[list[patches.Patch], list[str]]:
        """Create legend handles and labels for this categorical.

        Args:
            label_type (str, optional):
                Whether to use the bin "value", "interval" or "name" for the legend labels.

        Returns:
            tuple[list[patches.Patch], list[str]]:
                (handles, labels)

        """

        match label_type:
            case "value":
                labels = self.bins
            case "interval":
                labels = self.descriptions
            case "name":
                labels = self.bin_names
            case _:
                raise ValueError("The label must be 'value', 'interval' or 'name'.")

        handles = []
        for color in self.colors:
            handles.append(
                patches.Patch(
                    facecolor=color,
                    edgecolor=None,
                )
            )
        return handles, labels

    def create_legend(self, label_type: str = "value", **kwargs) -> Legend:
        """Create a legend for this categorical.

        Args:
            label_type (str, optional):
                Whether to use the bin "value", "interval" or "name" for the legend labels.
            **kwargs:
                Additional keyword arguments to pass to plt.legend.

        Returns:
            plt.Legend:
                The legend object.
        """
        handles, labels = self.create_legend_handles_labels(label_type=label_type)
        return plt.legend(handles, labels, **kwargs)

    @bhom_analytics()
    def annual_monthly_histogram(
        self,
        series: pd.Series,
        ax: plt.Axes = None,
        show_legend: bool = False,
        show_labels: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Create a monthly histogram of a pandas Series.

        Args:
            series (pd.Series):
                The pandas Series to plot. Must have a datetime index.
            ax (plt.Axes, optional):
                An optional plt.Axes object to populate. Defaults to None,
                which creates a new plt.Axes object.
            show_legend (bool, optional):
                Whether to show the legend. Defaults to False.
            show_labels (bool, optional):
                Whether to show the labels on the bars. Defaults to False.
            **kwargs:
                Additional keyword arguments to pass to plt.bar.

        Returns:
            plt.Axes:
                The populated plt.Axes object.
        """

        validate_timeseries(series)

        if ax is None:
            ax = plt.gca()

        color_lookup = dict(zip(self.bin_names, self.colors))

        t = self.timeseries_summary_monthly(series, density=True)

        t.plot(
            ax=ax,
            kind="bar",
            stacked=True,
            color=[color_lookup[i] for i in t.columns],
            width=kwargs.pop("width", 1),
            legend=False,
            **kwargs,
        )
        ax.set_xlim(-0.5, len(t) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticklabels(
            [calendar.month_abbr[int(i._text)] for i in ax.get_xticklabels()],
            ha="center",
            rotation=0,
        )
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))

        if show_legend:
            ax.legend(
                bbox_to_anchor=(1, 1),
                loc="upper left",
                borderaxespad=0.0,
                frameon=False,
                title=self.name,
            )

        if show_labels:
            for i, c in enumerate(ax.containers):
                label_colors = [contrasting_color(i.get_facecolor()) for i in c.patches]
                labels = [f"{v.get_height():0.1%}" if v.get_height() > 0.15 else "" for v in c]
                ax.bar_label(
                    c,
                    labels=labels,
                    label_type="center",
                    color=label_colors[i],
                    fontsize="x-small",
                )

        return ax

    @bhom_analytics()
    def annual_monthly_table(
        self,
        series: pd.Series,
        ax: plt.Axes = None,
        show_legend: bool = False,
        show_labels: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Create a monthly histogram of a pandas Series.

        Args:
            series (pd.Series):
                The pandas Series to plot. Must have a datetime index.
            ax (plt.Axes, optional):
                An optional plt.Axes object to populate. Defaults to None,
                which creates a new plt.Axes object.
            show_legend (bool, optional):
                Whether to show the legend. Defaults to False.
            show_labels (bool, optional):
                Whether to show the labels on the bars. Defaults to False.
            **kwargs:
                Additional keyword arguments to pass to plt.bar.

        Returns:
            plt.Axes:
                The populated plt.Axes object.
        """

        validate_timeseries(series)

        if ax is None:
            ax = plt.gca()

        color_lookup = dict(zip(self.bin_names, self.colors))

        t = self.timeseries_summary_monthly(series, density=False)
        t = t.iloc[:,:-1]
        t = t.transpose().iloc[::-1]

        # Hide axes
        ax.axis('off')

        # Create table
        colors = self.colors[::-1][1:]
        table = ax.table(cellText=t.values,  rowLabels = t.index, colLabels = t.columns, rowColours = colors, loc='center')
        for (row, col), cell in table.get_celld().items():
            if (row == 0) or (col == -1):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))

        return ax

    @bhom_analytics()
    def annual_heatmap(self, series: pd.Series, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """Create a heatmap showing the annual hourly categorical assignment for the given series.

        Args:
            series (pd.Series):
                A time-indexed pandas Series object.
            ax (plt.Axes, optional):
                A matplotlib Axes object to plot on. Defaults to None.
            **kwargs:
                Additional keyword arguments to pass to the heatmap function.

        Returns:
            plt.Axes:
                A matplotlib Axes object.
        """

        validate_timeseries(series)

        if ax is None:
            ax = plt.gca()

        heatmap(
            series,
            ax=ax,
            cmap=self.cmap,
            norm=self.norm,
            extend="both",
            **kwargs,
        )

        return ax

    @bhom_analytics()
    def annual_heatmap_histogram(
        self,
        series: pd.Series,
        location: Location = None,
        figsize: tuple[float] = (15, 5),
        show_legend: bool = True,
        **kwargs,
    ) -> plt.Figure:
        """Plot the shade benefit category.

        Args:
            series (pd.Series):
                A pandas Series object with a datetime index.
            location (Location, optional):
                A location object used to plot sun up/down times in the heatmap axis.
                Defaults to None.
            figsize (tuple[float], optional):
                The size of the figure. Defaults to (15, 5).
            show_legend (bool, optional):
                Whether to show the legend. Defaults to True.
            **kwargs:
                Additional keyword arguments to pass to the plotting function.
                title (str, optional):
                    The title of the plot. Defaults to None.
                sunrise_color (str, optional):
                    The color of the sunrise line. Defaults to "k" ... if location is also provided.
                sunset_color (str, optional):
                    The color of the sunset line. Defaults to "k" ... if location is also provided.
                ncol (int, optional):
                    The number of columns in the legend. Defaults to 5.
                show_labels (bool, optional):
                    Whether to show the labels on the bars. Defaults to True.

        Returns:
            plt.Figure:
                A figure object.
        """

        sunrise_color = kwargs.pop("sunrise_color", "k")
        sunset_color = kwargs.pop("sunset_color", "k")
        ncol = kwargs.pop("ncol", 5)
        show_labels = kwargs.pop("show_labels", True)
        title = kwargs.pop("title", None)

        ti = self.name + ("" if title is None else f"\n{title}")

        if show_legend:
            fig, (hmap_ax, legend_ax, bar_ax) = plt.subplots(
                3, 1, figsize=figsize, height_ratios=(7, 0.5, 2)
            )
            legend_ax.set_axis_off()
            handles, labels = self.create_legend_handles_labels(label_type="name")
            legend_ax.legend(
                handles,
                labels,
                loc="center",
                bbox_to_anchor=(0.5, 0),
                fontsize="small",
                ncol=ncol,
                borderpad=2.5,
            )
        else:
            fig, (hmap_ax, bar_ax) = plt.subplots(2, 1, figsize=figsize, height_ratios=(7, 2))

        self.annual_heatmap(series, show_colorbar=False, ax=hmap_ax)

        self.annual_monthly_histogram(ax=bar_ax, series=series, show_labels=show_labels)

        hmap_ax.set_title(ti)

        # add sun up indicator lines
        if location is not None:
            ylimz = hmap_ax.get_ylim()
            xlimz = hmap_ax.get_xlim()
            ymin = min(hmap_ax.get_ylim())
            sun_rise_set = sunrise_sunset(location=location)
            sunrise = [
                ymin + (((i.time().hour * 60) + (i.time().minute)) / (60 * 24))
                for i in sun_rise_set.sunrise
            ]
            sunset = [
                ymin + (((i.time().hour * 60) + (i.time().minute)) / (60 * 24))
                for i in sun_rise_set.sunset
            ]
            xx = np.arange(min(hmap_ax.get_xlim()), max(hmap_ax.get_xlim()) + 1, 1)
            hmap_ax.plot(xx, sunrise, zorder=9, c=sunrise_color, lw=1)
            hmap_ax.plot(xx, sunset, zorder=9, c=sunset_color, lw=1)
            hmap_ax.set_xlim(xlimz)
            hmap_ax.set_ylim(ylimz)

        return fig

    @bhom_analytics()
    def annual_threshold_chart(self, series: pd.Series, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """Create an annual chart annotated with lines at the category's cutoff values for the given series.

        Args:
            series (pd.Series):
                A time-indexed pandas Series object.
            ax (plt.Axes, optional):
                A matplotlib Axes object to plot on. Defaults to None.
            **kwargs:
                Additional keyword arguments to pass to the heatmap function.
            show_legend (bool, optional):
                Whether to show the legend. Defaults to True.

        Returns:
            plt.Axes:
                A matplotlib Axes object.
        """

        validate_timeseries(series)

        if ax is None:
            ax = plt.gca()

        plt = timeseries(
            series,
            ax=ax,
            **kwargs,
        )

        for bin_name, interval in list(zip(*[self.bin_names, self.interval_index])):
            val = interval.right
            if np.isfinite(val) and val>ax.get_ylim()[0]:
                col = self.color_from_bin_name(bin_name)
                ax.axhline(val, 0, 1, color = col, ls="--", lw=2)
                ax.text(0.5, val, val, va='center', ha='center', backgroundcolor='w', color = col, transform=ax.get_yaxis_transform())
        return ax


class ComfortClass(Enum):
    """Thermal comfort categories."""

    TOO_COLD = auto()
    COMFORTABLE = auto()
    TOO_HOT = auto()

    @property
    def color(self) -> str:
        """Get the associated color."""
        d = {
            ComfortClass.TOO_COLD: "#3C65AF",
            ComfortClass.COMFORTABLE: "#2EB349",
            ComfortClass.TOO_HOT: "#C31F25",
        }
        return d[self]

    @property
    def text(self) -> str:
        """Get the associated text."""
        d = {
            ComfortClass.TOO_COLD: "Too cold",
            ComfortClass.COMFORTABLE: "Comfortable",
            ComfortClass.TOO_HOT: "Too hot",
        }
        return d[self]


@dataclass(init=True, repr=True)
class CategoricalComfort(Categorical):
    """A class to hold categorical comfort data.

    Args:
        comfort_classes (tuple[ComfortClass]):
            The comfort classes to use.
    """

    comfort_classes: tuple[ComfortClass] = field(default_factory=tuple, repr=False)

    def __post_init__(self):
        if len(self.comfort_classes) == 0:
            raise ValueError("The comfort classes cannot be empty.")
        if len(self.comfort_classes) != len(self):
            raise ValueError("The number of comfort classes must match the number of bins.")
        return super().__post_init__()

    @bhom_analytics()
    def simplify(self) -> "CategoricalComfort":
        """Return a simplified version of this object based on the assigned comfort clases.

        Returns:
            CategoricalComfort:
                The simplified categorical comfort.
        """
        d = {}
        for comfort_class, bin_left in list(zip(*[self.comfort_classes, self.bins])):
            if comfort_class in d:
                continue
            d[comfort_class] = bin_left
        if len(d.keys()) != len(ComfortClass):
            raise ValueError(
                f"The comfort classes must include all comfort classes {[i.name for i in ComfortClass]}."
            )

        return CategoricalComfort(
            bins=list(d.values()) + [self.bins[-1]],
            bin_names=[i.text for i in ComfortClass],
            colors=[i.color for i in ComfortClass],
            name=self.name + " (simplified)",
            comfort_classes=list(ComfortClass),
        )

    def heatmap_histogram(
        self, series: pd.Series, show_colorbar: bool = True, figsize: tuple[float] = (15, 5)
    ) -> Figure:
        """Create a heatmap histogram. This combines the heatmap and histogram.

        Args:
            series (pd.Series):
                A time indexed pandas series.
            show_colorbar (bool, optional):
                Whether to show the colorbar in the plot. Defaults to True.
            figsize (tuple[float], optional):
                Change the figsize. Defaults to (15, 5).

        Returns:
            Figure:
                A figure!
        """

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        spec = fig.add_gridspec(
            ncols=1, nrows=2, width_ratios=[1], height_ratios=[5, 2], hspace=0.0
        )
        heatmap_ax = fig.add_subplot(spec[0, 0])
        histogram_ax = fig.add_subplot(spec[1, 0])

        # Add heatmap
        self.annual_heatmap(series, ax=heatmap_ax, show_colorbar=False)
        # Add stacked plot
        self.annual_monthly_histogram(series=series, ax=histogram_ax, show_labels=True)

        if show_colorbar:
            # add colorbar
            divider = make_axes_locatable(histogram_ax)
            colorbar_ax = divider.append_axes("bottom", size="20%", pad=0.7)
            cb = fig.colorbar(
                mappable=heatmap_ax.get_children()[0],
                cax=colorbar_ax,
                orientation="horizontal",
                drawedges=False,
                extend="both",
            )
            cb.outline.set_visible(False)
            for bin_name, interval in list(zip(*[self.bin_names, self.interval_index])):
                if np.isinf(interval.left):
                    ha = "right"
                    position = interval.right
                elif np.isinf(interval.right):
                    ha = "left"
                    position = interval.left
                else:
                    ha = "center"
                    position = np.mean([interval.left, interval.right])
                colorbar_ax.text(
                    position,
                    1.05,
                    textwrap.fill(bin_name, 11),
                    ha=ha,
                    va="bottom",
                    fontsize="x-small",
                )

        heatmap_ax.set_title(f"{self.name}\n{series.name}", y=1, ha="left", va="bottom", x=0)

        return fig
