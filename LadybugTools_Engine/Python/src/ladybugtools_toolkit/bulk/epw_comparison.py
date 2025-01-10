import calendar
import functools
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from ladybug.epw import EPW, Location
from ladybug.header import Header
from ladybug_comfort.collection.utci import UTCI
from matplotlib.axes import Axes
from python_toolkit.bhom.logging import CONSOLE_LOGGER
from tqdm import tqdm

from ..ladybug_extension.datacollection import summarise_collection
from ..ladybug_extension.epw import (
    HourlyContinuousCollection,
    average_collection,
    average_epw,
    average_location,
    collection_to_series,
    degree_time,
    epw_to_dataframe,
)
from ..ladybug_extension.header import header_from_string, header_to_string
from .process import SummariseClimate, SummariseClimateConfig

MAX_DISTANCE = 25  # km, between EPWs before warning raised
VARIABLES = {
    "albedo": "Albedo (fraction)",
    "atmospheric_station_pressure": "Atmospheric Station Pressure (Pa)",
    "ceiling_height": "Ceiling Height (m)",
    "days_since_last_snowfall": "Days Since Last Snowfall (day)",
    "dew_point_temperature": "Dew Point Temperature (C)",
    "diffuse_horizontal_illuminance": "Diffuse Horizontal Illuminance (lux)",
    "diffuse_horizontal_radiation": "Diffuse Horizontal Radiation (Wh/m2)",
    "direct_normal_illuminance": "Direct Normal Illuminance (lux)",
    "direct_normal_radiation": "Direct Normal Radiation (Wh/m2)",
    "dry_bulb_temperature": "Dry Bulb Temperature (C)",
    "extraterrestrial_direct_normal_radiation": "Extraterrestrial Direct Normal Radiation (Wh/m2)",
    "extraterrestrial_horizontal_radiation": "Extraterrestrial Horizontal Radiation (Wh/m2)",
    "global_horizontal_illuminance": "Global Horizontal Illuminance (lux)",
    "global_horizontal_radiation": "Global Horizontal Radiation (Wh/m2)",
    "horizontal_infrared_radiation_intensity": "Horizontal Infrared Radiation Intensity (W/m2)",
    "liquid_precipitation_depth": "Liquid Precipitation Depth (mm)",
    "liquid_precipitation_quantity": "Liquid Precipitation Quantity (fraction)",
    "opaque_sky_cover": "Opaque Sky Cover (tenths)",
    "precipitable_water": "Precipitable Water (mm)",
    "present_weather_codes": "Present Weather Codes (codes)",
    "present_weather_observation": "Present Weather Observation (observation)",
    "relative_humidity": "Relative Humidity (%)",
    "snow_depth": "Snow Depth (cm)",
    "total_sky_cover": "Total Sky Cover (tenths)",
    "visibility": "Visibility (km)",
    "wind_direction": "Wind Direction (degrees)",
    "wind_speed": "Wind Speed (m/s)",
    "years": "Year (yr)",
    "zenith_luminance": "Zenith Luminance (cd/m2)",
    "wet_bulb_temperature": "Wet Bulb Temperature (C)",
    "universal_thermal_climate_index_shaded": "Universal Thermal Climate Index [Shaded] (C)",
    "universal_thermal_climate_index_unshaded": "Universal Thermal Climate Index [Unshaded] (C)",
    "universal_thermal_climate_index_unshaded_nowind": "Universal Thermal Climate Index [Unshaded+NoWind] (C)",
}


def location_distance(location1: Location, location2: Location) -> float:
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)

    Args:
        location1: Location object of the first location
        location2: Location object of the second location

    Returns:
        distance: The distance between the two locations in km
    """

    r = 6373.0  # approximate radius of earth in km
    lat1 = np.radians(location1.latitude)
    lon1 = np.radians(location1.longitude)
    lat2 = np.radians(location2.latitude)
    lon2 = np.radians(location2.longitude)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = r * c
    return distance


class EPWComparison:
    """A class for the holding of EPW objects and comparison of their data.

    Args:
        epws: A list of EPW objects to compare

    """

    def __init__(self, epws: list[EPW]):
        self.epws = epws

        self._assign_properties()

    def __repr__(self) -> str:
        txt = "\n".join([f"- {i}" for i in self.epw_ids])
        return f"EPWComparison:\n{txt}"

    def __len__(self) -> int:
        return len(self.epws)

    @property
    def epws(self):
        """The list of EPW objects to compare."""
        return self._epws

    @epws.setter
    def epws(self, d):
        if not isinstance(d, (list, tuple)):
            raise ValueError("epws must be a list")
        for epw in d:
            if not isinstance(epw, EPW):
                raise ValueError("epws must be a 1d list of EPW objects")
        if len(set(d)) != len(d):
            raise ValueError("epws must contain unique EPW objects")
        if len(d) < 2:
            raise ValueError("epws must contain at least 2 EPW objects")
        if len(d) > 24:
            raise ValueError("epws must contain 24 or fewer EPW objects")

        # check that EPWs are within a certain distance of each other and raise a warning if not
        locations = [i.location for i in d]
        for loc in locations:
            distance = location_distance(locations[0], loc)
            if distance > MAX_DISTANCE:
                warnings.warn(f"EPWs are not within {MAX_DISTANCE} km of each other", UserWarning)

        self._epws = d

    @functools.cached_property
    def average_epw(self) -> EPW:
        """Get the average EPW object of the EPWs."""
        return average_epw(self.epws)

    @property
    def epw_ids(self) -> list[str]:
        """The IDs of the EPWs."""
        return [Path(i.file_path).stem for i in self.epws]

    @property
    def locations(self) -> list[Location]:
        """The locations of the EPWs."""
        return [i.location for i in self.epws]

    @property
    def average_location(self) -> Location:
        """The average location of the EPWs."""
        return average_location([i.location for i in self.epws])

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """A DataFrame of the EPW data."""
        dfs = []
        pbar = tqdm(self.epws)
        for i in pbar:
            pbar.set_description(f"Processing {Path(i.file_path).stem}")
            df = epw_to_dataframe(i, include_additional=True)

            # add comfort metrics
            df["Universal Thermal Climate Index [Shaded] (C)"] = collection_to_series(
                UTCI.from_epw(
                    epw=i, include_wind=True, include_sun=False
                ).universal_thermal_climate_index
            )
            df["Universal Thermal Climate Index [Unshaded] (C)"] = collection_to_series(
                UTCI.from_epw(
                    epw=i, include_wind=True, include_sun=True
                ).universal_thermal_climate_index
            )
            df["Universal Thermal Climate Index [Unshaded+NoWind] (C)"] = collection_to_series(
                UTCI.from_epw(
                    epw=i, include_wind=False, include_sun=True
                ).universal_thermal_climate_index
            )

            dfs.append(df)

        return pd.concat(dfs, axis=1, keys=self.epw_ids)

    def _assign_properties(self) -> None:
        """Assign properties of the EPW objects to @property objects of this object."""
        # run the df method to ensure that the data is cached
        df = self.df.swaplevel(axis=1)

        # get the attribute from the EPW objects
        for prop in VARIABLES.keys():
            setattr(self, prop, df[VARIABLES[prop]])

        # add wet-bulb temperature
        setattr(self, "wet_bulb_temperature", df["Wet Bulb Temperature (C)"])

        # add utci (shaded) and utci (unshaded)
        setattr(
            self,
            "universal_thermal_climate_index_shaded",
            df["Universal Thermal Climate Index [Shaded] (C)"],
        )
        setattr(
            self,
            "universal_thermal_climate_index_unshaded",
            df["Universal Thermal Climate Index [Unshaded] (C)"],
        )
        setattr(
            self,
            "universal_thermal_climate_index_unshaded_nowind",
            df["Universal Thermal Climate Index [Unshaded+NoWind] (C)"],
        )

    def _get_collections(self, variable: str) -> list[HourlyContinuousCollection]:
        """Get a list of collections of a variable from the EPWs."""
        return [getattr(epw, variable) for epw in self.epws]

    def _get_series(self, variable: str) -> list[pd.Series]:
        """Get a list of series of a variable from the EPWs."""
        if variable not in VARIABLES.keys():
            raise ValueError(
                f"{variable} is not a valid variable. Must be one of {VARIABLES.keys()}"
            )
        df = self.df.swaplevel(axis=1)

        if variable == "wet_bulb_temperature":
            return df["Wet Bulb Temperature (C)"]
        if variable == "universal_thermal_climate_index_shaded":
            return df["Universal Thermal Climate Index [Shaded] (C)"]
        if variable == "universal_thermal_climate_index_unshaded":
            return df["Universal Thermal Climate Index [Unshaded] (C)"]
        if variable == "universal_thermal_climate_index_unshaded_nowind":
            return df["Universal Thermal Climate Index [Unshaded+NoWind] (C)"]

        return df[header_to_string(getattr(self.epws[0], variable).header)]

    def _get_header(self, variable: str) -> Header:
        """Get the header of a variable from the EPWs."""
        if variable == "wet_bulb_temperature":
            return header_from_string("Wet Bulb Temperature (C)")
        return getattr(self.epws[0], variable).header

    def average_collection(self, variable: str) -> HourlyContinuousCollection:
        """Get the average collection of a variable from the EPWs."""
        return average_collection(self._get_collections(variable=variable))

    def statistics(self, variable: str) -> pd.DataFrame:
        """Get statistics for a variable from the EPW data.

        Args:
            variable: The variable to get statistics for.

        Returns:
            DataFrame: A DataFrame of statistics for the variable.

        """
        return self._get_series(variable=variable).describe().T

    def date_range(self) -> pd.DataFrame:
        """Find the range of dates the set of EPWs span"""
        return self._get_series("years").agg(["min", "max"]).T

    def summary(self, variable: str) -> pd.DataFrame:
        """Get a summary of a variable from the EPW data.

        Args:
            variable: The variable to get a summary for.

        Returns:
            DataFrame: A DataFrame of the summary for the variable.

        """
        if variable not in VARIABLES.keys():
            raise ValueError(
                f"{variable} is not a valid variable. Must be one of {VARIABLES.keys()}"
            )

        summaries = []
        for collection, epw_id in zip(*[self._get_collections(variable=variable), self.epw_ids]):
            summaries.append([epw_id] + summarise_collection(collection))

        return summaries

    def compare_against(
        self, variables: list[str], comparable_data: pd.DataFrame, formatted: bool = True
    ) -> pd.DataFrame | pd.io.formats.style.Styler:
        """Determine how the EPW data compares against a comparable series."""

        # VALIDATION #
        if not isinstance(variables, list):
            raise ValueError("variables must be a list")
        if not isinstance(comparable_data, pd.DataFrame):
            raise ValueError("comparable_data must be a pandas DataFrame")
        if not comparable_data.index.equals(self._get_series(variable=variables[0]).index):
            raise ValueError("comparable_data must have the same index as the EPW data")
        if len(variables) != len(comparable_data.columns):
            raise ValueError(
                "comparable_data must have the same number of columns as the number of variables requested"
            )

        xy = []
        for variable, (_, comparable_series) in zip(*[variables, comparable_data.items()]):
            # get single variable data
            data = self._get_series(variable=variable)
            header = self._get_header(variable=variable)

            # check indices match
            if not comparable_series.index.equals(data.index):
                raise ValueError("comparable_data must have the same index as the EPW data")

            # if wind_direction, then shift to -180 to 180, to avoid weird 359-1 jumps
            if variable == "wind_direction":
                data = data - 180
                comparable_series = comparable_series - 180

            # create temp dataframe with epw and comparable series, and correlate
            temp = (
                pd.concat([data, comparable_series], axis=1)
                .corr(method="kendall")
                .iloc[-1]
                .iloc[:-1]
                .rename(header_to_string(header))
            )

            # create correlation matrix
            xy.append(temp)

        df = pd.concat(xy, axis=1).T

        if formatted:
            return df.style.highlight_max(axis=1, props="color:red")

        return df

    def degree_time(
        self, heat_base: float = 18, cool_base: float = 23, return_type: str = "days"
    ) -> pd.DataFrame:
        """Create a summary of degree time for the EPW data."""
        return degree_time(
            self.epws, heat_base=heat_base, cool_base=cool_base, return_type=return_type
        )

    def diurnal_annual_monthly(
        self,
        variable: str,
        agg: str = "mean",
        ax: Axes = None,
        comparable_series: pd.Series = None,
        show_legend: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot the EPW data as a histogram, with each month in a line.

        Args:
            variable (str):
                The variable to plot.
            ax (Axes, optional):
                The matplotlib axes to plot on. If None, the current axes are used.
            comparable_series (pd.Series, optional):
                A comparable series to plot on the same axes.
            **kwargs:
                Additional keyword arguments to pass to the matplotlib plot method.

        Returns:
            Axes: The matplotlib axes object.
        """

        if variable == "wind_direction":
            raise ValueError(
                "wind_direction comparison is not supported for diurnal_annual_monthly plots"
            )

        if ax is None:
            ax = plt.gca()

        data = self._get_series(variable=variable)

        colors = kwargs.pop(
            "colors", plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(data.columns)]
        )
        group = data.groupby([data.index.month, data.index.hour])
        target_idx = pd.MultiIndex.from_product([range(1, 13, 1), range(24)])
        major_ticks = range(len(target_idx))[::12]
        minor_ticks = range(len(target_idx))[::6]
        major_ticklabels = []
        for i in target_idx:
            if i[1] == 0:
                major_ticklabels.append(f"{calendar.month_abbr[i[0]]}")
            elif i[1] == 12:
                major_ticklabels.append("")

        aggregated = group.agg(agg)

        # create df for re-indexing
        df = aggregated.reindex(target_idx)
        # populate plot
        for n, i in enumerate(range(len(df) + 1)[::24]):
            if n == len(range(len(df) + 1)[::24]) - 1:
                continue
            for col, clr in zip(*[df.columns, colors]):
                ax.plot(
                    range(len(df) + 1)[i : i + 25],
                    (df[col].tolist() + [df[col].values[0]])[i : i + 24]
                    + [(df[col].tolist() + [df[col].values[0]])[i : i + 24][0]],
                    c=clr,
                    ls="-",
                    lw=1,
                    label=col if n == 0 else "_nolegend_",
                )

        # add comparable series, if present
        if comparable_series is not None:
            if not isinstance(comparable_series, pd.Series):
                raise ValueError("comparable_series must be a pandas Series")
            if not comparable_series.index.equals(data.index):
                raise ValueError("comparable_series must have the same index as the EPW data")
            cgroup = comparable_series.groupby(
                [comparable_series.index.month, comparable_series.index.hour]
            )
            cgmean = cgroup.mean().reindex(target_idx).to_frame()
            for n, i in enumerate(range(len(cgmean) + 1)[::24]):
                if n == len(range(len(cgmean) + 1)[::24]) - 1:
                    continue
                for col, clr in zip(*[cgmean.columns, ["k"]]):
                    ax.plot(
                        range(len(cgmean) + 1)[i : i + 25],
                        (cgmean[col].tolist() + [cgmean[col].values[0]])[i : i + 24]
                        + [(cgmean[col].tolist() + [cgmean[col].values[0]])[i : i + 24][0]],
                        c=clr,
                        ls="-",
                        lw=1,
                        label=col if n == 0 else "_nolegend_",
                    )

        # format axes
        ax.set_xlim(0, len(df))
        ax.xaxis.set_major_locator(mtick.FixedLocator(major_ticks))
        ax.xaxis.set_minor_locator(mtick.FixedLocator(minor_ticks))
        ax.set_xticklabels(
            major_ticklabels,
            minor=False,
            ha="left",
        )
        if show_legend:
            ax.legend(
                bbox_to_anchor=(1, 1),
                loc="upper left",
                ncol=1,
                borderaxespad=0,
            )
        ax.set_title(f"{agg.title()}")
        ax.set_ylabel(VARIABLES[variable])

        return ax

    def histogram(
        self,
        variable: str,
        overlap: bool = False,
        comparable_series: pd.Series = None,
        ax: Axes = None,
        show_legend: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot a histogram of a variable from the EPW data.

        Args:
            variable: The variable to plot.
            overlap: If True, the histograms are overlapped. If False, they are arranged next to each other.
            ax: The matplotlib axes to plot on. If None, the current axes are used.
            comparable_series: A comparable series to plot on the same axes.
            **kwargs: Additional keyword arguments to pass to the matplotlib hist method.

        Returns:
            Axes: The matplotlib axes object.

        """

        if ax is None:
            ax = plt.gca()

        data = self._get_series(variable=variable)

        if comparable_series is not None:
            if not isinstance(comparable_series, pd.Series):
                raise ValueError("comparable_series must be a pandas Series")
            if not comparable_series.index.equals(data.index):
                raise ValueError("comparable_series must have the same index as the EPW data")
            data = pd.concat([data, comparable_series], axis=1)

        vmin = kwargs.pop("vmin", data.min().min())
        vmax = kwargs.pop("vmax", data.max().max())

        color = kwargs.pop("color", None)
        bins = kwargs.pop("bins", np.linspace(vmin, vmax, 10))
        if isinstance(bins, int):
            bins = np.linspace(vmin, vmax, bins)
        density = kwargs.pop("density", False)

        if color is None:
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(self.epws)]
        if len(color) < len(self.epws):
            raise ValueError("color must have a color for each EPW")
        if comparable_series is not None:
            color.append("k")

        if isinstance(bins, (list, tuple, np.ndarray)):
            xlim = min(bins), max(bins)
        else:
            xlim = min(data.min()), max(data.max())

        if overlap:
            for n, _ in enumerate(self.epws):
                ax.hist(
                    data.T.values[n],
                    label=data.columns[n],
                    color=color[n],
                    bins=bins,
                    density=density,
                    **kwargs,
                )
        else:
            _, edges, _ = ax.hist(
                data.values, label=data.columns, color=color, bins=bins, density=density, **kwargs
            )
            ax.set_xticks(edges)

        # format x axis
        ax.xaxis.set_major_locator(mtick.AutoLocator())

        ax.set_xlim(xlim)
        ax.set_xlabel(header_to_string(getattr(self.epws[0], variable).header))
        ax.set_ylabel("Frequency" if density else "Count")
        if density:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        if show_legend:
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
        return ax

    def line(
        self,
        variable: str,
        ax: Axes = None,
        comparable_series: pd.Series = None,
        show_legend: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot a line plot of a variable from the EPW data.

        Args:
            variable: The variable to plot.
            ax: The matplotlib axes to plot on. If None, the current axes are used.
            comparable_series: A comparable series to plot on the same axes.
            **kwargs: Additional keyword arguments to pass to the matplotlib plot method.

        Returns:
            Axes: The matplotlib axes object.

        """

        if ax is None:
            ax = plt.gca()

        data = self._get_series(variable=variable)

        color = kwargs.pop("color", None)
        if color is None:
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(self.epws)]
        if len(color) < len(self.epws):
            raise ValueError("color must have a color for each EPW")

        ax.plot(data.index, data.values, label=data.columns, **kwargs)
        # data.plot(ax=ax, **kwargs)

        if comparable_series is not None:
            if not isinstance(comparable_series, pd.Series):
                raise ValueError("comparable_series must be a pandas Series")
            if not comparable_series.index.equals(data.index):
                raise ValueError("comparable_series must have the same index as the EPW data")
            ax.plot(
                comparable_series.index,
                comparable_series.values,
                c="k",
                ls="-",
                lw=0.5,
                # alpha=0.5,
                label=comparable_series.name,
            )

        ax.set_xlabel(None)
        ax.set_xlim(data.index.min(), data.index.max())
        ax.set_ylabel(header_to_string(getattr(self.epws[0], variable).header))
        if show_legend:
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
        return ax

    def diurnal(
        self,
        variable: str,
        month: int,
        ax: Axes = None,
        agg: str = "mean",
        comparable_series: pd.Series = None,
        show_legend: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot a diurnal plot of a variable from the EPW data."""

        if month < 1 or month > 12:
            raise ValueError("month must be between 1 and 12")

        if variable == "wind_direction":
            raise ValueError("wind_direction comparison is not supported for diurnal plots")

        if ax is None:
            ax = plt.gca()

        data = self._get_series(variable=variable)
        data = data.loc[data.index.month == month]
        data = data.groupby(data.index.hour).agg(agg)
        data.loc[len(data)] = data.loc[0].values

        color = kwargs.pop("colors", None)
        if color is None:
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(data)]
        if len(color) < len(self.epws):
            raise ValueError("color must have a color for each EPW")

        for n, (k, v) in enumerate(data.items()):
            ax.plot(v.index, v.values, color=color[n], label=k, **kwargs)

        if comparable_series is not None:
            if not isinstance(comparable_series, pd.Series):
                raise ValueError("comparable_series must be a pandas Series")
            if not comparable_series.index.equals(data.index):
                raise ValueError("comparable_series must have the same index as the EPW data")
            ax.plot(
                comparable_series.index,
                comparable_series.values,
                c="k",
                ls="-",
                lw=2,
                alpha=0.5,
                label=comparable_series.name,
            )

        ax.set_xlabel("Hour of Day")
        ax.set_ylabel(VARIABLES[variable])
        ax.set_title(f'"{agg.title()}" across all days in {calendar.month_name[month]}')
        if show_legend:
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
        return ax


def process_epws(epw_files: list[str], target_directory: Path) -> EPWComparison:
    """Process a list of EPW files into an EPWComparison object.

    Args:
        epw_files: A list of EPW file paths.

    Returns:
        EPWComparison: An EPWComparison object.

    """

    epws = [EPW(epw_file) for epw_file in epw_files]
    ecomp = EPWComparison(epws)
    cfg = SummariseClimateConfig()

    # create the target directory if it does not exist
    target_directory.mkdir(parents=True, exist_ok=True)

    # process each EPW file
    for epw in epws:
        pass
        # CONSOLE_LOGGER.info(f"Processing {Path(epw.file_path).name}")
        # sc = SummariseClimate.from_epw(
        #     epw, target_directory=target_directory / Path(epw.file_path).stem
        # )
        # ov = False
        # # frun ec methods to precalcualte typologyes
        # sc._default_external_comforts()
        # # run plotting methods
        # sc.plot_windroses(overwrite=ov)
        # sc.plot_windmatrices(overwrite=ov)
        # sc.plot_evaporative_cooling_potential(overwrite=ov)
        # sc.plot_sunriseset(overwrite=ov)
        # sc.plot_seasonality(overwrite=ov)
        # sc.plot_radiationrose(overwrite=ov)
        # sc.plot_radiationmatrix(overwrite=ov)
        # sc.plot_utci_shadebenefit(overwrite=ov)
        # sc.plot_material_temperatures(overwrite=ov)
        # sc.plot_utci_typologies(overwrite=ov)
        # sc.plot_utci_limits(overwrite=ov)
        # sc.plot_diurnals(overwrite=ov)
        # sc.plot_utci_shadebenefit(overwrite=ov)
        # sc.plot_utci_typologies(overwrite=ov)
        # sc.plot_utci_limits(overwrite=ov)
        # sc.plot_other_comfort_metrics(overwrite=ov)

    # create geojson of EPW locations
    gis_dir = target_directory
    lats = []
    longs = []
    enames = []
    dnames = []
    tzs = []
    for fp in epw_files:
        eep = EPW(fp)
        loc = eep.location
        lats.append(loc.latitude)
        longs.append(loc.longitude)
        enames.append(Path(fp).stem)
        dnames.append(str(eep))
        tzs.append(loc.time_zone)
    CONSOLE_LOGGER.info(f"Creating {gis_dir / 'epw_files.geojson'}")
    gdf = pd.DataFrame({"Latitude": lats, "Longitude": longs, "EPWName": enames, "LBName": dnames})
    gdf = gpd.GeoDataFrame(
        gdf, geometry=gpd.points_from_xy(gdf.Longitude, gdf.Latitude), crs="EPSG:4326"
    )
    gdf.to_file(gis_dir / "epw_locations.geojson", driver="GeoJSON")

    # create comparison subdir
    comparison_dir = target_directory / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # create comparison charts
    for var in [
        "dry_bulb_temperature",
        "relative_humidity",
        "wind_speed",
        "global_horizontal_radiation",
    ]:

        # get the header
        header: Header = getattr(ecomp.epws[0], var).header

        # histogram
        sp = comparison_dir / f"{header.data_type.abbreviation}_histogram.png"
        if sp.exists():
            CONSOLE_LOGGER.info(f"- {sp} already exists")
            pass
        else:
            CONSOLE_LOGGER.info(f"+ Creating {sp}")
            fig, ax = plt.subplots(1, 1, figsize=cfg.evaporativecoolingpotential_figsize)
            if var == "global_horizontal_radiation":
                ecomp.histogram(variable=var, ax=ax, overlap=False, bins=30, vmin=0.01)
            else:
                ecomp.histogram(variable=var, ax=ax, overlap=False, bins=30)
            # create title string
            title_str = f"Comparison between EPWs for {header.data_type.name} ({header.unit})"
            ax.set_title(title_str)
            plt.tight_layout()
            plt.savefig(sp, dpi=cfg.dpi, transparent=True)
            plt.close("all")

        # diurnal_annual_monthly
        sp = comparison_dir / f"{header.data_type.abbreviation}_diurnal_XX.png"
        if sp.exists():
            CONSOLE_LOGGER.info(f"- {sp} already exists")
            pass
        else:
            CONSOLE_LOGGER.info(f"+ Creating {sp}")
            fig, ax = plt.subplots(1, 1, figsize=cfg.evaporativecoolingpotential_figsize)
            ecomp.diurnal_annual_monthly(variable=var, ax=ax)
            # create title string
            title_str = f"Comparison between EPWs for {header.data_type.name} ({header.unit})"
            ax.set_title(title_str)
            plt.tight_layout()
            plt.savefig(sp, dpi=cfg.dpi, transparent=True)
            plt.close("all")

        # line
        sp = comparison_dir / f"{header.data_type.abbreviation}_line.png"
        if sp.exists():
            CONSOLE_LOGGER.info(f"- {sp} already exists")
            pass
        else:
            CONSOLE_LOGGER.info(f"+ Creating {sp}")
            fig, ax = plt.subplots(1, 1, figsize=cfg.evaporativecoolingpotential_figsize)
            ecomp.line(variable=var, ax=ax)
            # create title string
            title_str = f"Comparison between EPWs for {header.data_type.name} ({header.unit})"
            ax.set_title(title_str)
            plt.tight_layout()
            plt.savefig(sp, dpi=cfg.dpi, transparent=True)
            plt.close("all")

        # diurnal
        for month in range(1, 13, 1):
            sp = comparison_dir / f"{header.data_type.abbreviation}_diurnal_{month:02d}.png"
            if sp.exists():
                CONSOLE_LOGGER.info(f"- {sp} already exists")
                pass
            else:
                CONSOLE_LOGGER.info(f"+ Creating {sp}")
                fig, ax = plt.subplots(1, 1, figsize=cfg.evaporativecoolingpotential_figsize)
                ecomp.diurnal(variable=var, ax=ax, month=month)
                # create title string
                title_str = f"Comparison between EPWs for {header.data_type.name} ({header.unit}), in {calendar.month_name[month]}"
                ax.set_title(title_str)
                plt.tight_layout()
                plt.savefig(sp, dpi=cfg.dpi, transparent=True)
                plt.close("all")

    return ecomp
