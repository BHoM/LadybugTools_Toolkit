"""Methods for handling solar radiation."""

# pylint: disable=E0401
import itertools
import textwrap
from datetime import datetime
import json
import concurrent
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

# pylint: enable=E0401

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from ladybug.wea import EPW, AnalysisPeriod, Wea, Location
from matplotlib import pyplot as plt
from tqdm import tqdm

from .helpers import (
    OpenMeteoVariable,
    angle_from_north,
    angle_to_vector,
    cardinality,
    circular_weighted_mean,
    rolling_window,
    scrape_meteostat,
    scrape_openmeteo,
    wind_speed_at_height,
    remove_leap_days,
)
from .bhom.analytics import bhom_analytics
from .bhom.logging import CONSOLE_LOGGER
from .ladybug_extension.analysisperiod import (
    analysis_period_to_boolean,
    analysis_period_to_datetimes,
    describe_analysis_period,
)
from .ladybug_extension.location import location_to_string
from ladybug.sunpath import Sunpath, Sun
from .ladybug_extension.datacollection import header_to_string

from .plot.utilities import contrasting_color, format_polar_plot


class IrradianceType(Enum):
    """Irradiance types."""

    TOTAL = auto()
    DIRECT = auto()
    DIFFUSE = auto()
    REFLECTED = auto()

    def to_string(self) -> str:
        """Get the string representation of the IrradianceType."""
        return self.name.title()


@dataclass(init=True, repr=True, eq=True)
class Solar:
    """An object to handle solar radiation."""

    global_horizontal_irradiance: list[float]
    direct_normal_irradiance: list[float]
    diffuse_horizontal_irradiance: list[float]
    datetimes: list[datetime] | pd.DatetimeIndex
    source: str = None

    def __post_init__(self):
        if not (
            len(self.global_horizontal_irradiance)
            == len(self.direct_normal_irradiance)
            == len(self.diffuse_horizontal_irradiance)
            == len(self.datetimes)
        ):
            raise ValueError(
                "global_horizontal_irradiance, direct_normal_irradiance, diffuse_horizontal_irradiance and datetimes must be the same length."
            )

        if len(self.global_horizontal_irradiance) <= 1:
            raise ValueError(
                "global_horizontal_irradiance, direct_normal_irradiance, diffuse_horizontal_irradiance and datetimes must be at least 2 items long."
            )

        if len(set(self.datetimes)) != len(self.datetimes):
            raise ValueError("datetimes contains duplicates.")

        # convert to lists
        self.global_horizontal_irradiance = np.array(self.global_horizontal_irradiance)
        self.direct_normal_irradiance = np.array(self.direct_normal_irradiance)
        self.diffuse_horizontal_irradiance = np.array(
            self.diffuse_horizontal_irradiance
        )
        self.datetimes = pd.DatetimeIndex(self.datetimes)

        # validate
        if np.any(np.isnan(self.global_horizontal_irradiance)):
            raise ValueError("global_horizontal_irradiance contains null values.")

        if np.any(np.isnan(self.direct_normal_irradiance)):
            raise ValueError("direct_normal_irradiance contains null values.")

        if np.any(np.isnan(self.diffuse_horizontal_irradiance)):
            raise ValueError("diffuse_horizontal_irradiance contains null values.")

        if np.any(self.global_horizontal_irradiance < 0):
            raise ValueError("global_horizontal_irradiance must be >= 0")

        if np.any(self.direct_normal_irradiance < 0):
            raise ValueError("direct_normal_irradiance must be >= 0")

        if np.any(self.diffuse_horizontal_irradiance < 0):
            raise ValueError("diffuse_horizontal_irradiance must be >= 0")

    def __len__(self) -> int:
        return len(self.datetimes)

    def __repr__(self) -> str:
        """The printable representation of the given object"""
        if self.source:
            return f"{self.__class__.__name__} from {self.source}"

        return (
            f"{self.__class__.__name__}({min(self.datetimes):%Y-%m-%d} to "
            f"{max(self.datetimes):%Y-%m-%d}, n={len(self.datetimes)}, "
            "NO SOURCE"
        )

    def __str__(self) -> str:
        """The string representation of the given object"""
        return self.__repr__()

    #################
    # CLASS METHODS #
    #################

    def to_dict(self) -> dict:
        """Return the object as a dictionary."""

        return {
            "_t": "BH.oM.LadybugTools.Solar",
            "global_horizontal_irradiance": [
                float(i) for i in self.global_horizontal_irradiance
            ],
            "direct_normal_irradiance": [
                float(i) for i in self.direct_normal_irradiance
            ],
            "diffuse_horizontal_irradiance": [
                i for i in self.diffuse_horizontal_irradiance
            ],
            "datetimes": [i.isoformat() for i in self.datetimes],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Solar":
        """Create this object from a dictionary."""

        return cls(
            global_horizontal_irradiance=d["global_horizontal_irradiance"],
            direct_normal_irradiance=d["direct_normal_irradiance"],
            diffuse_horizontal_irradiance=d["diffuse_horizontal_irradiance"],
            datetimes=pd.to_datetime(d["datetimes"]),
            source=d["source"],
        )

    def to_json(self) -> str:
        """Convert this object to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_string: str) -> "Solar":
        """Create this object from a JSON string."""

        return cls.from_dict(json.loads(json_string))

    def to_file(self, path: Path) -> Path:
        """Convert this object to a JSON file."""

        if Path(path).suffix != ".json":
            raise ValueError("path must be a JSON file.")

        with open(Path(path), "w") as fp:
            fp.write(self.to_json())

        return Path(path)

    @classmethod
    def from_file(cls, path: Path) -> "Solar":
        """Create this object from a JSON file."""
        with open(Path(path), "r") as fp:
            return cls.from_json(fp.read())

    def to_csv(self, path: Path) -> Path:
        """Save this object as a csv file.

        Args:
            path (Path):
                The path containing the CSV file.

        Returns:
            Path:
                The resultant CSV file.
        """
        csv_path = Path(path)
        self.df.to_csv(csv_path)
        return csv_path

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        global_horizontal_irradiance_column: Any,
        direct_normal_irradiance_column: Any,
        diffuse_horizontal_irradiance_columns: Any,
        source: str = "DataFrame",
    ) -> "Solar":
        """Create a Solar object from a Pandas DataFrame, with global horizontal irradiance, direct normal irradiance and diffuse horizontal irradiance columns.

        Args:
            df (pd.DataFrame):
                A DataFrame object containing speed and direction columns, and a datetime index.
            global_horizontal_irradiance_column (str):
                The name of the column where global horizontal irradiance data exists.
            direct_normal_irradiance_column (str):
                The name of the column where direct normal irradiance data exists.
            diffuse_horizontal_irradiance_columns (str):
                The name of the column where diffuse horizontal irradiance data exists.
            source (str, optional):
                The source of the data. Defaults to "DataFrame".

        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"df must be of type {type(pd.DataFrame)}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"The DataFrame's index must be of type {type(pd.DatetimeIndex)}"
            )

        # remove NaN values
        df.dropna(axis=0, how="any", inplace=True)

        # remove duplicates in input dataframe
        df = df.loc[~df.index.duplicated()]

        return cls(
            global_horizontal_irradiance=df[
                global_horizontal_irradiance_column
            ].tolist(),
            direct_normal_irradiance=df[direct_normal_irradiance_column].tolist(),
            diffuse_horizontal_irradiance=df[
                diffuse_horizontal_irradiance_columns
            ].tolist(),
            datetimes=df.index.tolist(),
            source=source,
        )

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        global_horizontal_irradiance_column: Any,
        direct_normal_irradiance_column: Any,
        diffuse_horizontal_irradiance_columns: Any,
        **kwargs,
    ) -> "Solar":
        """Create a Wind object from a csv containing wind speed and direction columns.

        Args:
            csv_path (Path):
                The path to the CSV file containing speed and direction columns, and a datetime index.
            global_horizontal_irradiance_column (str):
                The name of the column where global horizontal irradiance data exists.
            direct_normal_irradiance_column (str):
                The name of the column where direct normal irradiance data exists.
            diffuse_horizontal_irradiance_columns (str):
                The name of the column where diffuse horizontal irradiance data exists.
            **kwargs:
                Additional keyword arguments passed to pd.read_csv.
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path, **kwargs)
        return cls.from_dataframe(
            df,
            global_horizontal_irradiance=df[
                global_horizontal_irradiance_column
            ].tolist(),
            direct_normal_irradiance=df[direct_normal_irradiance_column].tolist(),
            diffuse_horizontal_irradiance=df[
                diffuse_horizontal_irradiance_columns
            ].tolist(),
            datetimes=df.index.tolist(),
            source=csv_path.name,
        )

    @classmethod
    def from_epw(cls, epw: Path | EPW) -> "Solar":
        """Create a Solar object from an EPW file or object.

        Args:
            epw (Path | EPW):
                The path to the EPW file, or an EPW object.
        """

        if isinstance(epw, (str, Path)):
            source = Path(epw).name
            epw = EPW(epw)
        else:
            source = Path(epw.file_path).name

        return cls(
            global_horizontal_irradiance=epw.global_horizontal_radiation.values,
            direct_normal_irradiance=epw.direct_normal_radiation.values,
            diffuse_horizontal_irradiance=epw.diffuse_horizontal_radiation.values,
            datetimes=analysis_period_to_datetimes(AnalysisPeriod()),
            source=source,
        )

    @classmethod
    def from_openmeteo(
        cls,
        latitude: float,
        longitude: float,
        start_date: datetime | str,
        end_date: datetime | str,
    ) -> "Solar":
        """Create a Solar object from data obtained from the Open-Meteo database of historic weather station data.

        Args:
            latitude (float):
                The latitude of the target site, in degrees.
            longitude (float):
                The longitude of the target site, in degrees.
            start_date (datetime | str):
                The start-date from which records will be obtained.
            end_date (datetime | str):
                The end-date beyond which records will be ignored.
        """

        df = scrape_openmeteo(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            variables=[
                OpenMeteoVariable.SHORTWAVE_RADIATION,
                OpenMeteoVariable.DIRECT_NORMAL_IRRADIANCE,
                OpenMeteoVariable.DIFFUSE_RADIATION,
            ],
            convert_units=True,
        )

        df.dropna(how="any", axis=0, inplace=True)

        global_horizontal_irradiance = df[
            "Global Horizontal Radiation (Wh/m2)"
        ].tolist()
        direct_normal_irradiance = df["Direct Normal Radiation (Wh/m2)"].tolist()
        diffuse_horizontal_irradiance = df[
            "Diffuse Horizontal Radiation (Wh/m2)"
        ].tolist()

        if (
            len(global_horizontal_irradiance) == 0
            or len(direct_normal_irradiance) == 0
            or len(diffuse_horizontal_irradiance) == 0
        ):
            raise ValueError(
                "OpenMeteo did not return any data for the given latitude, longitude and start/end dates."
            )
        datetimes = df.index.tolist()

        return cls(
            global_horizontal_irradiance=global_horizontal_irradiance,
            direct_normal_irradiance=direct_normal_irradiance,
            diffuse_horizontal_irradiance=diffuse_horizontal_irradiance,
            datetimes=datetimes,
            source="OpenMeteo",
        )

    @property
    def df(self) -> pd.DataFrame:
        """Get a dataframe of the data."""
        return pd.concat([self.ghi, self.dni, self.dhi], axis=1)

    @property
    def ghi(self) -> pd.Series:
        """Get the global horizontal irradiance."""
        return pd.Series(
            self.global_horizontal_irradiance,
            index=self.datetimes,
            name="Global Horizontal Irradiance (Wh/m2)",
        )

    @property
    def dni(self) -> pd.Series:
        """Get the direct normal irradiance."""
        return pd.Series(
            self.direct_normal_irradiance,
            index=self.datetimes,
            name="Direct Normal Irradiance (Wh/m2)",
        )

    @property
    def dhi(self) -> pd.Series:
        """Get the diffuse horizontal irradiance."""
        return pd.Series(
            self.diffuse_horizontal_irradiance,
            index=self.datetimes,
            name="Diffuse Horizontal Irradiance (Wh/m2)",
        )

    def suns(self, location: Location) -> list[Sun]:
        """Create a list f suns representing position for the given location.

        Args:
            location (Location):
                The location of the site.

        Returns:
            list[Sun]:
                A list of Sun objects.
        """
        sp = Sunpath.from_location(location=location)
        return [sp.calculate_sun_from_date_time(dt) for dt in self.datetimes]

    def wea(self, location: Location) -> Wea:
        """Create a Wea object from this Solar object.

        Args:
            location (Location):
                The location of the site.

        Returns:
            Wea:
                A Wea object.
        """

        # create annual values
        if max(self.datetimes) - min(self.datetimes) < pd.Timedelta(
            days=365
        ) - pd.Timedelta(minutes=60):
            raise ValueError(
                "The Solar object must contain at least 1 year's worth of data to generate a Wea."
            )

        df = remove_leap_days(self.df)

        grouped = df.groupby([df.index.month, df.index.day, df.index.hour]).mean()
        index = pd.date_range("2017-01-01", periods=8760, freq="60T")
        grouped.set_index(index, inplace=True)

        return Wea.from_annual_values(
            location=location,
            direct_normal_irradiance=grouped[
                "Direct Normal Irradiance (Wh/m2)"
            ].tolist(),
            diffuse_horizontal_irradiance=grouped[
                "Diffuse Horizontal Irradiance (Wh/m2)"
            ].tolist(),
        )

    @staticmethod
    def altitudes(n: int) -> list[float]:
        """Get a list of altitudes."""
        if not isinstance(n, int):
            raise ValueError("n_altitudes must be an integer.")
        if n < 3:
            raise ValueError("n_altitudes must be an integer >= 3.")
        return np.linspace(0, 90, n).tolist()

    @staticmethod
    def azimuths(n: int) -> list[float]:
        """Get a list of azimuths."""
        if not isinstance(n, int):
            raise ValueError("n_azimuths must be an integer.")
        if n < 3:
            raise ValueError("n_azimuths must be an integer >= 3.")
        return np.linspace(0, 360, n).tolist()

    def directional_irradiance_matrix(
        self,
        location: Location,
        altitude: float | int = 0,
        directions: int = 32,
        ground_reflectance: float = 0.2,
        isotropic: bool = True,
        reload: bool = True,
    ) -> pd.DataFrame:
        """Calculate the irradiance in W/m2 for each direction, for the given altitude.

        Args:
            location (Location):
                The location of the site.
            altitude (float, optional):
                The altitude of the facade. Defaults to 0 for a vertical surface.
            directions (int, optional):
                The number of directions to calculate. Defaults to 32.
            ground_reflectance (float, optional):
                The ground reflectance. Defaults to 0.2.
            isotropic (bool, optional):
                Calculate isotropic diffuse irradiance. Defaults to True.
            reload (bool, optional):
                reload results if they've already been saved.

        Returns:
            pd.DataFrame:
                A dataframe containing a huge amount of information!
        """
        _wea = self.wea(location=location)
        midpoints = np.linspace(0, 360, directions + 1)[:-1]

        _dir = Path(hb_folders.default_simulation_folder) / "_lbt_tk_solar"
        _dir.mkdir(exist_ok=True, parents=True)
        sp = (
            _dir
            / f"{location_to_string(location)}_{ground_reflectance}_{isotropic}_{directions}_{altitude}.h5"
        )
        if reload and sp.exists():
            CONSOLE_LOGGER.info(f"Loading results from {sp}.")
            return pd.read_hdf(sp, key="df")

        # get headers
        cols = _wea.directional_irradiance(0, 0)
        units = [header_to_string(i.header) for i in cols]
        idx = analysis_period_to_datetimes(cols[0].header.analysis_period)

        results = []
        pbar = tqdm(total=len(midpoints), desc="Calculating irradiance")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for _az in midpoints:
                futures.append(
                    executor.submit(
                        _wea.directional_irradiance,
                        altitude,
                        _az,
                        ground_reflectance,
                        isotropic,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                pbar.update(n=1)
                results.append(future.result())

        # convert results into a massive array
        headers = []
        for _az in midpoints:
            for rad_type, unit in zip(
                *[["Total", "Direct", "Diffuse", "Reflected"], units]
            ):
                headers.append((altitude, _az, rad_type, unit))

        df = pd.DataFrame(
            np.array(results).reshape(len(midpoints) * 4, 8760),
            columns=idx,
            index=pd.MultiIndex.from_tuples(
                headers, names=["Altitude", "Azimuth", "Type", "Unit"]
            ),
        ).T

        df.to_hdf(sp, key="df", complevel=9, complib="blosc:zlib")

        return df

    def detailed_irradiance_matrix(
        self,
        location: Location,
        altitudes: int = 3,
        azimuths: int = 3,
        ground_reflectance: float = 0.2,
        isotropic: bool = True,
        reload: bool = True,
    ) -> pd.DataFrame:
        """Calculate the irradiance in W/m2 for each combination of tilt and orientation.

        Args:
            location (Location):
                The location of the site.
            altitudes (int, optional):
                The number of altitudes to calculate. Defaults to 3.
            azimuths (int, optional):
                The number of azimuths to calculate. Defaults to 3.
            ground_reflectance (float, optional):
                The ground reflectance. Defaults to 0.2.
            isotropic (bool, optional):
                Calculate isotropic diffuse irradiance. Defaults to True.
            reload (bool, optional):
                reload results if they've already been saved.

        Returns:
            pd.DataFrame:
                A dataframe containing a huge amount of information!
        """

        _wea = self.wea(location=location)

        altitudes = self.altitudes(altitudes)
        azimuths = self.azimuths(azimuths)
        combinations = list(itertools.product(altitudes, azimuths))

        _dir = Path(hb_folders.default_simulation_folder) / "_lbt_tk_solar"
        _dir.mkdir(exist_ok=True, parents=True)
        sp = (
            _dir
            / f"{location_to_string(location)}_{ground_reflectance}_{isotropic}_{len(altitudes)}_{len(azimuths)}.h5"
        )

        if reload and sp.exists():
            CONSOLE_LOGGER.info(f"Loading results from {sp}.")
            return pd.read_hdf(sp, key="df")

        # get headers
        cols = _wea.directional_irradiance(0, 0)
        units = [header_to_string(i.header) for i in cols]
        idx = analysis_period_to_datetimes(cols[0].header.analysis_period)

        results = []
        pbar = tqdm(total=len(combinations), desc="Calculating irradiance")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for _alt, _az in combinations:
                futures.append(
                    executor.submit(
                        _wea.directional_irradiance,
                        _alt,
                        _az,
                        ground_reflectance,
                        isotropic,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                pbar.update(n=1)
                results.append(future.result())

        # convert results into a massive array
        headers = []
        for _alt, _az in combinations:
            for rad_type, unit in zip(
                *[["Total", "Direct", "Diffuse", "Reflected"], units]
            ):
                headers.append((_alt, _az, rad_type, unit))

        df = pd.DataFrame(
            np.array(results).reshape(len(combinations) * 4, 8760),
            columns=idx,
            index=pd.MultiIndex.from_tuples(
                headers, names=["Altitude", "Azimuth", "Type", "Unit"]
            ),
        ).T

        df.to_hdf(sp, key="df", complevel=9, complib="blosc:zlib")

        return df

    def plot_tilt_orientation_factor(
        self,
        location: Location,
        ax: plt.Axes = None,
        altitudes: int = 3,
        azimuths: int = 3,
        ground_reflectance: float = 0.2,
        isotropic: bool = True,
        irradiance_type: IrradianceType = IrradianceType.TOTAL,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        agg: str = "sum",
        **kwargs,
    ) -> plt.Axes:
        """Plot a tilt-orientation factor.

        Args:
            location (Location):
                The location of the site.
            ax (plt.Axes, optional):
                The axes to plot on. Defaults to None.
            altitudes (int, optional):
                The number of altitudes to calculate. Defaults to 3.
            azimuths (int, optional):
                The number of azimuths to calculate. Defaults to 3.
            ground_reflectance (float, optional):
                The ground reflectance. Defaults to 0.2.
            isotropic (bool, optional):
                Calculate isotropic diffuse irradiance. Defaults to True.
            irradiance_type (IrradianceType, optional):
                The irradiance type to plot. Defaults to IrradianceType.TOTAL.
            analysis_period (AnalysisPeriod, optional):
                The analysis period. Defaults to AnalysisPeriod().
            agg (str, optional):
                The aggregation method. Defaults to "sum".
            **kwargs:
                Keyword arguments to pass to tricontourf.
        """
        if ax is None:
            ax = plt.gca()

        mtx = (
            self.detailed_irradiance_matrix(
                location=location,
                azimuths=azimuths,
                altitudes=altitudes,
                ground_reflectance=ground_reflectance,
                isotropic=isotropic,
            )
            .filter(regex=irradiance_type.to_string())
            .iloc[analysis_period_to_boolean(analysis_period)]
        ).agg(agg, axis=0) / (1000 if agg == "sum" else 1)

        _max = mtx.max()
        _max_alt, _max_az, _, unit = mtx.idxmax()
        if agg == "sum":
            unit = unit.replace("(W/m2)", "kWh/m$^2$").replace("Irradiance ", "")
        else:
            unit = unit.replace("(W/m2)", "Wh/m$^2$").replace("Irradiance ", "")

        if _max == 0:
            raise ValueError(f"No solar radiation within {analysis_period}.")

        tcf = ax.tricontourf(
            mtx.index.get_level_values("Azimuth"),
            mtx.index.get_level_values("Altitude"),
            mtx.values,
            extend="max",
            cmap=kwargs.pop("cmap", "YlOrRd"),
            levels=kwargs.pop("levels", 51),
            **kwargs,
        )

        quantiles = [0.25, 0.5, 0.75, 0.95]
        quantile_values = mtx.quantile(quantiles).values
        tcl = ax.tricontour(
            mtx.index.get_level_values("Azimuth"),
            mtx.index.get_level_values("Altitude"),
            mtx.values,
            levels=quantile_values,
            colors="k",
            linestyles="--",
            alpha=0.5,
        )

        def cl_fmt(x):
            return f"{x:,.0f}{unit}"

        _ = ax.clabel(tcl, fontsize="small", fmt=cl_fmt)
        ax.scatter(_max_az, _max_alt, c="k", s=10, marker="x")
        alt_offset = (90 / 100) * 0.5 if _max_alt <= 45 else -(90 / 100) * 0.5
        az_offset = (360 / 100) * 0.5 if _max_az <= 180 else -(360 / 100) * 0.5
        ha = "left" if _max_az <= 180 else "right"
        va = "bottom" if _max_alt <= 45 else "top"
        ax.text(
            _max_az + az_offset,
            _max_alt + alt_offset,
            f"{_max:,.0f}{unit}\n({_max_az:0.0f}°, {_max_alt:0.0f}°)",
            ha=ha,
            va=va,
            c="k",
            weight="bold",
            size="small",
        )

        # format plot
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(base=30))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(base=10))

        cb = plt.colorbar(
            tcf,
            ax=ax,
            orientation="vertical",
            drawedges=False,
            fraction=0.05,
            aspect=25,
            pad=0.02,
            label=unit,
        )
        cb.outline.set_visible(False)
        for quantile_val in quantile_values:
            cb.ax.plot([0, 1], [quantile_val] * 2, "k", ls="--", alpha=0.5)

        ax.set_title(
            (
                f"{location_to_string(location)}, from {self.source}\n{irradiance_type.to_string()} "
                f"Irradiance ({agg.upper()})\n{describe_analysis_period(analysis_period)}"
            )
        )
        ax.set_xlabel("Panel orientation (clockwise from North at 0°)")
        ax.set_ylabel("Panel tilt (0° facing the horizon, 90° facing the sky)")
        return ax

    def plot_directional_irradiance(
        self,
        location: Location,
        ax: plt.Axes = None,
        altitude: float | int = 0,
        directions: int = 32,
        ground_reflectance: float = 0.2,
        isotropic: bool = True,
        irradiance_type: IrradianceType = IrradianceType.TOTAL,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        agg: str = "sum",
        labelling: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """Plot the directional irradiance for the given configuration.

        Args:
            location (Location):
                The location of the site.
            ax (plt.Axes, optional):
                The axes to plot on. Defaults to None.
            altitude (int, optional):
                The altitude to calculate. Defaults to 0 for vertical surfaces.
            directions (int, optional):
                The number of directions to calculate. Defaults to 32.
            ground_reflectance (float, optional):
                The ground reflectance. Defaults to 0.2.
            isotropic (bool, optional):
                Calculate isotropic diffuse irradiance. Defaults to True.
            irradiance_type (IrradianceType, optional):
                The irradiance type to plot. Defaults to IrradianceType.TOTAL.
            analysis_period (AnalysisPeriod, optional):
                The analysis period. Defaults to AnalysisPeriod().
            agg (str, optional):
                The aggregation method. Defaults to "sum".
            labelling (bool, optional):
                Label the plot. Defaults to True.
            **kwargs:
                Keyword arguments to pass to tricontourf.

        Returns:
            plt.Axes: The matplotlib axes.
        """

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})

        format_polar_plot(ax)

        cmap = plt.get_cmap(
            kwargs.pop(
                "cmap",
                "Spectral_r",
            )
        )
        width = kwargs.pop("width", 0.9)
        vmin = kwargs.pop("vmin", 0)

        # plot data
        data = self.directional_irradiance_matrix(
            location=location,
            altitude=altitude,
            directions=directions,
            ground_reflectance=ground_reflectance,
            isotropic=isotropic,
        ).filter(regex=irradiance_type.to_string()).iloc[
            analysis_period_to_boolean(analysis_period)
        ].agg(
            agg, axis=0
        ) / (
            1000 if agg == "sum" else 1
        )

        _max = data.max()
        _, _max_az, _, unit = data.idxmax()
        if agg == "sum":
            unit = unit.replace("(W/m2)", "kWh/m$^2$").replace("Irradiance ", "")
        else:
            unit = unit.replace("(W/m2)", "Wh/m$^2$").replace("Irradiance ", "")
        if _max == 0:
            raise ValueError(f"No solar radiation within {analysis_period}.")

        vmax = kwargs.pop("vmax", data.max())
        colors = [cmap(i) for i in np.interp(data.values, [vmin, vmax], [0, 1])]
        thetas = np.deg2rad(data.index.get_level_values("Azimuth"))
        radiis = data.values
        bars = ax.bar(
            thetas,
            radiis,
            zorder=2,
            width=width * np.deg2rad(360 / len(thetas)),
            color=colors,
        )

        # colorbar
        if (data.min() < vmin) and (data.max() > vmax):
            extend = "both"
        elif data.min() < vmin:
            extend = "min"
        elif data.max() > vmax:
            extend = "max"
        else:
            extend = "neither"
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, orientation="vertical", label=unit, extend=extend)
        cb.outline.set_visible(False)

        # labelling
        if labelling:
            for rect, idx, val, colr in list(
                zip(
                    *[bars, data.index.get_level_values("Azimuth"), data.values, colors]
                )
            ):
                if len(set(data.values)) == 1:
                    ax.text(
                        0,
                        0,
                        textwrap.fill(
                            f"{val:,.0f}{unit} {agg} total insolation",
                            16,
                        ),
                        ha="center",
                        va="center",
                        bbox={
                            "ec": "none",
                            "fc": "w",
                            "alpha": 0.5,
                            "boxstyle": "round,pad=0.3",
                        },
                    )
                    break
                if val == data.max():
                    rect.set_edgecolor("k")
                if val > data.max() / 1.5:
                    ax.text(
                        np.deg2rad(idx),
                        val,
                        f" {val:,.0f}{unit} ",
                        rotation_mode="anchor",
                        rotation=(-idx + 90) if idx < 180 else 180 + (-idx + 90),
                        ha="right" if idx < 180 else "left",
                        va="center",
                        fontsize="xx-small",
                        c=contrasting_color(colr)
                        # bbox=dict(ec="none", fc="w", alpha=0.5, boxstyle="round,pad=0.3"),
                    )

        _ = ax.set_title(
            (
                f"{location_to_string(location)}, from {self.source}\n"
                f"{irradiance_type.to_string()} irradiance ({agg.upper()}) at {altitude}° tilt\n"
                f"{describe_analysis_period(analysis_period)}"
            )
        )

        return ax
