"""One wrapper to rule them all, one wrapper to find them, one wrapper to bring them all and in the darkness generate a load of plots into a target directory."""

# region: IMPORTS

import calendar
import json
import tempfile
import textwrap
import warnings
from functools import cached_property
from itertools import cycle
from pathlib import Path
from typing import Optional

import dataframe_image as dfi
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from honeybee.model import Model
from honeybee.typing import clean_string, valid_string
from honeybee_radiance.modifier.material import Plastic
from ladybug.location import Location
from ladybug.psychrometrics import dew_point_from_db_rh, wet_bulb_from_db_rh
from ladybug_comfort.collection.utci import UTCI
from ladybug_comfort.hi import heat_index as heat_index_temperature
from ladybug_comfort.humidex import humidex, humidex_degree_of_comfort
from ladybug_comfort.pet import physiologic_equivalent_temperature
from ladybug_comfort.wbgt import wet_bulb_globe_temperature
from ladybug_geometry.geometry3d.pointvector import Point3D
from ladybugtools_toolkit.categorical.categories import (
    HEAT_INDEX_CATEGORIES,
    HUMIDEX_CATEGORIES,
    UTCI_DEFAULT_CATEGORIES,
    WBGT_CATEGORIES,
    Categorical,
    CategoricalComfort,
    ComfortClass,
)
from matplotlib.colors import Colormap, LinearSegmentedColormap, rgb2hex
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pydantic import BaseModel, Field, validator
from python_toolkit.bhom.logging import CONSOLE_LOGGER
from tqdm import tqdm

from ..external_comfort._externalcomfortbase import ExternalComfort
from ..external_comfort._shelterbase import Shelter
from ..external_comfort.comfort_feasibility import (
    EPW,
    ThermalComfortIndex,
    thermal_comfort_bounds,
    thermal_comfort_data,
    thermal_comfort_datas,
    thermal_comfort_summary,
)
from ..external_comfort.externalcomfort import ExternalComfort
from ..external_comfort.material import Materials
from ..external_comfort.shelter import Shelter, TreeShelter
from ..external_comfort.simulate import SimulationResult
from ..external_comfort.typology import Typologies, Typology
from ..external_comfort.utci import (
    EPW,
    UTCI_DEFAULT_CATEGORIES,
    feasible_utci_category_limits,
    feasible_utci_limits,
)
from ..helpers import (
    OpenMeteoVariable,
    default_analysis_periods,
    default_combined_analysis_periods,
    safe_filename,
    scrape_openmeteo,
    wind_speed_at_height,
)
from ..honeybee_extension.results import load_ill, load_npy, load_pts, load_res, make_annual
from ..honeybee_extension.sri import calculate_sri, material_sri
from ..ladybug_extension.analysisperiod import (
    AnalysisPeriod,
    analysis_period_to_boolean,
    describe_analysis_period,
)
from ..ladybug_extension.datacollection import (
    analysis_period_to_datetimes,
    collection_from_series,
    summarise_collection,
)
from ..ladybug_extension.epw import (
    EPW,
    AnalysisPeriod,
    collection_to_series,
    epw_from_dataframe,
    epw_to_dataframe,
)
from ..ladybug_extension.header import header_from_string
from ..plot._diurnal import diurnal
from ..plot._evaporative_cooling_potential import (
    evaporative_cooling_potential,
    evaporative_cooling_potential_epw,
)
from ..plot._misc import hours_sunrise_sunset
from ..plot._seasonality import EPW, seasonality_comparison
from ..plot._sunpath import sunpath
from ..plot._utci import (
    heatmap,
    shade_benefit_category,
    utci_comfort_band_comparison,
    utci_comfort_band_comparison_series,
    utci_histogram,
    utci_shade_benefit,
)
from ..plot.spatial_heatmap import spatial_heatmap
from ..plot.utilities import colormap_sequential, create_triangulation
from ..solar import radiation_rose, tilt_orientation_factor
from ..wind import Wind
from .config import Config as OutputConfig
from .config import MissingVariableException

humidex_v = np.vectorize(humidex)
wbgt_v = np.vectorize(wet_bulb_globe_temperature)
heat_index_temperature_v = np.vectorize(heat_index_temperature)
pet_v = np.vectorize(physiologic_equivalent_temperature)

wet_bulb_from_db_rh_v = np.vectorize(wet_bulb_from_db_rh)

# endregion: IMPORTS

# TODO - add bulk processing methods here, inherit from coparsion to apply to multiple files in a compariosn, handle folder structures, ect..
# TODO - SAP stuff, in Downloads fiolder...


class SummariseClimate(BaseModel):
    """A class to summarize the contents of a dataframe representing historic climate data."""

    identifier: str = Field(description="A unique identifier for the summary.")
    df: pd.DataFrame = Field(description="The dataframe to summarize.", repr=False)
    target_directory: str | Path = Field(description="The directory to save the summary data into.")
    location: Optional[Location] = Field(description="The location of the data.", repr=False)
    config: Optional[OutputConfig] = Field(
        description="The configuration for the plots.", default=OutputConfig()
    )

    class Config:
        arbitrary_types_allowed = True

    # region: VALIDATION
    @validator("identifier")
    def identifier_validator(cls, v):  # pylint: disable=no-self-argument
        """Validate the identifier, and mutate to be valid."""
        return valid_string(v)

    @validator("df")
    def df_validator(cls, v):  # pylint: disable=no-self-argument
        """Validate the input dataframe."""

        # check that the following essential columns are in the dataframe
        essential_columns = [
            "Dry Bulb Temperature (C)",
        ]
        for col in essential_columns:
            if col not in v.columns:
                raise MissingVariableException(
                    f"The input dataframe must contain a column named '{col}'."
                )
        return v

    @validator("config")
    def config_validator(cls, v):  # pylint: disable=no-self-argument
        """Validate the configuration."""
        if not isinstance(v, OutputConfig):
            raise ValueError("Config must be an instance of OutputConfig.")
        return v

    @validator("target_directory")
    def target_directory_validator(cls, v):  # pylint: disable=no-self-argument
        """Validate the parent directory."""
        dirlength = len(Path(v).resolve().as_posix())
        if dirlength > 200:
            raise ValueError(f"Directory path is too long ({dirlength}).")
        if not Path(v).exists():
            raise ValueError(f"Directory '{v}' does not exist.")
        return v

    # endregion: VALIDATION

    # region: CLASSMETHODS
    @classmethod
    def from_epw(cls, epw: EPW, target_directory: str | Path, config: OutputConfig = None):
        """Create this object from an EPW object."""

        # copy the EPW file to the target directory
        epw_file = Path(target_directory) / f"{Path(epw.file_path).name}"
        epw.save(epw_file)

        # convert the epw to a dataframe
        df = epw_to_dataframe(epw, include_additional=True)

        if config is None:
            obj = cls(
                identifier=clean_string(Path(epw.file_path).stem),
                df=df,
                target_directory=target_directory,
                location=epw.location,
            )
        else:
            obj = cls(
                identifier=clean_string(Path(epw.file_path).stem),
                df=df,
                target_directory=target_directory,
                location=epw.location,
                config=config,
            )

        # return the object
        return obj

    @classmethod
    def from_web(
        cls,
        identifier: str,
        location: Location,
        target_directory: str | Path,
        start: str = "2015-01-01",
        end: str = "2024-06-01",
        config: OutputConfig = None,
    ):
        """Create this object from an online dataset."""

        sp = target_directory / f"era5_{start}_{end}.h5"
        if sp.exists():
            era5_df = pd.read_hdf(sp, "df")
        else:
            variables = [
                OpenMeteoVariable.DEWPOINT_2M,
                OpenMeteoVariable.CLOUDCOVER,
                OpenMeteoVariable.DIFFUSE_RADIATION,
                OpenMeteoVariable.DIRECT_NORMAL_IRRADIANCE,
                OpenMeteoVariable.RELATIVEHUMIDITY_2M,
                OpenMeteoVariable.SHORTWAVE_RADIATION,
                OpenMeteoVariable.SURFACE_PRESSURE,
                OpenMeteoVariable.TEMPERATURE_2M,
                OpenMeteoVariable.WINDDIRECTION_10M,
                OpenMeteoVariable.WINDSPEED_10M,
                OpenMeteoVariable.SOIL_TEMPERATURE_0_TO_7CM,
                OpenMeteoVariable.SOIL_TEMPERATURE_7_TO_28CM,
                OpenMeteoVariable.SOIL_TEMPERATURE_28_TO_100CM,
                OpenMeteoVariable.SOIL_TEMPERATURE_100_TO_255CM,
            ]
            # scrape the data
            tz_offset = int(location.time_zone)
            era5_df = scrape_openmeteo(
                latitude=location.latitude,
                longitude=location.longitude,
                start_date=start,
                end_date=end,
                convert_units=True,
            )
            era5_df = era5_df.shift(tz_offset).dropna()
            era5_df["Wet Bulb Temperature (C)"] = wet_bulb_from_db_rh_v(
                db_temp=era5_df["Dry Bulb Temperature (C)"],
                rel_humid=era5_df["Relative Humidity (%)"],
                b_press=era5_df["Atmospheric Station Pressure (Pa)"],
            )
            era5_df.to_hdf(sp, key="df")

        # return the object
        return cls(
            identifier=identifier,
            df=era5_df,
            target_directory=target_directory,
            location=location,
            config=config,
        )

    # endregion: CLASSMETHODS

    @property
    def epw(self) -> EPW:
        """Convert the dataframe to an EPW object."""

        # find EPW files in the target directory
        _files = list(self.target_directory.glob("*.epw"))
        # if one, then just load that one
        if len(_files) == 1:
            if self.identifier == clean_string(_files[0].stem):
                return EPW(_files[0])
            else:
                raise AssertionError(
                    f"EPW file exists in target directory, but does not match the current objects identifier. Expected '{self.identifier}', got '{_files[0].stem}'."
                )
        # if more than one, then raise an error
        if len(_files) > 1:
            raise AssertionError(
                "Multiple EPW files found in target directory. This is a bit too ambiguous so is being flagged as an error. Remove all but one, or have none."
            )

        # # attempt to create an EPW from the current dataframe
        # if (not self.df.index.is_monotonic_increasing) or (not len(self.df.index) in [8760, 8761]):
        #     raise ValueError("The dataframe must be annual and sorted.")

        # create epw
        epw = epw_from_dataframe(dataframe=self.df, location=self.location)

        # save to file in target directory
        pth = self.target_directory / f"{self.identifier}.epw"
        epw.save(pth)

        return EPW(pth)

    @property
    def epw_file(self) -> Path:
        """The path of the associated EPW file."""
        return Path(self.epw.file_path)

    @property
    def dry_bulb_temperature(self) -> pd.Series:
        """Alias for the dry bulb temperature series."""
        try:
            return self.df["Dry Bulb Temperature (C)"]
        except KeyError as e:
            raise MissingVariableException(
                f"'Dry Bulb Temperature (C)' column not found in dataframe."
            ) from e

    @property
    def dbt(self) -> pd.Series:
        """Alias for the dry bulb temperature series."""
        return self.dry_bulb_temperature

    @property
    def relative_humidity(self) -> pd.Series:
        """Alias for the relative humidity series."""
        try:
            return self.df["Relative Humidity (%)"]
        except KeyError as e:
            raise MissingVariableException(
                f"'Relative Humidity (%)' column not found in dataframe."
            ) from e

    @property
    def rh(self) -> pd.Series:
        """Alias for the relative humidity series."""
        return self.relative_humidity

    @property
    def wet_bulb_temperature(self) -> pd.Series:
        """Alias for the wet bulb temperature series."""
        try:
            return self.df["Wet Bulb Temperature (C)"]
        except KeyError as e:
            raise MissingVariableException(
                f"'Wet Bulb Temperature (C)' column not found in dataframe."
            ) from e

    @property
    def wbt(self) -> pd.Series:
        """Alias for the wet bulb temperature series."""
        return self.wet_bulb_temperature

    @property
    def dew_point_temperature(self) -> pd.Series:
        """Alias for the dew point temperature series."""
        try:
            return self.df["Dew Point Temperature (C)"]
        except KeyError as e:
            raise MissingVariableException(
                f"'Dew Point Temperature (C)' column not found in dataframe."
            ) from e

    @property
    def dpt(self) -> pd.Series:
        """Alias for the dew point temperature series."""
        return self.dew_point_temperature

    @property
    def wind_speed(self) -> pd.Series:
        """Alias for the wind speed series."""
        try:
            return self.df["Wind Speed (m/s)"]
        except KeyError as e:
            raise MissingVariableException(
                f"'Wind Speed (m/s)' column not found in dataframe."
            ) from e

    @property
    def ws(self) -> pd.Series:
        """Alias for the wind speed series."""
        return self.wind_speed

    @property
    def wind_direction(self) -> pd.Series:
        """Alias for the wind direction series."""
        try:
            return self.df["Wind Direction (degrees)"]
        except KeyError as e:
            raise MissingVariableException(
                f"'Wind Direction (degrees)' column not found in dataframe."
            ) from e

    @property
    def wd(self) -> pd.Series:
        """Alias for the wind direction series."""
        return self.wind_direction

    @property
    def global_horizontal_radiation(self) -> pd.Series:
        """Alias for the global horizontal radiation series."""
        try:
            return self.df["Global Horizontal Radiation (Wh/m2)"]
        except KeyError as e:
            raise MissingVariableException(
                f"'Global Horizontal Radiation (Wh/m2)' column not found in dataframe."
            ) from e

    @property
    def ghr(self) -> pd.Series:
        """Alias for the global horizontal radiation series."""
        return self.global_horizontal_radiation

    @property
    def direct_normal_radiation(self) -> pd.Series:
        """Alias for the direct normal radiation series."""
        try:
            return self.df["Direct Normal Radiation (Wh/m2)"]
        except KeyError as e:
            raise MissingVariableException(
                f"'Direct Normal Radiation (Wh/m2)' column not found in dataframe."
            ) from e

    @property
    def dnr(self) -> pd.Series:
        """Alias for the direct normal radiation series."""
        return self.direct_normal_radiation

    @property
    def diffuse_horizontal_radiation(self) -> pd.Series:
        """Alias for the diffuse horizontal radiation series."""
        try:
            return self.df["Diffuse Horizontal Radiation (Wh/m2)"]
        except KeyError as e:
            raise MissingVariableException(
                f"'Diffuse Horizontal Radiation (Wh/m2)' column not found in dataframe."
            ) from e

    @property
    def dhr(self) -> pd.Series:
        """Alias for the diffuse horizontal radiation series."""
        return self.diffuse_horizontal_radiation

    @property
    def wind(self) -> Wind:
        """The wind data object."""

        warnings.warn(
            "Wind data is assumed to be at 10m above the ground. If this is wrong, then the generated outputs will also be wrong."
        )
        return Wind.from_dataframe(
            self.df,
            wind_speed_column=self.wind_speed.name,
            wind_direction_column=self.wind_direction.name,
            height_above_ground=10,
        )

    def variable_lims(self, variable: str) -> tuple[float]:
        """Get sensible limits for the named variable."""

        match variable:
            case "Dry Bulb Temperature (C)":
                t = (np.floor(self.dbt.quantile(0.05)), np.ceil(self.dbt.quantile(0.95)))
            case "Relative Humidity (%)":
                t = (np.floor(self.rh.quantile(0.05)), np.ceil(self.rh.quantile(0.95)))
            case "Wind Speed (m/s)":
                t = (0, np.ceil(self.ws.quantile(0.8)))
            case "Wet Bulb Temperature (C)":
                t = (np.floor(self.wbt.quantile(0.05)), np.ceil(self.wbt.quantile(0.95)))
            case "Dew Point Temperature (C)":
                t = (np.floor(self.dpt.quantile(0.05)), np.ceil(self.dpt.quantile(0.95)))
            case "Global Horizontal Radiation (Wh/m2)":
                t = (0, np.ceil(self.ghr.quantile(0.95)))
            case _:
                raise MissingVariableException(f"'{variable}' is not a valid variable.")

        return t

    def plot_sunriseset(self, overwrite: bool = True) -> None:
        """Plot the sunrise and sunset times."""
        sp = self.target_directory / "sunrise_sunset.png"
        if sp.exists() and not overwrite:
            CONSOLE_LOGGER.info(f"- {sp} already exists")
            return
        if self.location is None:
            raise MissingVariableException(
                "'Location' must be provided to plot sunrise and sunset times."
            )

        fig, ax = plt.subplots(1, 1, figsize=self.config.sunriseset_figsize)
        hours_sunrise_sunset(location=self.epw.location, ax=ax)
        ax.set_title(f"{self.identifier}\nAnnual sunrise and sunset times")
        plt.tight_layout()
        fig.savefig(sp, dpi=self.config.dpi, transparent=True)
        plt.close(fig)

    def plot_evaporative_cooling_potential(self, overwrite: bool = True) -> None:
        sp = self.target_directory / "evaporative_cooling_potential.png"
        if sp.exists() and not overwrite:
            CONSOLE_LOGGER.info(f"- {sp} already exists")
            return
        CONSOLE_LOGGER.info(f"+ Creating {sp}")
        fig, ax = plt.subplots(1, 1, figsize=self.config.evaporativecoolingpotential_figsize)
        evaporative_cooling_potential(dbt=self.dbt, dpt=self.dpt, ax=ax)
        ax.set_title(f"{self.identifier}\nEvaporative Cooling Potential")
        plt.tight_layout()
        fig.savefig(sp, dpi=self.config.dpi, transparent=True)
        plt.close(fig)

    def _plot_windmatrix(
        self, variable: str, cmap: Colormap, lims: tuple[float], overwrite: bool = True
    ) -> None:
        """General function to handle the creation of windmatrix plots."""

        # obtain data
        try:
            series = self.df[variable]
        except KeyError as e:
            raise MissingVariableException(f"'{variable}' column not found in dataframe.") from e

        # construct filepaths
        header = header_from_string(variable)
        sp = self.target_directory / f"windmatrix_{header.data_type.abbreviation}.png"
        if sp.exists() and not overwrite:
            CONSOLE_LOGGER.info(f"- {sp} already exists")
            return

        # generate plot
        CONSOLE_LOGGER.info(f"+ Creating {sp}")
        fig, ax = plt.subplots(1, 1, figsize=self.config.windmatrix_figsize)
        self.wind.plot_windmatrix(
            ax=ax,
            other_data=series,
            show_values=True,
            show_arrows=True,
            vmin=min(lims),
            vmax=max(lims),
            cmap=cmap,
            cbar_title=header.data_type.name,
            unit=header.unit,
        )
        ax.set_title(
            f"{self.identifier}\nTypical monthly-hourly {header.data_type.name} and Wind Direction"
        )
        plt.tight_layout()
        fig.savefig(sp, dpi=self.config.dpi, transparent=True)
        plt.close(fig)

    def plot_windmatrices(self, overwrite: bool = True) -> None:
        """Plot the windmatrix for each variable in the dataframe."""
        d = {
            "Dry Bulb Temperature (C)": {
                "cmap": "Reds",
            },
            "Relative Humidity (%)": {"cmap": "Blues"},
            "Wind Speed (m/s)": {"cmap": "YlGnBu"},
            "Wet Bulb Temperature (C)": {
                "cmap": "Blues",
            },
            "Dew Point Temperature (C)": {
                "cmap": "Blues",
            },
        }
        for k, v in d.items():
            try:
                self._plot_windmatrix(
                    variable=k,
                    cmap=plt.get_cmap(v["cmap"]),
                    lims=self.variable_lims(variable=k),
                    overwrite=overwrite,
                )
            except MissingVariableException as e:
                pass

    def plot_windroses(self, overwrite: bool = True) -> None:

        wind = self.wind

        # do the thing!
        for ap in default_analysis_periods():
            sp = (
                self.target_directory
                / f"windrose_{describe_analysis_period(ap, save_path=True)}.png"
            )
            if sp.exists() and not overwrite:
                CONSOLE_LOGGER.info(f"- {sp} already exists")
                continue
            CONSOLE_LOGGER.info(f"+ Creating {sp}")

            fig, ax = plt.subplots(
                1, 1, figsize=self.config.windrose_figsize, subplot_kw={"projection": "polar"}
            )
            wind.filter_by_analysis_period(ap).plot_windrose(
                ax=ax, directions=16, colors=plt.get_cmap("YlGnBu")
            )
            ax.set_title(f"{self.identifier}\n{describe_analysis_period(ap)}")
            fig.savefig(sp, dpi=self.config.dpi, transparent=True)
            plt.close("all")

    def _plot_diurnal(self, variable: str, color: str, overwrite: bool = True) -> None:
        """General function to handle the creation of diurnal plots."""

        # obtain data
        try:
            series = self.df[variable]
        except KeyError as e:
            raise MissingVariableException(f"'{variable}' column not found in dataframe.") from e

        # construct filepaths
        header = header_from_string(variable)
        sp = self.target_directory / f"diurnal_{header.data_type.abbreviation}.png"
        if sp.exists() and not overwrite:
            CONSOLE_LOGGER.info(f"- {sp} already exists")
            return
        CONSOLE_LOGGER.info(f"+ Creating {sp}")
        # plot figure
        fig, ax = plt.subplots(1, 1, figsize=self.config.diurnal_figsize)
        diurnal(
            ax=ax,
            series=series,
            color=color,
            period="monthly",
            title=f"{self.identifier}\nTypical daily profile of {header.data_type.name}",
        )
        plt.tight_layout()
        plt.savefig(
            sp,
            dpi=self.config.dpi,
            transparent=True,
        )
        plt.close(fig)

    def plot_diurnals(self, overwrite: bool = True) -> None:

        d = {
            "Dry Bulb Temperature (C)": {
                "color": self.config.dbt_color,
            },
            "Relative Humidity (%)": {"color": self.config.rh_color},
            "Wind Speed (m/s)": {"color": self.config.ws_color},
            "Wet Bulb Temperature (C)": {
                "color": "Blue",
            },
            "Dew Point Temperature (C)": {
                "color": "Blue",
            },
            "Global Horizontal Radiation (Wh/m2)": {
                "color": "Orange",
            },
            "Direct Normal Radiation (Wh/m2)": {
                "color": "Orange",
            },
            "Diffuse Horizontal Radiation (Wh/m2)": {
                "color": "Orange",
            },
        }
        for k, v in d.items():
            try:
                self._plot_diurnal(variable=k, color=v["color"], overwrite=overwrite)
            except MissingVariableException as e:
                pass

    def plot_seasonality(self, overwrite: bool = True) -> None:
        """Plot the seasonality of the dataset."""

        # attempt to get the object as an EPW, including location
        if self.location is None:
            raise MissingVariableException(
                "'Location' must be provided to plot seasonality of the dataset."
            )
        sp = self.target_directory / "seasonality.png"
        if sp.exists() and not overwrite:
            CONSOLE_LOGGER.info(f"- {sp} already exists")
            return
        CONSOLE_LOGGER.info(f"+ Creating {sp}")
        fig, ax = plt.subplots(1, 1, figsize=self.config.seasonality_figsize)
        seasonality_comparison(
            ax=ax,
            epw=self.epw,
            winter_color=self.config.winter_color,
            summer_color=self.config.summer_color,
            spring_color=self.config.spring_color,
            autumn_color=self.config.autumn_color,
        )
        ax.set_title(f"{self.identifier}\nSeasonality")
        plt.tight_layout()
        plt.savefig(sp, dpi=self.config.dpi, transparent=True)
        plt.close(fig)

    def plot_radiationmatrix(self, overwrite: bool = True) -> None:
        """Plot the solar tilt-orientation factor charts."""

        # attempt to get the object as an EPW, including location
        if self.location is None:
            raise MissingVariableException(
                "'Location' must be provided to plot seasonality of the dataset."
            )

        # attempt to get radiation values from dataframe, if error raised, then it's likely this is from elswhere and error will be raised
        _ = self.global_horizontal_radiation
        _ = self.direct_normal_radiation
        _ = self.diffuse_horizontal_radiation

        for _type, fs in zip(
            *[
                ["total", "direct", "diffuse"],
                [
                    self.config.solartof_figsize,
                    (self.config.solartof_figsize[0] / 2, self.config.solartof_figsize[1]),
                    (self.config.solartof_figsize[0] / 2, self.config.solartof_figsize[1]),
                ],
            ]
        ):
            sp = self.target_directory / f"radiationmatrix_{_type}.png"
            if sp.exists() and not overwrite:
                CONSOLE_LOGGER.info(f"- {sp} already exists")
                continue
            CONSOLE_LOGGER.info(f"+ Creating {sp}")
            fig, ax = plt.subplots(1, 1, figsize=fs)
            tilt_orientation_factor(
                ax=ax,
                epw_file=Path(self.epw.file_path).resolve(),
                rad_type=_type,
                directions=36 * 3,
                tilts=9 * 3,
                lims=(0, max(self.variable_lims("Global Horizontal Radiation (Wh/m2)"))),
                quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
            )
            plt.tight_layout()
            plt.savefig(sp, dpi=self.config.dpi, transparent=True)
            plt.close(fig)

    def plot_radiationrose(self, overwrite: bool = True) -> None:

        # attempt to get the object as an EPW, including location
        if self.location is None:
            raise MissingVariableException(
                "'Location' must be provided to plot seasonality of the dataset."
            )

        # attempt to get radiation values from dataframe, if error raised, then it's likely this is from elswhere and error will be raised
        _ = self.global_horizontal_radiation
        _ = self.direct_normal_radiation
        _ = self.diffuse_horizontal_radiation

        for ap in [
            AnalysisPeriod(),
            AnalysisPeriod(st_hour=0, end_hour=11),
            AnalysisPeriod(st_hour=12, end_hour=23),
        ]:
            for rad_type in ["total", "direct", "diffuse"]:
                sp = (
                    self.target_directory
                    / f"radiationrose_{rad_type.lower()}_{describe_analysis_period(ap, save_path=True)}.png"
                )
                if sp.exists() and not overwrite:
                    CONSOLE_LOGGER.info(f"- {sp} already exists")
                    continue
                CONSOLE_LOGGER.info(f"+ Creating {sp}")
                fig, ax = plt.subplots(
                    1, 1, figsize=self.config.solarrose_figsize, subplot_kw={"projection": "polar"}
                )
                radiation_rose(
                    epw_file=self.epw_file,
                    analysis_period=ap,
                    rad_type=rad_type,
                    label=True,
                    ax=ax,
                    cmap="Spectral_r",
                    lims=(0, 1500),
                )
                fig.savefig(
                    sp,
                    dpi=self.config.dpi,
                    transparent=True,
                )
                plt.close(fig)

    def plot_utci_shadebenefit(self, overwrite: bool = True) -> None:

        sp = self.target_directory / "utci_shadebenefit.png"
        if sp.exists() and not overwrite:
            CONSOLE_LOGGER.info(f"- {sp} already exists")
            return

        CONSOLE_LOGGER
        simulation_result = self._default_simulation_result()
        unshaded_utci = ExternalComfort(
            simulation_result=simulation_result, typology=Typologies.OPENFIELD
        ).universal_thermal_climate_index
        shaded_utci = ExternalComfort(
            simulation_result=simulation_result, typology=Typologies.ENCLOSED
        ).universal_thermal_climate_index

        fig = utci_shade_benefit(
            unshaded_utci=unshaded_utci,
            shaded_utci=shaded_utci,
            comfort_limits=self.config.utci_categories.simplify().bins_finite,
            location=self.epw.location,
            figsize=self.config.utcishadebenefit_figsize,
            title=self.identifier,
        )
        fig.savefig(sp, dpi=self.config.dpi, transparent=True)
        plt.close(fig)

    def plot_material_temperatures(
        self,
        months: tuple[int] = (
            1,
            4,
            7,
        ),
        overwrite: bool = True,
    ) -> None:
        """Plot the material temperatures for the EPW file."""

        # validation
        if min(months) < 1 or max(months) > 12:
            raise ValueError("Months must be between 1 and 12.")
        if len(months) == 0:
            raise ValueError("At least one month must be provided.")

        # create list of materials and colours
        materials = {
            "Concrete Pavement": {
                "color": "#b8b8b8",
                "material": Materials.Concrete_Pavement.value,
            },
            "Asphalt Pavement": {"color": "#666561", "material": Materials.Asphalt_Pavement.value},
            "Dry Dust": {"color": "#e3bc7f", "material": Materials.Dry_Dust.value},
            "Moist Soil": {"color": "#796f69", "material": Materials.Moist_Soil.value},
            "Wood Siding": {"color": "#bb946f", "material": Materials.Wood_Siding.value},
            "Metal Surface": {"color": "#bdbdbd", "material": Materials.Metal_Surface.value},
        }

        resses = []
        ids = []
        for _, v in materials.items():
            resses.append(
                SimulationResult(
                    epw_file=self.epw_file,
                    ground_material=v["material"],
                    shade_material=Materials.Fabric,
                    identifier=f"{self.epw_file.stem}_SrfTemp_{clean_string(v['material'].identifier)}",
                )
            )
            ids.append(v["material"].identifier)

        lines = ["-", "--", "-.", ":"]
        linecycler = cycle(lines)

        mat_temp_df = pd.concat(
            [i.unshaded_down_temperature_series for i in resses],
            axis=1,
            keys=ids,
        )
        mat_mrt_df = pd.concat(
            [i.unshaded_mean_radiant_temperature_series for i in resses],
            axis=1,
            keys=ids,
        )

        for month in months:
            mat_temp_df_g = (
                mat_temp_df.groupby([mat_temp_df.index.month, mat_temp_df.index.hour])
                .mean()
                .loc[month, :]
            )
            mat_temp_df_g.index = pd.date_range(
                f"2017-{month:02d}-15 00:00:00", freq="60min", periods=24
            )

            mat_mrt_df_g = (
                mat_mrt_df.groupby([mat_mrt_df.index.month, mat_mrt_df.index.hour])
                .mean()
                .loc[month, :]
            )
            mat_mrt_df_g.index = pd.date_range(
                f"2017-{month:02d}-15 00:00:00", freq="60min", periods=24
            )

            # get dbt
            dbt = collection_to_series(self.epw.dry_bulb_temperature)
            dbts = dbt.groupby([dbt.index.month, dbt.index.hour]).mean().loc[month, :]

            sp = self.target_directory / f"groundtemperature_{month:02d}.png"
            if sp.exists() and not overwrite:
                CONSOLE_LOGGER.info(f"- {sp} already exists")
            else:
                CONSOLE_LOGGER.info(f"+ Creating {sp}")
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                ax.plot(
                    mat_temp_df_g.index,
                    dbts.values,
                    color="#800026",
                    lw=4,
                    ls="-",
                    label="Air Temperature",
                    alpha=0.5,
                )
                for col in mat_temp_df_g:
                    ax.plot(
                        mat_temp_df_g.index,
                        mat_temp_df_g[col].values,
                        color=materials[col]["color"],
                        lw=3 if "_" in col else 2,
                        ls=next(linecycler),
                        label=col.replace("_", " "),
                    )
                ax.legend()
                ax.set_xlim(mat_temp_df_g.index.min(), mat_temp_df_g.index.max())
                ax.set_ylim([5, 80])
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %H:%M"))
                ax.set_ylabel("Ground surface temperature (°C)")
                ax.set_title(
                    f"{self.identifier}\nUnshaded ground surface temperature - {calendar.month_name[month]}"
                )
                plt.tight_layout()
                fig.savefig(
                    sp,
                    dpi=self.config.dpi,
                    transparent=True,
                )
                plt.close(fig)

            ################
            sp = self.target_directory / f"groundtemperature_mrt_{month:02d}.png"
            if sp.exists() and not overwrite:
                CONSOLE_LOGGER.info(f"- {sp} already exists")
            else:
                CONSOLE_LOGGER.info(f"+ Creating {sp}")
                linecycler = cycle(lines)
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                ax.plot(
                    mat_mrt_df_g.index,
                    dbts.values,
                    color="#800026",
                    lw=4,
                    ls="-",
                    label="Air Temperature",
                    alpha=0.5,
                )
                for col in mat_mrt_df_g:
                    ax.plot(
                        mat_mrt_df_g.index,
                        mat_mrt_df_g[col].values,
                        color=materials[col]["color"],
                        lw=3 if "_" in col else 2,
                        ls=next(linecycler),
                        label=col.replace("_", " "),
                    )
                ax.legend()
                ax.set_xlim(mat_mrt_df_g.index.min(), mat_mrt_df_g.index.max())
                ax.set_ylim([5, 110])
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %H:%M"))
                ax.set_ylabel("Mean Radiant Temperature (°C)")
                ax.set_title(
                    f"{self.identifier}\nUnshaded mean radiant temperature - {calendar.month_name[month]}"
                )
                plt.tight_layout()
                fig.savefig(
                    sp,
                    dpi=self.config.dpi,
                    transparent=True,
                )
                plt.close(fig)

    def _default_simulation_result(self) -> SimulationResult:
        sp = self.target_directory / "_sr.json"
        if sp.exists():
            return SimulationResult.from_file(sp)
        res = SimulationResult(
            epw_file=self.epw_file,
            ground_material=Materials.Concrete_Pavement,
            shade_material=Materials.Fabric,
        )
        res.to_file(sp)
        return res

    def _default_typologies(self) -> list[Typology]:

        # run baseline simulation
        res = self._default_simulation_result()

        # create canyon shelters
        ns_canyon = [
            Shelter.from_adjacent_wall(
                angle=0, distance_from_wall=7.5, wall_length=150, wall_height=35
            ),
            Shelter.from_adjacent_wall(
                angle=0, distance_from_wall=-7.5, wall_length=150, wall_height=35
            ),
        ]
        nesw_canyon = [
            Shelter.from_adjacent_wall(
                angle=45, distance_from_wall=7.5, wall_length=150, wall_height=35
            ),
            Shelter.from_adjacent_wall(
                angle=45, distance_from_wall=-7.5, wall_length=150, wall_height=35
            ),
        ]
        ew_canyon = [
            Shelter.from_adjacent_wall(
                angle=90, distance_from_wall=7.5, wall_length=150, wall_height=35
            ),
            Shelter.from_adjacent_wall(
                angle=90, distance_from_wall=-7.5, wall_length=150, wall_height=35
            ),
        ]
        nwse_canyon = [
            Shelter.from_adjacent_wall(
                angle=135, distance_from_wall=7.5, wall_length=150, wall_height=35
            ),
            Shelter.from_adjacent_wall(
                angle=135, distance_from_wall=-7.5, wall_length=150, wall_height=35
            ),
        ]

        # create openfield conditions
        typologies = [
            Typology(
                identifier="Openfield",
                evaporative_cooling_effect=[0] * 8760,
                radiant_temperature_adjustment=[0] * 8760,
                shelters=[],
                target_wind_speed=[None] * 8760,
            ),
            Typologies.ENCLOSED.value,
        ]

        # create shelters for layerwed stratgye based on learing from openfield and MRT valeus
        layered_shelters = [
            Shelter.from_overhead_circle(
                radius=5,
                radiation_porosity=np.where(
                    (
                        (res.unshaded_mean_radiant_temperature_series > 60)
                        & (collection_to_series(res.epw.direct_normal_radiation) > 0)
                    ),
                    0.2,
                    0.8,
                ).tolist(),
            )
        ]
        layered_evap = np.where(
            ExternalComfort(
                typology=typologies[0], simulation_result=res
            ).universal_thermal_climate_index_series
            > 32,
            0.2,
            0.05,
        ).tolist()
        layered_wind = np.where(
            (collection_to_series(res.epw.wind_speed) < 1.5)
            & (
                ExternalComfort(
                    typology=typologies[0], simulation_result=res
                ).universal_thermal_climate_index_series
                > 32
            ),
            1.5,
            collection_to_series(res.epw.wind_speed),
        ).tolist()
        typologies.extend(
            [
                Typology(
                    identifier="Overhead shade",
                    evaporative_cooling_effect=[0] * 8760,
                    radiant_temperature_adjustment=[0] * 8760,
                    shelters=Typologies.SKY_SHELTER.value.shelters,
                    target_wind_speed=[None] * 8760,
                ),
                Typology(
                    identifier="NS street canyon",
                    evaporative_cooling_effect=[0] * 8760,
                    radiant_temperature_adjustment=[0] * 8760,
                    shelters=ns_canyon,
                    target_wind_speed=[None] * 8760,
                ),
                Typology(
                    identifier="NE/SW street canyon",
                    evaporative_cooling_effect=[0] * 8760,
                    radiant_temperature_adjustment=[0] * 8760,
                    shelters=nesw_canyon,
                    target_wind_speed=[None] * 8760,
                ),
                Typology(
                    identifier="EW street canyon",
                    evaporative_cooling_effect=[0] * 8760,
                    radiant_temperature_adjustment=[0] * 8760,
                    shelters=ew_canyon,
                    target_wind_speed=[None] * 8760,
                ),
                Typology(
                    identifier="NW/SE street canyon",
                    evaporative_cooling_effect=[0] * 8760,
                    radiant_temperature_adjustment=[0] * 8760,
                    shelters=nwse_canyon,
                    target_wind_speed=[None] * 8760,
                ),
                Typology(
                    identifier="Layered strategy (overhead shade, minor evaporative cooling, supplemental fans)",
                    evaporative_cooling_effect=layered_evap,
                    radiant_temperature_adjustment=[0] * 8760,
                    shelters=layered_shelters,
                    target_wind_speed=layered_wind,
                ),
            ]
        )

        return typologies

    def _default_external_comforts(self) -> list[ExternalComfort]:

        typologies = self._default_typologies()

        # run baseline simulation
        res = self._default_simulation_result()

        ecs = []
        for typology in typologies:
            ec_file = self.target_directory / f"_ec_{clean_string(typology.identifier)}.json"
            if ec_file.exists():
                ecs.append(ExternalComfort.from_file(ec_file))
            else:
                ec = ExternalComfort(typology=typology, simulation_result=res)
                ec.to_file(ec_file)
                ecs.append(ec)

        # sort in terms of average temperature between 08:00-20:00
        return [
            x
            for _, x in sorted(
                zip(
                    [
                        i.universal_thermal_climate_index_series.between_time(
                            "08:00", "20:59"
                        ).mean()
                        for i in ecs
                    ],
                    ecs,
                )
            )
        ]

    def plot_utci_typologies(
        self,
        overwrite: bool = True,
    ) -> None:

        ecs_sorted = self._default_external_comforts()

        # generate heatmap-histogram
        for n, ec in enumerate(ecs_sorted):
            sp = self.target_directory / f"utci_typology{n:02d}.png"
            if sp.exists() and not overwrite:
                CONSOLE_LOGGER.info(f"- {sp} already exists")
                continue
            CONSOLE_LOGGER.info(f"+ Creating {sp}")
            fig = self.config.utci_categories.annual_heatmap_histogram(
                series=ec.universal_thermal_climate_index_series,
                title=f"{self.identifier} - {ec.typology.identifier}",
                show_legend=True,
                show_labels=True,
                ncol=10,
                figsize=self.config.heatmaphistogram_figsize,
            )
            fig.savefig(sp, dpi=self.config.dpi, transparent=True)
            plt.close(fig)

        # generate comparison
        for ap in default_analysis_periods():

            # create mask for analysis periods
            msk = analysis_period_to_boolean(analysis_periods=ap)

            # create name for analysis periods
            name = describe_analysis_period(ap, include_timestep=False)
            sname = describe_analysis_period(ap, include_timestep=False, save_path=True)

            sp = self.target_directory / f"utci_comparison_{sname}.png"
            if sp.exists() and not overwrite:
                CONSOLE_LOGGER.info(f"- {sp} already exists")
                continue
            CONSOLE_LOGGER.info(f"+ Creating {sp}")

            fig, ax = plt.subplots(1, 1, figsize=(15, 4))
            utci_comfort_band_comparison_series(
                ax=ax,
                density=True,
                identifiers=[textwrap.fill(i.typology.identifier, 30) for i in ecs_sorted[::-1]],
                utci_series=[
                    i.universal_thermal_climate_index_series[msk] for i in ecs_sorted[::-1]
                ],
                utci_categories=self.config.utci_categories,
                title=f"{self.identifier}\nComparison between different environmental conditions",
            )
            plt.tight_layout()
            plt.savefig(sp, dpi=self.config.dpi, transparent=True)
            plt.close(fig)

    def plot_utci_limits(self, overwrite: bool = True) -> None:

        for n, mst in enumerate([(0, 0), (0, 0.7)]):
            sp = (
                self.target_directory
                / f"utci_feasiblelimits_{'moisture' if n else 'nomoisture'}.png"
            )
            if sp.exists() and not overwrite:
                CONSOLE_LOGGER.info(f"- {sp} already exists")
                continue
            CONSOLE_LOGGER.info(f"+ Creating {sp}")
            df_styled = thermal_comfort_summary(
                epw=self.epw,
                thermal_comfort_index=ThermalComfortIndex.UTCI,
                comfort_limits=self.config.utci_categories.simplify().bins_finite,
                formatted=True,
                hour_limits=[8, 20],
                moisture_limits=mst,
            )
            dfi.export(df_styled, sp, dpi=self.config.dpi)

    def plot_other_comfort_metrics(self, overwrite: bool = True) -> None:

        # simulate MRT
        ecs_sorted = self._default_external_comforts()

        # process
        for n, ec in enumerate(ecs_sorted):
            tdp = pd.Series(
                data=[
                    dew_point_from_db_rh(db_temp=ddbbtt, rel_humid=rrhh)
                    for (ddbbtt, rrhh) in zip(
                        *[ec.dry_bulb_temperature_series, ec.relative_humidity_series]
                    )
                ],
                index=ec.dry_bulb_temperature_series.index,
                name="Dew Point Temperature (C)",
            )

            # calculate metrics
            wbgt = pd.Series(
                wbgt_v(
                    mrt=ec.mean_radiant_temperature_series,
                    ta=ec.dry_bulb_temperature_series,
                    rh=ec.relative_humidity_series,
                    ws=wind_speed_at_height(
                        ec.wind_speed_series, reference_height=10, target_height=1.5
                    ),
                ),
                name=ec.typology.identifier,
                index=tdp.index,
            )
            hmdex = pd.Series(
                humidex_v(ta=ec.dry_bulb_temperature_series, tdp=tdp),
                name=ec.typology.identifier,
                index=tdp.index,
            )
            hindex = pd.Series(
                heat_index_temperature_v(
                    ta=ec.dry_bulb_temperature_series, rh=ec.relative_humidity_series
                ),
                name=ec.typology.identifier,
                index=tdp.index,
            )

            # Wet bulb globe temperature
            sp = self.target_directory / f"wbgt_typology{n:02d}.png"
            if sp.exists() and not overwrite:
                CONSOLE_LOGGER.info(f"- {sp} already exists")
            else:
                CONSOLE_LOGGER.info(f"+ Creating {sp}")
                f = WBGT_CATEGORIES.annual_heatmap_histogram(
                    series=wbgt,
                    figsize=self.config.heatmaphistogram_figsize,
                    show_legend=True,
                    show_labels=True,
                    ncol=len(WBGT_CATEGORIES),
                    title=f"{self.identifier} - {ec.typology.identifier}",
                )
                f.savefig(sp, dpi=self.config.dpi, transparent=True)
                plt.close("all")

            # Humidex
            sp = self.target_directory / f"humidex_typology{n:02d}.png"
            if sp.exists() and not overwrite:
                CONSOLE_LOGGER.info(f"- {sp} already exists")
            else:
                CONSOLE_LOGGER.info(f"+ Creating {sp}")
                f = HUMIDEX_CATEGORIES.annual_heatmap_histogram(
                    series=hmdex,
                    figsize=self.config.heatmaphistogram_figsize,
                    show_legend=True,
                    show_labels=True,
                    ncol=len(HUMIDEX_CATEGORIES),
                    title=f"{self.identifier} - {ec.typology.identifier}",
                )
                f.savefig(sp, dpi=self.config.dpi, transparent=True)
                plt.close("all")

            # Heat index
            sp = self.target_directory / f"heatindex_typology{n:02d}.png"
            if sp.exists() and not overwrite:
                CONSOLE_LOGGER.info(f"- {sp} already exists")
            else:
                CONSOLE_LOGGER.info(f"+ Creating {sp}")
                f = HEAT_INDEX_CATEGORIES.annual_heatmap_histogram(
                    series=hindex,
                    figsize=self.config.heatmaphistogram_figsize,
                    show_legend=True,
                    show_labels=True,
                    ncol=len(HEAT_INDEX_CATEGORIES),
                    title=f"{self.identifier} - {ec.typology.identifier}",
                )
                f.savefig(sp, dpi=self.config.dpi, transparent=True)
                plt.close("all")

    # def summarise(self) -> None:
    #     """Summarize the EPW file."""

    #     # create the directory which will contain all outputs
    #     self.output_directory(exist_ok=True, parents=True)

    #     # write EPW file to output directory, if required
    #     if self.config.copy_epw:
    #         self.epw.save((self.output_directory / self.epw_file.name).as_posix())

    #     # write config file
    #     with open(self.config_file, "w", encoding="utf-8") as fp:
    #         fp.write(self.config.json, indent=4)
    #     # TODO - Skip files that already exist
    #     # TODO - progress bar
    #     # TODO - ask to overwrite existing files, if config matches
    #     # TODO - textual summarys
    #     # TODO - make generic to process

    #     # plot diurnal ranges
    #     self.plot_monthly_diurnal(
    #         variables=(
    #             "Dry Bulb Temperature (C)",
    #             "Relative Humidity (%)",
    #             "Wet Bulb Temperature (C)",
    #             "Dew Point Temperature (C)",
    #             "Global Horizontal Radiation (Wh/m2)",
    #         )
    #     )

    #     # plot seasonality chart
    #     self.plot_seasonality()

    #     # plot sunrise and sunset times
    #     self.plot_sunriseset()

    #     # plot solar tilt-orientation factors
    #     self.plot_solar_tof()

    #     # plot radiation roses
    #     self.plot_radiation_rose()

    #     # plot material temperatures
    #     self.plot_material_temperatures(months=(1, 4, 7))

    #     # self.epw.summarise(self.parent_directory, copy_file=self._copy_file)

    # def plot_utci_typologies(self) -> None:
    #     pass

    # def plot_utci_thresholds(self) -> None:
    #     pass

    # def plot_other_comfort_metrics(self) -> None:
    #     pass

    # # def summarise_variables(self, variables: tuple[str] = None) -> None:
    # #     """Summarize the variables in the EPW file.

    # #     Args:
    # #         variables (tuple[str], optional):
    # #             The variables to summarize. Defaults to None, which will summarize the following:
    # #             - Dry Bulb Temperature (C)
    # #             - Relative Humidity (%)
    # #             - Wet Bulb Temperature (C)
    # #             - Dew Point Temperature (C)
    # #             - Global Horizontal Radiation (Wh/m2)
    # #     """
    # #     pass
