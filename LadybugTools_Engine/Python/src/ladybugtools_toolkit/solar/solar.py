"""Methods for handling solar data. This module relies heavily on numpy, pandas, and ladybug."""

# TODO - Shade benefit calc (on window) - https://github.com/ladybug-tools/ladybug-grasshopper/blob/master/ladybug_grasshopper/src/LB%20Shade%20Benefit.py
# TODO - PV calc from pvlib
# TODO - PV with shade objects (from sky matrix, or get total incident radiation on surface using Radiance and then feed into PVLib)
# TODO - Use DirectSun/RadiationStudy to calculate shadedness of a point given context meshes
# todo - glare risk for aperture in direction

import copy
import hashlib
import json
import subprocess
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Union

import ephem
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from honeybee.config import folders as hb_folders
from honeybee.model import Model, Shade
from honeybee_energy.generator.loadcenter import ElectricLoadCenter
from honeybee_energy.generator.pv import PVProperties
from honeybee_energy.lib.constructions import opaque_construction_by_identifier
from honeybee_energy.result.err import Err
from honeybee_energy.run import run_idf
from honeybee_energy.simulation.parameter import SimulationParameter
from honeybee_energy.writer import energyplus_idf_version
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.base import DataTypeBase
from ladybug.datatype.energy import Energy
from ladybug.dt import Date
from ladybug.epw import EPW
from ladybug.futil import write_to_file_by_name
from ladybug.sql import SQLiteResult
from ladybug.sunpath import Location, Sun, Sunpath
from ladybug.viewsphere import ViewSphere
from ladybug.wea import Wea
from ladybug_geometry.geometry3d import (
    Face3D,
    Mesh3D,
    Plane,
    Point3D,
    Polyline3D,
    Ray3D,
)
from ladybug_radiance.config import folders as lbr_folders
from ladybug_radiance.skymatrix import SkyMatrix
from ladybug_radiance.study.radiation import RadiationStudy
from ladybug_radiance.visualize.raddome import RadiationDome
from ladybug_radiance.visualize.radrose import RadiationRose
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap
from matplotlib.ticker import MultipleLocator

from ..convert.to_ladybug import to_ladybug
from ..convert.to_pandas import to_pandas
from ..convert.to_shapely import to_shapely #TODO
from ..honeybee_energy_extension.util import get_schedule_as_data_collection #TODO
from ..json_encoding import AllPowerfulEncoder #TODO
from ..ladybug_extension.datatype import RadiationBenefit #TODO
from ..ladybug_extension.location import (
    average_location,
    get_tzinfo,
    location_to_pytz_fixed_offset,
    location_to_timezone,
)
from ..ladybug_extension.sunpath import sunrise_sunset
from ..ladybug_extension.util import ( #TODO
    analysis_period_to_string,
    metadata_dict_to_str,
    metadata_str_to_dict,
)
from ..ladybug_geometry_extension.util import (
    _create_azimuth_mesh,
    azimuth_altitude_to_vector,
    vector_to_azimuth_altitude,
)
from ..bhom.logger import CONSOLE_LOGGER
from python_toolkit.helpers import contrasting_color
from .util import radiation_from_location


@dataclass
class Solar:
    """An object containing solar data.

    Args:
        location (Location):
            A ladybug Location object.
        datetimes (pd.DatetimeIndex):
            An iterable of datetime-like objects.
        direct_normal_radiation (list[float]):
            An iterable of direct normal irradiance values, in Wh/m2.
        diffuse_horizontal_radiation (list[float]):
            An iterable of diffuse horizontal irradiance values, in Wh/m2.
        global_horizontal_radiation (list[float]):
            An iterable of global horizontal irradiance values, in Wh/m2.

    """

    # NOTE - BE STRICT WITH THE TYPING!
    # NOTE - Values must cover at least an entire year
    # NOTE - Conversions happen in class methods.
    # NOTE - Validation happens at instantiation.

    location: Location
    datetimes: pd.DatetimeIndex
    global_horizontal_radiation: list[float]
    direct_normal_radiation: list[float]
    diffuse_horizontal_radiation: list[float]

    # region: DUNDER METHODS

    def __post_init__(self):
        """Check for validation of the inputs."""
        # location validation
        if not isinstance(self.location, Location):
            raise ValueError("location must be a ladybug Location object.")
        if self.location.source is None:
            warnings.warn(
                "The source field of the Location input is None. This means that things are a bit ambiguous!"
            )

        # datetimes validation
        self.datetimes = pd.DatetimeIndex(self.datetimes, freq="infer")
        # check datetimes are increasing, and unique, and don't contain nulls
        if any(self.datetimes.duplicated()):
            raise ValueError("datetimes must be unique.")
        if any(self.datetimes.isna()):
            raise ValueError("datetimes cannot contain null values.")
        if any(np.diff(self.datetimes) < timedelta(0)):
            raise ValueError("datetimes must be in increasing order.")

        # timezone validation
        if self.datetimes[0].tzinfo is None:
            self.datetimes = self.datetimes.tz_localize(
                location_to_timezone(self.location), ambiguous="NaT"
            )

        # irradiance validation
        array_names = [
            "direct_normal_radiation",
            "diffuse_horizontal_radiation",
            "global_horizontal_radiation",
        ]
        for name in array_names:
            _temp = getattr(self, name)
            # length validation
            if len(_temp) != len(self.datetimes):
                raise ValueError(
                    f"{name} must be the same length as datetimes. {len(_temp)} != {len(self.datetimes)}."
                )
            # dtype validation
            if not all(isinstance(i, (int, float)) for i in _temp):
                raise ValueError(f"{name} must be a list of numeric values.")
            # null validation
            if any(np.isnan(_temp)):
                raise ValueError(f"{name} cannot contain null values.")
            # value limit validation
            if any([i < 0 for i in _temp]):
                raise ValueError(f"{name} must be >= 0")

    def __len__(self) -> int:
        return len(self.datetimes)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} data from {self.location.source}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(
            (
                self.location,
                tuple(self.datetimes),
                tuple(self.direct_normal_radiation),
                tuple(self.diffuse_horizontal_radiation),
                tuple(self.global_horizontal_radiation),
            )
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Solar):
            return False
        return (
            self.location == other.location
            and all(self.datetimes == other.datetimes)
            and self.direct_normal_radiation == other.direct_normal_radiation
            and self.diffuse_horizontal_radiation == other.diffuse_horizontal_radiation
            and self.global_horizontal_radiation == other.global_horizontal_radiation
        )

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield (self.datetimes[i], self.global_horizontal_radiation[i], self.direct_normal_radiation[i], self.diffuse_horizontal_radiation[i])

    def __getitem__(self, idx: int) -> dict[str, Union[datetime, float]]:
        return {
            "datetime": self.datetimes[idx],
            "global_horizontal_radiation": self.global_horizontal_radiation[idx],
            "direct_normal_radiation": self.direct_normal_radiation[idx],
            "diffuse_horizontal_radiation": self.diffuse_horizontal_radiation[idx],
        }

    def __copy__(self) -> "Solar":
        return Solar(
            location=self.location.duplicate(),
            datetimes=self.datetimes.copy(),
            direct_normal_radiation=copy.copy(self.direct_normal_radiation),
            diffuse_horizontal_radiation=copy.copy(self.diffuse_horizontal_radiation),
            global_horizontal_radiation=copy.copy(self.global_horizontal_radiation),
        )

    # endregion: DUNDER METHODS

    # region: PROPERTIES

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata for this object as a dictionary."""
        d = self.location.to_dict()
        # drop the "type" key
        del d["type"]
        return d

    @property
    def start_datetime(self) -> date:
        return min(self.datetimes)

    @property
    def end_datetime(self) -> date:
        return max(self.datetimes)

    @property
    def lb_datetimes(self) -> list[date]:
        return [to_ladybug(dt) for dt in self.datetimes]

    @property
    def lb_dates(self) -> list[Date]:
        return [to_ladybug(d) for d in self.datetimes]

    @property
    def direct_normal_radiation_series(self) -> pd.Series:
        return pd.Series(
            data=self.direct_normal_radiation,
            index=self.datetimes,
            name=(
                "Direct Normal Radiation",
                "W/m2",
                metadata_dict_to_str(self.metadata),
            ),
        )

    @property
    def direct_normal_radiation_collection(self) -> HourlyContinuousCollection:
        # create an aggregate dataset
        series = self.direct_normal_radiation_series
        agg_year = series.groupby(
            [series.index.month, series.index.day, series.index.hour]
        ).mean()
        if (2, 29) in agg_year.index:
            # if leap day is present, we need to remove it
            agg_year = agg_year.drop((2, 29))
        # add a generic index
        agg_year.index = to_pandas(AnalysisPeriod())
        # change unit to "Wh/m2"
        agg_year.name = (
            agg_year.name[0],
            "Wh/m2",
            agg_year.name[2] + " | __type__: HourlyContinuousCollection",
        )

        return to_ladybug(agg_year)

    @property
    def diffuse_horizontal_radiation_series(self) -> pd.Series:
        return pd.Series(
            data=self.diffuse_horizontal_radiation,
            index=self.datetimes,
            name=(
                "Diffuse Horizontal Radiation",
                "W/m2",
                metadata_dict_to_str(self.metadata),
            ),
        )

    @property
    def diffuse_horizontal_radiation_collection(self) -> HourlyContinuousCollection:
        # create an aggregate dataset
        series = self.diffuse_horizontal_radiation_series
        agg_year = series.groupby(
            [series.index.month, series.index.day, series.index.hour]
        ).mean()
        if (2, 29) in agg_year.index:
            # if leap day is present, we need to remove it
            agg_year = agg_year.drop((2, 29))
        # add a generic index
        agg_year.index = to_pandas(AnalysisPeriod())
        # change unit to "Wh/m2"
        agg_year.name = (
            agg_year.name[0],
            "Wh/m2",
            agg_year.name[2] + " | __type__: HourlyContinuousCollection",
        )

        return to_ladybug(agg_year)

    @property
    def global_horizontal_radiation_series(self) -> pd.Series:
        return pd.Series(
            data=self.global_horizontal_radiation,
            index=self.datetimes,
            name=(
                "Global Horizontal Radiation",
                "W/m2",
                metadata_dict_to_str(self.metadata),
            ),
        )

    @property
    def global_horizontal_radiation_collection(self) -> HourlyContinuousCollection:
        # create an aggregate dataset
        series = self.global_horizontal_radiation_series
        agg_year = series.groupby(
            [series.index.month, series.index.day, series.index.hour]
        ).mean()
        if (2, 29) in agg_year.index:
            # if leap day is present, we need to remove it
            agg_year = agg_year.drop((2, 29))
        # add a generic index
        agg_year.index = to_pandas(AnalysisPeriod())
        # change unit to "Wh/m2"
        agg_year.name = (
            agg_year.name[0],
            "Wh/m2",
            agg_year.name[2] + " | __type__: HourlyContinuousCollection",
        )

        return to_ladybug(agg_year)

    @property
    def df(self) -> pd.DataFrame:
        return pd.concat(
            [
                self.direct_normal_radiation_series,
                self.diffuse_horizontal_radiation_series,
                self.global_horizontal_radiation_series,
            ],
            axis=1,
        )

    @property
    def analysis_period(self) -> AnalysisPeriod:
        """Return an appropriate analysis period for this object."""
        # convert datetimes to "single-year"
        data = pd.Series(index=self.datetimes, data=np.zeros_like(self.datetimes))
        data = data.groupby([data.index.month, data.index.day, data.index.time]).mean()
        m_index = data.index
        year = 2016 if [2, 29] in m_index.to_frame(index=False)[[0, 1]].values else 2017
        dts = pd.to_datetime(
            [
                datetime(year, month, day, time.hour, time.minute, time.second)
                for month, day, time in m_index
            ]
        )
        return to_ladybug(dts)

    @property
    def sunpath(self) -> Sunpath:
        return Sunpath.from_location(self.location)

    @property
    def suns(self) -> list[Sun]:
        sunpath = self.sunpath
        suns = [sunpath.calculate_sun_from_date_time(i) for i in self.lb_datetimes]
        return suns

    @property
    def suns_df(self) -> pd.DataFrame:
        """Get a DataFrame of the sun positions for the analysis period."""
        suns = self.suns
        return pd.DataFrame(
            {
                "azimuth": [s.azimuth for s in suns],
                "altitude": [s.altitude for s in suns],
            },
            index=self.datetimes,
        )

    @property
    def sunrise_sunset(self) -> pd.DataFrame:
        """Get sunrise and sunset times for the analysis period.

        Returns:
            pd.DataFrame:
                A DataFrame with sunrise and sunset times for each date in the analysis period.

        """
        return sunrise_sunset(
            dates=np.unique(self.datetimes.date),
            location=self.location,
        )

    @property
    def solstices_equinoxes(self) -> pd.DataFrame:
        """Get the solstices and equinoxes for this object.

        Returns:
            pd.DataFrame:
                A DataFrame with solstices and equinoxes for each date in the analysis period.

        """
        d = {}
        for year in self.datetimes.year.unique():
            d[year] = {
                "vernal equinox": ephem.next_vernal_equinox(str(year))
                .datetime()
                .date(),
                "summer solstice": ephem.next_summer_solstice(str(year))
                .datetime()
                .date(),
                "autumnal equinox": ephem.next_autumnal_equinox(str(year))
                .datetime()
                .date(),
                "winter solstice": ephem.next_winter_solstice(str(year))
                .datetime()
                .date(),
            }
        return pd.DataFrame(d).T

    @property
    def optimal_pv_position(self) -> tuple[float, float, float]:
        """Get the optimal tilt and azimuth for a PV panel at this location.

        Returns:
            tuple[float, float, float]:
                A tuple containing the max insolation, optimal tilt (degrees) and azimuth (degrees)
                for a PV panel at this location.

        """
        # location/time: use EPW or pvlib Location
        tz = get_tzinfo(
            longitude=self.location.longitude, latitude=self.location.latitude
        )
        location = pvlib.location.Location(
            self.location.latitude,
            self.location.longitude,
            altitude=self.location.elevation,
            name="site",
        )

        times = pd.date_range(
            "2017-01-01", periods=8760, freq="h", tz=tz
        )  # annual if possible

        # get solar angles and weather (use clear-sky or real)
        sp = pvlib.solarposition.get_solarposition(
            times, location.latitude, location.longitude
        )
        cs = location.get_clearsky(times)  # GHI, DNI, DHI
        ghi = cs["ghi"].values
        dni = cs["dni"].values
        dhi = cs["dhi"].values
        zenith = sp["zenith"].values
        azimuth = sp["azimuth"].values

        # grid
        tilts = np.linspace(0, 45, 45 * 2)  # test 0..45 deg
        azis = np.linspace(0, 306, 306 * 2)  # test 0..360 step 5°

        best = None
        for tilt in tilts:
            # vectorised POA calc for all azimuths at once (fast)
            # get_total_irradiance accepts scalar tilt and array of azimuths if you vectorize per azimuth
            poa_sums = []
            for azi in azis:
                poa = pvlib.irradiance.get_total_irradiance(
                    surface_tilt=tilt,
                    surface_azimuth=azi,
                    solar_zenith=zenith,
                    solar_azimuth=azimuth,
                    dni=dni,
                    ghi=ghi,
                    dhi=dhi,
                )["poa_global"]
                poa_sums.append(
                    poa.sum()
                )  # if hourly irradiance in Wh/m2; sum approximates annual Wh/m2

            poa_sums = np.array(poa_sums)
            idx = poa_sums.argmax()
            if best is None or poa_sums[idx] > best[0]:
                best = (poa_sums[idx], tilt, azis[idx])

            # best now holds (annual_wh_m2, tilt, azimuth)
        return best[0], 90 - best[1], best[2]

    # endregion: PROPERTIES

    # region: CLASS METHODS
    @classmethod
    def from_wea(cls, wea: Wea) -> "Solar":
        if not isinstance(wea, Wea):
            raise ValueError("wea must be a ladybug Wea object.")

        # modify location to state the Wea object in the source field
        location = wea.location.duplicate()
        location.source = "WEA"

        # obtain the datetimes
        datetimes = pd.to_datetime(
            wea.analysis_period.datetimes, freq="infer"
        ).tz_localize(location_to_pytz_fixed_offset(location))

        return cls(
            location=location,
            datetimes=datetimes,
            direct_normal_radiation=wea.direct_normal_irradiance.values,
            diffuse_horizontal_radiation=wea.diffuse_horizontal_irradiance.values,
            global_horizontal_radiation=wea.global_horizontal_irradiance.values,
        )

    @classmethod
    def from_pvlib(
        cls,
        location: Location,
        start_date: Union[str, date],
        end_date: Union[str, date],
        total_sky_cover: Union[float, list[float], None] = None,
    ) -> "Solar":
        """Construct a Solar object using PVLib."""
        # construct dataframe using PVLib
        df = radiation_from_location(
            location=location,
            start_date=start_date,
            end_date=end_date,
            total_sky_cover=total_sky_cover,
        )

        # modify the location to state the PVLib in the source field
        location = location.duplicate()
        location.source = metadata_str_to_dict(df.columns[0][-1])["source"]

        # construct the resulting object
        return Solar.from_dataframe(df=df, location=location)

    @classmethod
    def from_epw(cls, epw: Union[Path, EPW]) -> "Solar":
        """Create a Solar object from an EPW file or object.

        Args:
            epw (Union[Path, EPW]):
                The path to the EPW file, or an EPW object.

        """
        if isinstance(epw, (str, Path)):
            epw = EPW(epw)

        # modify location to state the EPW file in the source field
        location = epw.location
        location.source = f"{Path(epw.file_path).name}"

        # obtain the datetimes
        datetimes = to_pandas(epw.dry_bulb_temperature.header.analysis_period)
        datetimes = datetimes.tz_localize(location_to_pytz_fixed_offset(location))

        return cls(
            location=location,
            datetimes=datetimes,
            direct_normal_radiation=epw.direct_normal_radiation.values,
            diffuse_horizontal_radiation=epw.diffuse_horizontal_radiation.values,
            global_horizontal_radiation=epw.global_horizontal_radiation.values,
        )

    def to_dict(self) -> dict:
        """Represent the object as a python-native dtype dictionary."""
        return {
            "type": "Solar",
            "location": self.location.to_dict(),
            "datetimes": [i.isoformat() for i in self.datetimes],
            "direct_normal_radiation": self.direct_normal_radiation,
            "diffuse_horizontal_radiation": self.diffuse_horizontal_radiation,
            "global_horizontal_radiation": self.global_horizontal_radiation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Solar":
        """Create this object from a dictionary."""
        if d.get("type", None) != "Solar":
            raise ValueError("The dictionary cannot be converted Solar object.")

        return cls(
            location=Location.from_dict(d["location"]),
            datetimes=pd.to_datetime(d["datetimes"]),
            direct_normal_radiation=d["direct_normal_radiation"],
            diffuse_horizontal_radiation=d["diffuse_horizontal_radiation"],
            global_horizontal_radiation=d["global_horizontal_radiation"],
        )

    def to_json(self) -> str:
        """Convert this object to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_string: str) -> "Solar":
        """Create this object from a JSON string."""
        return cls.from_dict(json.loads(json_string))

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        location: Location,
    ) -> "Solar":
        """Create this object from a DataFrame.

        Args:
            df (pd.DataFrame):
                A DataFrame object containing the solar data.
            location (Location, optional):
                A ladybug Location object. If not provided, the location data
                will be extracted from the DataFrame if present.

        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame's index must be of type pd.DatetimeIndex.")
        if not isinstance(location, Location):
            raise ValueError("location must be a ladybug Location object.")
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("The DataFrame's columns must be of type pd.MultiIndex.")
        unit = "W/m2"
        dni_name = "Direct Normal Radiation"
        dhi_name = "Diffuse Horizontal Radiation"
        ghi_name = "Global Horizontal Radiation"
        # check that each name above is in the df columns at the first level of the MultiIndex
        for name in [dni_name, dhi_name, ghi_name]:
            if name not in df.columns.get_level_values(0):
                raise ValueError(f"{name} not found in DataFrame columns.")
            if df[name].columns.get_level_values(0)[0] != unit:
                raise ValueError(
                    f"{name} column does not have the correct unit ({unit})."
                )

        metadata = []
        # get the direct normal radiation
        dni_series: pd.Series = df[dni_name][unit].squeeze()
        metadata.append(dni_series.name)
        # get the diffuse horizontal radiation
        dhi_series: pd.Series = df[dhi_name][unit].squeeze()
        metadata.append(dhi_series.name)
        # get the global horizontal radiation
        ghi_series: pd.Series = df[ghi_name][unit].squeeze()
        metadata.append(ghi_series.name)

        loc_copy = location.duplicate()
        try:
            d = metadata_str_to_dict(metadata[0])
            loc_copy.source = d["source"]
        except Exception as e:
            print(e)
            loc_copy.source = "pd.DataFrame"

        return cls(
            location=loc_copy,
            datetimes=df.index,
            direct_normal_radiation=dni_series.values.tolist(),
            diffuse_horizontal_radiation=dhi_series.values.tolist(),
            global_horizontal_radiation=ghi_series.values.tolist(),
        )

    @classmethod
    def from_average(
        cls,
        objects: list["Solar"],
        weights: Union[list[Union[int, float]], None] = None,
    ) -> "Solar":
        # validation
        if not all(isinstance(i, Solar) for i in objects):
            raise ValueError("objects must be a 1D list of Solar objects.")
        if len(objects) == 0:
            raise ValueError("objects cannot be empty.")
        if len(objects) == 1:
            return objects[0]

        # check datetimes are the same
        for obj in objects:
            if not all(obj.datetimes == objects[0].datetimes):
                raise ValueError("All objects must share the same datetimes.")

        # create the average data's
        dni = np.average(
            [i.direct_normal_radiation for i in objects], weights=weights, axis=0
        )
        dhi = np.average(
            [i.diffuse_horizontal_radiation for i in objects],
            weights=weights,
            axis=0,
        )
        ghi = np.average(
            [i.global_horizontal_radiation for i in objects],
            weights=weights,
            axis=0,
        )
        location = average_location([i.location for i in objects], weights=weights)
        return cls(
            location=location,
            datetimes=objects[0].datetimes,
            direct_normal_radiation=dni,
            diffuse_horizontal_radiation=dhi,
            global_horizontal_radiation=ghi,
        )

    # endregion: CLASS METHODS

    # region: INSTANCE METHODS

    def to_wea(self, timestep: int = 1) -> Wea:
        return Wea(
            location=self.location,
            direct_normal_irradiance=self.direct_normal_radiation_collection.interpolate_to_timestep(
                timestep=timestep
            ),
            diffuse_horizontal_irradiance=self.diffuse_horizontal_radiation_collection.interpolate_to_timestep(
                timestep=timestep
            ),
        )

    def apply_shade_objects(
        self,
        shade_objects: tuple[Shade, ...] = (),
    ) -> "Solar":
        """Apply shade objects to the solar data. This uses EnergyPlus in a
        way similar to the PV generation loads method as used in
        https://github.com/ladybug-tools/honeybee-grasshopper-energy/blob/master/honeybee_grasshopper_energy/src/HB%20Generation%20Loads.py

        Args:
            shade_objects (list[Shade], optional):
                A list of ladybug Shade objects which may also contain temporal
                variation in transmissivity. The shade objects will be used
                to calculate the shading effect on the solar data.

        Returns:
            Solar:
                A new Solar object with the shading applied.

        Example:
            >>> from lbttk.solar import Solar
            >>> ...

        FIXME - add example script here!!!!!!!!!!!!

        """

        # TODO - implement this method to calculate the impact of shades at varying locations surrounding the "sensor"
        raise NotImplementedError("Hourly shade impact not yet implemented.")

    def sun_exposure(
        self,
        shades: tuple[Shade, ...],
        origin: Point3D = Point3D(0, 0, 1.2),  # type: ignore
        parallel: bool = False,
    ) -> tuple[float, ...]:
        """Calculate annual hourly sun exposure.

        - 0 means the sun is blocked
        - 1 means it is visible
        - 0.25 mean the sun is only 25% visible (through a 75% porous medium)
        - np.nan means the sun is below the horizon

        Args:
            shades (tuple[Shade, ...]):
                A set of shades that may include temporal transmissivity
                properties.
            origin (Point3D, optional):
                The location of the sensor from which sun exposure is
                calculated, in relation to the shading objects.
            parallel (bool, optional):
                If True, run in parallel. Useful when number of shade objects
                is high, otherwise the overheads of set-up aren't worth it.
        Returns:
            List[float]:
                A list of annual hourly values denoting sun exposure values.

        Examples
        --------
        >>> from lbttk.plot.matplotlib import heatmap
        >>> from lbttk.solar.solar import Solar
        >>> from ladybug_geometry.geometry3d import Point3D, Face3D, LineSegment3D, Vector3D, Plane
        >>> from honeybee.shade import Shade
        >>> import numpy as np
        >>> import pandas as pd
        >>> from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval

        >>> sol = Solar.from_epw("./file.epw")

        >>> shd_dynamic_south = Shade(
        >>>     "shade",
        >>>     Face3D.from_extrusion(
        >>>         LineSegment3D.from_end_points(Point3D(-8, -5, 0), Point3D(8, -5, 0)),
        >>>         Vector3D(0, 0, 25),
        >>>     ),
        >>> )
        >>> shd_dynamic_south.properties.energy.transmittance_schedule = ScheduleFixedInterval(
        >>>     identifier="transmissivity", values=abs((np.arange(8760) - 4380) / 4380).tolist()
        >>> )

        >>> shd_opaque_above = Shade(
        >>>     identifier="shade_above",
        >>>     geometry=Face3D.from_regular_polygon(
        >>>         side_count=6, radius=10, base_plane=Plane(o=Point3D(0, 0, 10))
        >>>     ),
        >>> )

        >>> result = sol.sun_exposure(
        >>>     shades=(shd_dynamic_south, shd_opaque_above),
        >>>     origin=Point3D(0, 0, 1.2),
        >>>     parallel=False,
        >>> )

        >>> heatmap(pd.Series(result, name="hi", index=sol.datetimes))
        """

        # validate inputs
        if not all(isinstance(i, Shade) for i in shades):
            raise ValueError("shades must be an iterable of ladybug Shade objects.")

        # create shade transmissivities
        shade_transmissivities = []
        for hb_shd in shades:
            try:
                shade_transmissivities.append(
                    get_schedule_as_data_collection(
                        hb_shd.properties.energy.transmittance_schedule  # type: ignore
                    ).values
                )
            except ValueError:
                # no schedule available, assume fully opaque
                shade_transmissivities.append(tuple([0.0] * len(self)))

        if not parallel:
            # for each sun, determine whether a ray from the origin to the sun intersects
            # any of the shade objects, and the resultant visibility as product of the
            # intersecting shade transmissivities at that time

            sun_visibility = []
            for n, sun in enumerate(self.suns):
                if sun.altitude < 0:
                    # sun below horizon
                    sun_visibility.append(np.nan)
                    continue
                ray = Ray3D(origin, sun.sun_vector_reversed)
                transmissivity = 1.0
                for m, hb_shd in enumerate(shades):
                    # check for intersection
                    if hb_shd.geometry.intersect_line_ray(ray):
                        # sun is blocked by shade geometry
                        transmissivity *= shade_transmissivities[m][n]
                sun_visibility.append(transmissivity)
        else:

            def process_sun(n: int, sun: Sun) -> tuple[int, float]:
                """Process a single sun position and return its visibility.

                Args:
                    n: The timestep index
                    sun: The Sun object for this timestep

                Returns:
                    tuple: (index, transmissivity_value)
                """
                if sun.altitude < 0:
                    # sun below horizon
                    return (n, np.nan)

                ray = Ray3D(origin, sun.sun_vector_reversed)
                transmissivity = 1.0

                for m, hb_shd in enumerate(shades):
                    # check for intersection
                    if hb_shd.geometry.intersect_line_ray(ray):
                        # sun is blocked by shade geometry
                        transmissivity *= shade_transmissivities[m][n]

                return (n, transmissivity)

            # Process all suns in parallel
            sun_visibility: list[float] = [None] * len(self.suns)  # type: ignore

            with ThreadPoolExecutor(max_workers=min(len(self.suns), 8)) as executor:
                # Submit all sun calculations
                futures = {
                    executor.submit(process_sun, n, sun): n
                    for n, sun in enumerate(self.suns)
                }

                # Collect results in order
                for future in as_completed(futures):
                    idx, visibility = future.result()
                    sun_visibility[idx] = visibility

        return tuple(sun_visibility)

    def _sky_matrix(
        self,
        north: int = 0,
        ground_reflectance: float = 0.2,
        temperature: Optional[HourlyContinuousCollection] = None,
        balance_temperature: float = 15,
        balance_offset: float = 2,
        timestep: int = 1,
    ) -> SkyMatrix:
        """Create a ladybug sky matrix from the solar data.

        Args:
            north (int, optional):
                The north direction in degrees.
                Default is 0.
            ground_reflectance (float, optional):
                The ground reflectance value.
                Default is 0.2.
            temperature (Optional[HourlyContinuousCollection], optional):
                An iterable of temperature values, or a ladybug HourlyContinuousCollection
                for temperature, which will be used to establish whether radiation
                is desired or not for each time step. The collection must be aligned
                with the irradiance inputs.
            balance_temperature (float, optional):
                The temperature in Celsius between which radiation
                switches from being a benefit to a harm. Typical residential buildings
                have balance temperatures as high as 18C and commercial buildings tend
                to have lower values around 12C.
                Default is 15.
            balance_offset (float, optional):
                The temperature offset from the balance temperature
                in Celsius where radiation is neither harmful nor helpful.
                Default is 2.
            timestep (int, optional):
                The timestep (per hour) in minutes for the sky matrix.
                Default is 1.

        Returns:
            SkyMatrix:
                A ladybug SkyMatrix object.

        """
        if temperature is None:
            return SkyMatrix(
                wea=self.to_wea(timestep=timestep),
                north=north,
                high_density=True,
                ground_reflectance=ground_reflectance,
            )

        # check that temperature is a valid type
        dni = self.direct_normal_radiation_collection.interpolate_to_timestep(
            timestep=timestep
        )
        dhi = self.diffuse_horizontal_radiation_collection.interpolate_to_timestep(
            timestep=timestep
        )
        if not isinstance(temperature, HourlyContinuousCollection):
            raise ValueError(
                "temperature must be a ladybug HourlyContinuousCollection object."
            )
        if len(self) != len(temperature):
            raise ValueError(
                f"temperature must be the same length as the solar data (n={len(self)})."
            )

        return SkyMatrix.from_components_benefit(
            location=self.location,
            direct_normal_irradiance=dni,
            diffuse_horizontal_irradiance=dhi,
            north=north,
            high_density=True,
            ground_reflectance=ground_reflectance,
            temperature=temperature,
            balance_temperature=balance_temperature,
            balance_offset=balance_offset,
        )

    def _radiation_rose(
        self,
        sky_matrix: Optional[SkyMatrix] = None,
        intersection_matrix: Optional[Any] = None,
        direction_count: int = 36,
        tilt_angle: int = 0,
    ) -> RadiationRose:
        """Convert this object to a ladybug RadiationRose object.

        Args:
            sky_matrix (Optional[SkyMatrix], optional):
                A SkyMatrix object, which describes the radiation coming
                from the various patches of the sky.
                Default is None, which uses the default sky matrix from the solar data.
            intersection_matrix (Optional[Any], optional):
                An optional lists of lists, which can be used to account
                for context shade surrounding the radiation rose. The matrix
                should have a length equal to the direction_count and begin
                from north moving clockwise. Each sub-list should consist of
                booleans and have a length equal to the number of sky patches
                times 2 (indicating sky patches and ground patches). True
                indicates that a certain patch is seen and False indicates
                that the match is blocked. If None, the radiation rose will be
                computed assuming no obstructions.
                Default is None.
            direction_count (int, optional):
                An integer greater than or equal to 3, which notes the number
                of arrows to be generated for the radiation rose.
                Default is 36.
            tilt_angle (float, optional):
                A number between 0 and 90 that sets the vertical tilt angle
                (aka. the altitude) for all of the directions. By default,
                the Radiation Rose depicts the amount of solar energy
                received by a vertical wall (tilt_angle=0). The tilt_angle
                be changed to a specific value to assess the solar energy
                falling on geometries that are not perfectly vertical, such
                as a tilted photovoltaic panel.
                Default is 0.

        Returns:
            RadiationRose:
                A ladybug RadiationRose object.

        """
        if sky_matrix is None:
            sky_matrix = self._sky_matrix()

        return RadiationRose(
            sky_matrix=sky_matrix,
            intersection_matrix=intersection_matrix,
            direction_count=direction_count,
            tilt_angle=tilt_angle,
        )

    def radiation_benefit(
        self,
        temperature: HourlyContinuousCollection,
        north: int = 0,
        ground_reflectance: float = 0.2,
        balance_temperature: float = 15,
        balance_offset: float = 2,
    ) -> HourlyContinuousCollection:
        """Return the radiation benefit data from the sky matrix.

        See documentation for self.lb_sky_matrix for more information.
        """
        # create the sky matrix
        smx = self._sky_matrix(
            north=north,
            ground_reflectance=ground_reflectance,
            temperature=temperature,
            balance_temperature=balance_temperature,
            balance_offset=balance_offset,
        )

        # replace None values with NaN
        d = []
        for i in smx.benefit_matrix:
            if i is None:
                d.append(1)
            elif i:
                d.append(2)
            else:
                d.append(0)

        return temperature.get_aligned_collection(
            value=d, data_type=RadiationBenefit(), unit="category"
        )

    def _radiation_rose_data(
        self,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        directions: int = 36,
        tilt_angle: float = 0,
        north: int = 0,
        ground_reflectance: float = 0.2,
        shade_objects: tuple[Any, ...] = (),
    ) -> pd.DataFrame:
        """Get directional cumulative radiation in kWh/m2 for a given
        tilt_angle, within the analysis_period and subject to shade_objects.

        Args:
            analysis_period (AnalysisPeriod, optional):
                The analysis period over which radiation shall be summarised.
                Defaults to AnalysisPeriod().
            directions (int, optional):
                The number of directions to bin data into.
                Defaults to 36.
            tilt_angle (float, optional):
                The tilt (from 0 at horizon, to 90 facing the sky) to assess.
                Defaults to 89.999.
            north (int, optional):
                The north direction in degrees.
                Defaults to 0.
            ground_reflectance (float, optional):
                The reflectance of the ground.
                Defaults to 0.2.
            shade_objects (list, optional):
                A list of shades to apply to the plot.
                Defaults to an empty list.

        Returns:
            pd.DataFrame:
                A pandas DataFrame containing the radiation data.

        """
        if tilt_angle == 90:
            tilt_angle = 89.99999
        if (tilt_angle > 90) or (tilt_angle < 0):
            raise ValueError("Tilt angle must be between 0 and 90.")

        # create time-filtered sky-matrix
        smx = SkyMatrix.from_components(
            location=self.location,
            direct_normal_irradiance=self.direct_normal_radiation_collection,
            diffuse_horizontal_irradiance=self.diffuse_horizontal_radiation_collection,
            hoys=analysis_period.hoys,
            north=north,
            high_density=True,
            ground_reflectance=ground_reflectance,
        )

        # FixMe - the creation of an intersection matrix means that values do not match up with raw ladybug

        if shade_objects:
            # create a mesh with the same dumber of faces as the number of
            sensor_mesh = _create_azimuth_mesh(directions, tilt_angle)

            # create a radiation study and intersection matrix from given mesh/objects
            rd = RadiationStudy(
                sky_matrix=smx,
                study_mesh=sensor_mesh,
                context_geometry=shade_objects,
                use_radiance_mesh=True,
            )
            intersection_matrix = rd.intersection_matrix
        else:
            intersection_matrix = None

        # create rad rose
        lb_radrose = RadiationRose(
            sky_matrix=smx,
            intersection_matrix=intersection_matrix,
            direction_count=directions,
            tilt_angle=tilt_angle,
        )

        # get angles
        vectors = lb_radrose.direction_vectors
        angles = [
            vector_to_azimuth_altitude(vector=i, degrees=True)[0] for i in vectors
        ]

        # get the radiation data
        return pd.concat(
            [
                pd.Series(
                    lb_radrose.total_values,
                    index=angles,
                    name="total",
                ),
                pd.Series(
                    lb_radrose.direct_values,
                    index=angles,
                    name="direct",
                ),
                pd.Series(
                    lb_radrose.diffuse_values,
                    index=angles,
                    name="diffuse",
                ),
            ],
            axis=1,
        )

    def _tilt_orientation_factor_data(
        self,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        azimuth_count: int = 36,
        altitude_count: int = 9,
        shade_objects: tuple[Any, ...] = (),
    ) -> pd.DataFrame:
        """Get tilt-orientation-factor data for the given solar data. This is
        a set of values per tilt and orientation representing the kWh/m2
        received by a surface with that tilt and orientation.

        Args:
            analysis_period (AnalysisPeriod, optional):
                The analysis period over which radiation shall be summarised.
                Defaults to AnalysisPeriod().
            azimuth_count (int, optional):
                The number of azimuth angles to use.
                Defaults to 36.
            altitude_count (int, optional):
                The number of altitude angles to use.
                Defaults to 9.
            shade_objects (list, optional):
                A list of shade objects to apply to the plot.
                Defaults to an empty list.

        Returns:
            pd.DataFrame:
                A pandas DataFrame containing the tilt-orientation-factor data.

        """
        # warn if azimuth count is less than 12
        if azimuth_count < 12:
            warnings.warn(
                "The azimuth count is less than 12. This may result in inaccurate results."
            )
        # warn if altitude count is less than 6
        if altitude_count < 6:
            warnings.warn(
                "The altitude count is less than 6. This may result in inaccurate results."
            )

        loc = self.location.duplicate()

        # create time-filtered sky-matrix
        smx = SkyMatrix.from_components(
            location=loc,
            direct_normal_irradiance=self.direct_normal_radiation_collection,
            diffuse_horizontal_irradiance=self.diffuse_horizontal_radiation_collection,
            hoys=analysis_period.hoys,
            high_density=True,
        )

        if shade_objects:
            dome_vectors = RadiationDome.dome_vectors(
                azimuth_count=azimuth_count, altitude_count=altitude_count
            )
            faces = []
            for v in dome_vectors:
                faces.append(
                    Face3D.from_regular_polygon(
                        side_count=3,
                        radius=0.001,
                        base_plane=Plane(n=v, o=Point3D().move(v * 0.001)),
                    )
                )
            sensor_mesh = Mesh3D.from_face_vertices(faces=faces)

            # create a radiation study and intersection matrix from given mesh/objects
            rs = RadiationStudy(
                sky_matrix=smx,
                study_mesh=sensor_mesh,
                context_geometry=shade_objects,
                use_radiance_mesh=True,
            )
            intersection_matrix = rs.intersection_matrix
        else:
            intersection_matrix = None

        # create a radiation dome
        rd = RadiationDome(
            smx,
            intersection_matrix=intersection_matrix,
            azimuth_count=azimuth_count,
            altitude_count=altitude_count,
        )

        # get the raw data
        azimuths, altitudes = np.array(
            [vector_to_azimuth_altitude(i) for i in rd.direction_vectors]
        ).T

        # create a dataframe containing the results
        df = pd.DataFrame(
            {
                "azimuth": azimuths,
                "altitude": altitudes,
                "total": rd.total_values,
                "direct": rd.direct_values,
                "diffuse": rd.diffuse_values,
            }
        ).sort_values(by=["azimuth", "altitude"])

        # add missing extremity values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # FIXME - this is a hack to avoid the warning message from not slicing a dataframe correctly
            temp = df[df["altitude"] == 0]
            temp["altitude"] = 90
            temp["total"] = df[df["altitude"] == 90]["total"].values[0]
            temp["direct"] = df[df["altitude"] == 90]["direct"].values[0]
            temp["diffuse"] = df[df["altitude"] == 90]["diffuse"].values[0]

            temp2 = df[df["azimuth"] == 0]
            temp2["azimuth"] = 360

            temp3 = temp[(temp["azimuth"] == 0) & (temp["altitude"] == 90)]
            temp3["azimuth"] = 360

        df = (
            pd.concat([df, temp, temp2, temp3], axis=0)
            .reset_index(drop=True)
            .sort_values(by=["azimuth", "altitude"])
        )
        return df

    # endregion: INSTANCE METHODS

    # region: FILTERING METHODS

    def filter_by_boolean_mask(self, mask: Optional[list[bool]] = None) -> "Solar":
        """Filter the current object by a boolean mask.

        Args:
            mask (list[bool]):
                A boolean mask to filter the current object.

        Returns:
            Solar:
                A dataset describing solar radiation.

        """
        if mask is None:
            mask = [True] * len(self)

        # validations
        if not all(isinstance(i, bool) for i in mask):
            raise ValueError("mask must be a list of booleans.")
        if len(mask) != len(self):
            raise ValueError(
                "The length of the boolean mask must match the length of the current object."
            )
        if sum(mask) == 0:
            raise ValueError("No data remains within the given boolean filters.")
        if sum(mask) == len(self):
            return self

        loc = self.location.duplicate()
        loc.source = f"{self.location.source} (filtered)"

        return Solar(
            location=loc,
            datetimes=[i for i, j in zip(*[self.datetimes, mask]) if j],
            direct_normal_radiation=[
                i for i, j in zip(*[self.direct_normal_radiation, mask]) if j
            ],
            diffuse_horizontal_radiation=[
                i for i, j in zip(*[self.diffuse_horizontal_radiation, mask]) if j
            ],
            global_horizontal_radiation=[
                i for i, j in zip(*[self.global_horizontal_radiation, mask]) if j
            ],
        )

    def filter_by_analysis_period(
        self,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
    ) -> "Solar":
        """Filter the current object by a ladybug AnalysisPeriod object.

        Args:
            analysis_period (AnalysisPeriod):
                An AnalysisPeriod object.

        Returns:
            Solar:
                A dataset describing solar radiation.

        """
        mask = []
        for n, i in enumerate(self.lb_datetimes):
            mask.append(i in analysis_period.datetimes)

        # create new data
        loc = self.location.duplicate()
        loc.source = (
            f"{self.location.source} (filtered to {analysis_period_to_string(analysis_period)})",
        )
        datetimes = [i for i, j in zip(*[self.datetimes, mask]) if j]
        dni = [i for i, j in zip(*[self.direct_normal_radiation, mask]) if j]
        dhi = [i for i, j in zip(*[self.diffuse_horizontal_radiation, mask]) if j]
        ghi = [i for i, j in zip(*[self.global_horizontal_radiation, mask]) if j]

        return Solar(
            location=loc,
            datetimes=datetimes,
            direct_normal_radiation=dni,
            diffuse_horizontal_radiation=dhi,
            global_horizontal_radiation=ghi,
        )

    def filter_by_time(
        self,
        years: Optional[list[int]] = None,
        months: Optional[list[float]] = None,
        days: Optional[list[float]] = None,
        hours: Optional[list[int]] = None,
    ) -> "Solar":
        """Filter the current object by months, days, hours.

        Args:
            years (list[int], optional):
                A list of years to include.
                Default to all years.
            months (list[int], optional):
                A list of months.
                Defaults to all possible months.
            days (list[int], optional):
                A list of days.
                Defaults to all possible days.
            hours (list[int], optional):
                A list of hours.
                Defaults to all possible hours.

        Returns:
            Solar:
                A dataset describing historic solar data.

        """
        idx = self.datetimes
        filtered_by = []
        if years is None:
            years = idx.year.unique().tolist()
        else:
            filtered_by.append("year")
        if months is None:
            months = list(range(1, 13))
        else:
            filtered_by.append("month")
        if days is None:
            days = list(range(1, 32))
        else:
            filtered_by.append("day")
        if hours is None:
            hours = list(range(0, 24))
        else:
            filtered_by.append("hour")

        if len(filtered_by) > 2:
            filtered_by = ", ".join(filtered_by[:-1]) + ", and " + str(filtered_by[-1])
        elif len(filtered_by) == 2:
            filtered_by = " and ".join(filtered_by)
        elif len(filtered_by) == 1:
            filtered_by = filtered_by[0]

        # construct masks
        year_mask = idx.year.isin(years)
        month_mask = idx.month.isin(months)
        day_mask = idx.day.isin(days)
        hour_mask = idx.hour.isin(hours)
        mask = np.all([year_mask, month_mask, day_mask, hour_mask], axis=0)

        # create new data
        loc = self.location.duplicate()
        loc.source = f"{self.location.source} (filtered by {filtered_by})"
        datetimes = [i for i, j in zip(*[self.datetimes, mask]) if j]
        dni = [i for i, j in zip(*[self.direct_normal_radiation, mask]) if j]
        dhi = [i for i, j in zip(*[self.diffuse_horizontal_radiation, mask]) if j]
        ghi = [i for i, j in zip(*[self.global_horizontal_radiation, mask]) if j]

        return Solar(
            location=loc,
            datetimes=datetimes,
            direct_normal_radiation=dni,
            diffuse_horizontal_radiation=dhi,
            global_horizontal_radiation=ghi,
        )

    # endregion: FILTERING METHODS

    # region: PLOTTING METHODS

    def plot_radiation_rose(
        self,
        ax: Optional[Axes] = None,
        radiation_type: str = "total",
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        directions: int = 36,
        tilt_angle: float = 0,
        north: int = 0,
        ground_reflectance: float = 0.2,
        shade_objects: list[Any] = [],
        **kwargs,
    ) -> Axes:
        """Plot a radiation rose for the given solar data.

        Args:
            ax (Axes, optional):
                The matplotlib Axes to plot the radiation rose on.
            radiation_type (str, optional):
                The type of irradiance to plot. Defaults to total.
            analysis_period (AnalysisPeriod, optional):
                The analysis period over which radiation shall be summarised.
                Defaults to AnalysisPeriod().
            directions (int, optional):
                The number of directions to bin data into.
                Defaults to 36.
            tilt_angle (float, optional):
                The tilt (from 0 at horizon, to 90 facing the sky) to assess.
                Defaults to 0.
            north (int, optional):
                The north direction in degrees.
                Defaults to 0.
            ground_reflectance (float, optional):
                The reflectance of the ground.
                Defaults to 0.2.
            shade_objects (list, optional):
                A list of shades to apply to the plot.
                Defaults to an empty list.

        """
        if radiation_type not in ["total", "direct", "diffuse"]:
            raise ValueError(
                "radiation_type must be one of ['total', 'direct', 'diffuse']."
            )

        # create radiation results
        rad_df = self._radiation_rose_data(
            analysis_period=analysis_period,
            directions=directions,
            tilt_angle=tilt_angle,
            north=north,
            ground_reflectance=ground_reflectance,
            shade_objects=shade_objects,
        )

        # get the radiation data
        data = rad_df[radiation_type]

        # plot the radiation rose
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        if ax.name != "polar":
            raise ValueError("ax must be a polar axis.")

        # kwarg-ish vars
        # TODO - sort out kwargs
        ylim = kwargs.pop("ylim", (0, max(data) * 1.1))
        if len(ylim) != 2:
            raise ValueError("ylim must be a tuple of length 2.")
        bar_width = 1
        colors = plt.get_cmap("YlOrRd")(
            np.interp(data.values, (data.min(), data.max() * 1.05), (0, 1))
        )
        title = f"{self.location.source} at {tilt_angle}$\degree$\n{analysis_period_to_string(analysis_period)}"

        rect_s = ax.bar(
            x=np.deg2rad(data.index),
            height=data.values,
            width=((np.pi / directions) * 2) * bar_width,
            color=colors,
        )

        # add a text label to the peak value bar
        peak_value = max(data.values)
        peak_angle_deg = data.idxmax()
        peak_angle_rad = np.deg2rad(data.idxmax())
        peak_index = np.argmax(data.values)
        peak_bar = rect_s[peak_index]
        peak_bar.set_edgecolor("black")
        peak_bar.set_zorder(5)
        ax.text(
            peak_angle_rad,
            peak_value * 0.95,
            f"{peak_value:.0f}kWh/m$^2$",
            fontsize="xx-small",
            ha="right" if peak_angle_deg < 180 else "left",
            va="center",
            rotation=(90 - peak_angle_deg)
            if peak_angle_deg < 180
            else (90 - peak_angle_deg + 180),
            rotation_mode="anchor",
            color=contrasting_color(peak_bar.get_facecolor()),
            zorder=5,
        )

        # format the plot
        ax.set_title(title, fontsize="small")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(ylim)
        ax.spines["polar"].set_visible(False)
        ax.grid(True, which="both", ls="--", zorder=0, alpha=0.3)
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        plt.setp(ax.get_yticklabels(), fontsize="small")
        ax.set_xticks(np.radians((0, 90, 180, 270)), minor=False)
        ax.set_xticklabels(("N", "E", "S", "W"), minor=False, **{"fontsize": "medium"})
        ax.set_xticks(
            np.radians(
                (
                    22.5,
                    45,
                    67.5,
                    112.5,
                    135,
                    157.5,
                    202.5,
                    225,
                    247.5,
                    292.5,
                    315,
                    337.5,
                )
            ),
            minor=True,
        )
        ax.set_xticklabels(
            (
                "NNE",
                "NE",
                "ENE",
                "ESE",
                "SE",
                "SSE",
                "SSW",
                "SW",
                "WSW",
                "WNW",
                "NW",
                "NNW",
            ),
            minor=True,
            **{"fontsize": "x-small"},
        )

        return ax

    def plot_tilt_orientation_factor(
        self,
        ax: Optional[Axes] = None,
        radiation_type: str = "total",
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        azimuth_count: int = 36,
        altitude_count: int = 9,
        shade_objects: list[Any] = [],
        show_max: bool = True,
        quantiles: Union[list[float], None] = None,
        show_colorbar: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot a tilt-orientation-factor diagram for the given solar data.

        Args:
            ax (Axes, optional):
                The matplotlib Axes to plot the tilt-orientation-factor diagram on.
            radiation_type (str, optional):
                The type of irradiance to plot. Defaults to total.
            analysis_period (AnalysisPeriod, optional):
                The analysis period over which radiation shall be summarised.
                Defaults to AnalysisPeriod().
            azimuth_count (int, optional):
                The number of azimuth angles to use.
                Defaults to 36.
            altitude_count (int, optional):
                The number of altitude angles to use.
                Defaults to 9.
            shade_objects (list, optional):
                A list of shades to apply to the plot.
                Defaults to an empty list.
            show_max (bool, optional):
                If True, show the maximum value on the plot.
                Defaults to True.
            quantiles: (list[float], optional):
                A list of quantiles to use for the color levels.
                Defaults to None.
            show_colorbar (bool, optional):
                If True, show the colorbar on the plot.
                Defaults to True.
            **kwargs:
                Additional keyword arguments to pass to the plotting function.

        Return:
            ax: Axes:
                The matplotlib Axes object containing the plot.

        """
        if radiation_type not in ["total", "direct", "diffuse"]:
            raise ValueError(
                "radiation_type must be one of ['total', 'direct', 'diffuse']."
            )

        # get the data
        azimuths, altitudes, rads = self._tilt_orientation_factor_data(
            analysis_period=analysis_period,
            azimuth_count=azimuth_count,
            altitude_count=altitude_count,
            shade_objects=shade_objects,
        )[["azimuth", "altitude", radiation_type]].values.T

        # split kwargs by endpoint
        tricontourf_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "levels",
                "colors",
                "alpha",
                "cmap",
                "norm",
                "vmin",
                "vmax",
                "extend",
            ]
        }

        if ax is None:
            ax = plt.gca()

        title = f"{self.location.source}\n{analysis_period_to_string(analysis_period)}"
        ax.set_title(title)

        tcf = ax.tricontourf(
            azimuths,
            altitudes,
            rads,
            **tricontourf_kwargs,
        )

        if quantiles:
            q_values = np.quantile(rads, quantiles)
            tcl = ax.tricontour(
                azimuths,
                altitudes,
                rads,
                levels=q_values,
                colors="black",
                linewidths=0.5,
            )

            def cl_fmt(x):
                return f"{x:,.0f}kWh/m$^2$"

            _ = ax.clabel(tcl, fontsize="small", fmt=cl_fmt)

        if show_max:
            # get max value and location
            max_value = max(rads)
            max_indices = np.where(rads == max_value)
            max_azimuth = np.mean(azimuths[max_indices])
            max_altitude = np.mean(altitudes[max_indices])
            ax.scatter(
                max_azimuth,
                max_altitude,
                marker="o",
                color="black",
                s=50,
                zorder=10,
            )
            ax.text(
                max_azimuth,
                max_altitude + 1,
                f"{max_value:,.0f}kW/m$^2$\n{max_azimuth:.0f}°,{max_altitude:.0f}°",
                fontsize="small",
                ha="left" if max_azimuth < 300 else "right",
                va="bottom" if max_altitude < 80 else "top",
                color="black",
            )
            ax.axvline(max_azimuth, ymax=max_altitude / 90, color="black", ls="--")
            ax.axhline(max_altitude, xmax=max_azimuth / 360, color="black", ls="--")

        if shade_objects:
            ax.text(
                1,
                1,
                "*includes context shading",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
            )

        if show_colorbar:
            # add colorbar
            cb = plt.colorbar(
                tcf,
                ax=ax,
                orientation="vertical",
                drawedges=False,
                fraction=0.05,
                aspect=25,
                pad=0.02,
                label="Cumulative irradiance (kWh/m$^2$)",
            )
            cb.outline.set_visible(False)
            if quantiles:
                quant_vals = np.quantile(rads, quantiles)
                for quantile_val in quant_vals:
                    cb.ax.plot(
                        [0, 1],
                        [quantile_val, quantile_val],
                        scalex=False,
                        scaley=True,
                        color="k",
                        ls="-",
                        alpha=0.5,
                    )

        ax.set_xlim(0, 360)
        ax.set_ylim(0, 90)
        ax.xaxis.set_major_locator(MultipleLocator(base=30))
        ax.yaxis.set_major_locator(MultipleLocator(base=10))
        ax.set_xlabel("Orientation (clockwise from North at 0°)")
        ax.set_ylabel("Tilt (0° facing the horizon, 90° facing the sky)")

        return ax

    def plot_sunpath(
        self,
        ax: Optional[Axes] = None,
        other_data: Union[list[float], None] = None,
        other_datatype: Optional[DataTypeBase] = None,
        cmap: Union[Colormap, str] = "viridis",
        norm: Optional[BoundaryNorm] = None,
        sun_size: float = 10,
        show_grid: bool = True,
        show_legend: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot a sun-path for the given Location and analysis period.

        Args:
            location (Location):
                A ladybug Location object.
            ax (plt.Axes, optional):
                A matplotlib Axes object. Defaults to None.
            analysis_period (AnalysisPeriod, optional):
                _description_. Defaults to None.
            data_collection (HourlyContinuousCollection, optional):
                An aligned data collection. Defaults to None.
            cmap (str, optional):
                The colormap to apply to the aligned data_collection. Defaults to None.
            norm (BoundaryNorm, optional):
                A matplotlib BoundaryNorm object containing colormap boundary mapping information.
                Defaults to None.
            sun_size (float, optional):
                The size of each sun in the plot. Defaults to 0.2.
            show_grid (bool, optional):
                Set to True to show the grid. Defaults to True.
            show_legend (bool, optional):
                Set to True to include a legend in the plot if data_collection passed. Defaults to True.

        Returns:
            plt.Axes:
                A matplotlib Axes object.

        """
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})

        title = kwargs.pop("title", self.location.source)

        def to_spherical(point3d):
            """Convert a 3D point to spherical coordinates (r, theta, phi).
            r is the distance from the origin to the point,
            theta is the angle in the x-y plane from the x-axis, with y-north at 0 degrees,
            and phi is the angle from the z-axis.
            """
            r = np.sqrt(point3d.x**2 + point3d.y**2 + point3d.z**2)
            theta = np.arctan2(point3d.y, point3d.x)
            phi = np.arccos(point3d.z / r)
            # rotate theta -90 degrees to make 0 degrees north
            theta = theta - np.pi / 2
            return r, theta, phi

        radius = 1

        if other_data is not None:
            raise NotImplementedError("other_data is not implemented yet")
            # todo - implement colormap to other dat, and other data type (other_datatype: DataTypeBase = None)
            if len(other_data) != len(self):
                raise ValueError("other_data must be the same length")

        sunpath: Sunpath = self.sunpath

        # plot analemma
        analemma_polylines_3d = sunpath.hourly_analemma_polyline3d(
            steps_per_month=2, radius=radius
        ) + [
            Polyline3D(i.subdivide_evenly(24))
            for i in sunpath.monthly_day_arc3d(radius=radius)
        ]
        for polyline in analemma_polylines_3d:
            _, theta, phi = np.array([to_spherical(i) for i in polyline]).T
            ax.plot(theta, phi, linewidth=1, color="black", zorder=1)

        # plot suns
        suns = [sun for sun in self.suns if sun.altitude > 0]
        _, theta, phi = np.array(
            [to_spherical(i.position_3d(radius=1)) for i in suns]
        ).T
        ax.scatter(theta, phi, s=1, c="orange", zorder=1)

        # format plot
        ax.spines["polar"].set_visible(False)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        ax.set_title(title)

        return ax

    def plot_skymatrix(
        self,
        ax: Axes = None,
        radiation_type: str = "total",
        density: int = 1,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        **kwargs,
    ) -> Axes:
        # split kwargs by endpoint
        plot_kwargs = {
            k: v for k, v in kwargs.items() if k in ["levels", "alpha", "cmap", "norm"]
        }

        # create wea
        wea = self.to_wea(timestep=analysis_period.timestep).filter_by_analysis_period(
            analysis_period
        )
        wea_duration = len(wea) / wea.timestep
        wea_folder = Path(tempfile.gettempdir())
        wea_path = wea_folder / "skymatrix.wea"
        wea_file = wea.write(wea_path.as_posix())

        # run gendaymtx
        gendaymtx_exe = (Path(lbr_folders.radbin_path) / "gendaymtx.exe").as_posix()
        cmds = [gendaymtx_exe, "-m", str(density), "-d", "-O1", "-A", wea_file]
        with subprocess.Popen(cmds, stdout=subprocess.PIPE, shell=True) as process:
            stdout = process.communicate()
        dir_data_str = stdout[0].decode("ascii")
        cmds = [gendaymtx_exe, "-m", str(density), "-s", "-O1", "-A", wea_file]
        with subprocess.Popen(cmds, stdout=subprocess.PIPE, shell=True) as process:
            stdout = process.communicate()
        diff_data_str = stdout[0].decode("ascii")

        def _broadband_rad(data_str: str) -> list[float]:
            _ = data_str.split("\r\n")[:8]
            data = np.array(
                [[float(j) for j in i.split()] for i in data_str.split("\r\n")[8:]][
                    1:-1
                ]
            )
            patch_values = (
                np.array([0.265074126, 0.670114631, 0.064811243]) * data
            ).sum(axis=1)
            patch_steradians = np.array(ViewSphere().dome_patch_weights(density))
            broadband_radiation = patch_values * patch_steradians * wea_duration / 1000
            return broadband_radiation

        dir_vals = _broadband_rad(dir_data_str)
        diff_vals = _broadband_rad(diff_data_str)

        # create the ,mesh to assign data to
        msh = ViewSphere().dome_patches(density)[0]
        # reshape the data to align with mesh faces
        direct_values = np.concatenate(
            [dir_vals[:-1], np.repeat(dir_vals[0], len(msh.faces) - len(dir_vals) + 1)]
        )
        diffuse_values = np.concatenate(
            [
                diff_vals[:-1],
                np.repeat(diff_vals[0], len(msh.faces) - len(diff_vals) + 1),
            ]
        )
        # create geo-dataframe with data-linked geometry
        shapes = to_shapely(msh)
        df = pd.DataFrame(
            data=[direct_values, diffuse_values, direct_values + diffuse_values],
            index=[
                "direct",
                "diffuse",
                "total",
            ],
        ).T
        gdf = gpd.GeoDataFrame(df, geometry=list(shapes.geoms))

        if ax is None:
            ax = plt.gca()

        # todo - additional plot formatting in here ...

        return gdf.plot(ax=ax, column=radiation_type, **plot_kwargs)

    def plot_hours_sunlight(self, ax: Optional[Axes] = None) -> Axes:
        ax = plt.gca()

        df = self.sunrise_sunset
        adf = df.filter(regex="hours")[
            ["actual", "apparent", "astronomical", "civil", "nautical", "night"]
        ].droplevel(1, axis=1)
        solstices_equinoxes = self.solstices_equinoxes
        renamer = {
            "actual": "Daytime",
            "apparent": "Apparent daytime",
            "astronomical": "Astronomical twilight",
            "civil": "Civil twilight",
            "nautical": "Nautical twilight",
            "night": "Night-time",
        }
        ax = plt.gca()
        colors = ["#FCE49D", "#dbc892ff", "#B9AC86", "#908A7A", "#817F76", "#717171"]
        base = np.zeros_like(adf.index, dtype=float)
        for n, (col_name, col_values) in enumerate(adf.items()):
            vals = np.array(col_values, dtype=float)
            ax.fill_between(
                x=adf.index,
                y1=base,
                y2=base + vals,
                color=colors[n],
                label=renamer[col_name],
            )
            base += vals

        # add solstice and equinox lines
        for col_name, col_values in solstices_equinoxes.items():
            dt = col_values.values[0]
            ax.axvline(x=dt, color="black", ls="--", alpha=0.5)
            # add sunrise/set times for key dates too

            try:
                sunrise_time = pd.to_datetime(
                    df[(df.index.month == dt.month) & (df.index.day == dt.day)][
                        ("actual", "sunrise")
                    ].values[0]
                ).strftime("%H:%M")
            except AttributeError:
                sunrise_time = np.nan
            try:
                sunset_time = pd.to_datetime(
                    df[(df.index.month == dt.month) & (df.index.day == dt.day)][
                        ("actual", "sunset")
                    ].values[0]
                ).strftime("%H:%M")
            except AttributeError:
                sunset_time = np.nan
            ax.text(
                (dt + timedelta(days=1)) if dt.month < 6 else (dt - timedelta(days=1)),
                23.75,
                f"{col_name.title()}\n{dt.strftime('%d %b')}\nSunrise: {sunrise_time}\nSunset: {sunset_time}",
                rotation=0,
                ha="left" if dt.month < 6 else "right",
                va="top",
                fontsize="small",
                alpha=0.5,
            )

        ax.set_xlim(adf.index[0], adf.index[-1])
        ax.set_ylim(0, 24)
        ax.set_yticks(np.arange(0, 25, 3))
        ax.set_title(f"Hours of daylight and twilight\n{self}")
        ax.set_ylabel("Hours")
        ax.legend(
            bbox_to_anchor=(0.5, -0.05),
            loc="upper center",
            ncol=6,
            title="Day period",
        )
        ax.grid(
            which="both",
            ls="--",
            alpha=0.5,
        )
        return ax

    def plot_solar_elevation_azimuth(self, ax: Optional[Axes] = None) -> Axes:
        """Plot the solar elevation and azimuth for a location.

        Args:
            ax (plt.Axes, optional):
                A matplotlib axes to plot on. Defaults to None.

        Returns:
            Axes:
                The matplotlib axes.

        """
        # create suns
        sp = self.sunpath
        idx = pd.date_range("2017-01-01 00:00:00", "2018-01-01 00:00:00", freq="10min")
        suns = [sp.calculate_sun_from_date_time(i) for i in idx]
        a = pd.DataFrame(index=idx)
        a["altitude"] = [i.altitude for i in suns]
        a["azimuth"] = [i.azimuth for i in suns]

        # create cmap
        cmap = ListedColormap(
            colors=(
                "#809FB4",
                "#90ACBE",
                "#9FC7A2",
                "#90BF94",
                "#9FC7A2",
                "#CF807A",
                "#C86C65",
                "#CF807A",
                "#C6ACA0",
                "#BD9F92",
                "#C6ACA0",
                "#90ACBE",
                "#809FB4",
            ),
            name="noname",
        )
        cmap.set_over("#809FB4")
        cmap.set_under("#809FB4")
        norm = BoundaryNorm(
            boundaries=[
                0,
                22.5,
                45,
                67.5,
                112.5,
                135,
                157.5,
                202.5,
                225,
                247.5,
                292.5,
                315,
                337.5,
                360,
            ],
            ncolors=cmap.N,
        )

        # create plot
        if ax is None:
            ax = plt.gca()

        series = a["azimuth"]
        day_time_matrix = (
            series.dropna()
            .to_frame()
            .pivot_table(
                columns=series.dropna().index.date, index=series.dropna().index.time
            )
        )
        x = mdates.date2num(day_time_matrix.columns.get_level_values(1))
        y = mdates.date2num(
            pd.to_datetime([f"2017-01-01 {i}" for i in day_time_matrix.index])
        )
        z = day_time_matrix.values
        pcm = ax.pcolormesh(
            x,
            y,
            z[:-1, :-1],
            cmap=cmap,
            norm=norm,
        )
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        ax.yaxis_date()
        ax.yaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        ax.tick_params(labelleft=True, labelbottom=True)

        # hide all spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in ax.get_xticks():
            ax.axvline(i, color="w", ls=":", lw=0.5, alpha=0.5)
        for i in ax.get_yticks():
            ax.axhline(i, color="w", ls=":", lw=0.5, alpha=0.5)
        cb = plt.colorbar(
            pcm,
            ax=ax,
            orientation="horizontal",
            drawedges=False,
            fraction=0.05,
            aspect=100,
            pad=0.075,
            # extend=extend,
            label=series.name.title(),
        )
        cb.outline.set_visible(False)
        cb.set_ticks(
            [0, 45, 90, 135, 180, 225, 270, 315, 360],
            labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"],
        )
        ax.set_title(f"Sun Altitude and Azimuth\n{self}")
        ylim = ax.get_ylim()

        # create matrix of month/day/hour for pcolormesh
        pvt = a.pivot_table(columns=a.index.date, index=a.index.time)

        # plot the contours for sun positions
        x = mdates.date2num(pvt["altitude"].columns)
        y = mdates.date2num(pd.to_datetime([f"2017-01-01 {i}" for i in pvt.index]))
        z = pvt["altitude"].values
        # z = np.ma.masked_array(z, mask=z < 0)
        ct = ax.contour(x, y, z, colors="k", levels=np.arange(0, 91, 10))
        ax.clabel(ct, inline=1, fontsize="small")
        ax.set_ylim(ylim)

        return ax

    # endregion: PLOTTING METHODS

def pv_yield(
    epw_file: Path,
    azimuth: float = 180,
    altitude: float = 0,
    context_shades: Optional[list[Shade]] = [],
    rated_efficiency: float = 0.15,
    active_area_fraction: float = 0.9,
    module_type: Optional[Any] = None,
    mounting_type: Literal["FixedOpenRack", "FixedRoofMounted", "OneAxis", "OneAxisBacktracking", "TwoAxis"] = "FixedOpenRack",
    system_loss_fraction: float = 0.14,
    tracking_ground_coverage_ratio: float = 0.4,
    inverter_efficiency: float = 0.96,
    inverter_dc_to_ac_size_ratio: float = 1.1,
) -> pd.DataFrame:
    """Estimate PV yield for a given azimuth and tilt, and PV configuration. This
        method does not include effects from overshading or self-shading.

        Args:
            epw_file: The path to the EPW file to use for the simulation. This file is used
                to determine the solar radiation and temperature conditions at the site.
            azimuth: A number between 0 and 360 for the azimuth of the plane of the
                photovoltaic array. The azimuth is the angle between the plane normal and
                true north, which is 0 degrees. The azimuth is positive in the clockwise
                direction from true north. (Default: 180 degrees, which is south).
            altitude: A number between 0 and 90 for the tilt of the plane of the photovoltaic
                array. The tilt is the angle between the plane normal and the horizontal
                plane. A tilt of 0 degrees is horizontal and a tilt of 90 degrees is
                vertical. (Default: 0 degrees).
            rated_efficiency: A number between 0 and 1 for the rated nameplate efficiency
                of the photovoltaic solar cells under standard test conditions (STC).
                Standard test conditions are 1,000 Watts per square meter solar
                irradiance, 25 degrees C cell temperature, and ASTM G173-03 standard
                spectrum. Nameplate efficiencies reported by manufacturers are typically
                under STC. Standard poly- or mono-crystalline silicon modules tend to have
                rated efficiencies in the range of 14-17%. Premium high efficiency
                mono-crystalline silicon modules with anti-reflective coatings can have
                efficiencies in the range of 18-20%. Thin film photovoltaic modules
                typically have efficiencies of 11% or less. (Default: 0.15 for standard
                silicon solar cells).
            active_area_fraction: The fraction of the parent Shade geometry that is
                covered in active solar cells. This fraction includes the difference
                between the PV panel (aka. PV module) area and the active cells within
                the panel as well as any losses for how the (typically rectangular) panels
                can be arranged on the Shade geometry. When the parent Shade geometry
                represents just the solar panels, this fraction is typically around 0.9
                given that the metal framing elements of the panel reduce the overall
                active area. (Default: 0.9, assuming parent Shade geometry represents
                only the PV panel geometry).
            module_type: Text to indicate the type of solar module. This is used to
                determine the temperature coefficients used in the simulation of the
                photovoltaic modules. Choose from the three options below. If None,
                the module_type will be inferred from the rated_efficiency of these
                PVProperties using the rated efficiencies listed below. (Default: None).

                * Standard - 12% <= rated_efficiency < 18%
                * Premium - rated_efficiency >= 18%
                * ThinFilm - rated_efficiency < 12%

            mounting_type: Text to indicate the type of mounting and/or tracking used
                for the photovoltaic array. Note that the OneAxis options have an axis
                of rotation that is determined by the azimuth of the parent Shade
                geometry. Also note that, in the case of one or two axis tracking,
                shadows on the (static) parent Shade geometry still reduce the
                electrical output, enabling the simulation to account for large
                context geometry casting shadows on the array. However, the effects
                of smaller detailed shading may be improperly accounted for and self
                shading of the dynamic panel geometry is only accounted for via the
                tracking_ground_coverage_ratio property on this object. Choose from
                the following. (Default: FixedOpenRack).
                * FixedOpenRack - ground or roof mounting where the air flows freely
                * FixedRoofMounted - mounting flush with the roof with limited air flow
                * OneAxis - a fixed tilt and azimuth, which define an axis of rotation
                * OneAxisBacktracking - same as OneAxis but with controls to reduce self-shade
                * TwoAxis - a dynamic tilt and azimuth that track the sun
            system_loss_fraction: A number between 0 and 1 for the fraction of the
                electricity output lost due to factors other than EPW climate conditions,
                panel efficiency/type, active area, mounting, and inverter conversion from
                DC to AC. Factors that should be accounted for in this input include
                soiling, snow, wiring losses, electrical connection losses, manufacturer
                defects/tolerances/mismatch in cell characteristics, losses from power
                grid availability, and losses due to age or light-induced degradation.
                Losses from these factors tend to be between 10-20% but can vary widely
                depending on the installation, maintenance and the grid to which the
                panels are connected. The loss_fraction_from_components staticmethod
                on this class can be used to estimate this value from the various
                factors that it is intended to account for. (Default: 0.14).
            tracking_ground_coverage_ratio: A number between 0 and 1 that only applies to
                arrays with one-axis tracking mounting_type. The ground coverage ratio (GCR)
                is the ratio of module surface area to the area of the ground beneath
                the array, which is used to account for self shading of single-axis panels
                as they move to track the sun. A GCR of 0.5 means that, when the modules
                are horizontal, half of the surface below the array is occupied by
                the array. An array with wider spacing between rows of modules has a
                lower GCR than one with narrower spacing. A GCR of 1 would be for an
                array with no space between modules, and a GCR of 0 for infinite spacing
                between rows. Typical values range from 0.3 to 0.6. (Default: 0.4).
            inverter_efficiency: A number between 0 and 1 for the load centers's
                inverter nominal rated DC-to-AC conversion efficiency. An inverter
                converts DC power, such as that output by photovoltaic panels, to
                AC power, such as that distributed by the electrical grid and is available
                from standard electrical outlets. Inverter efficiency is defined
                as the inverter's rated AC power output divided by its rated DC power
                output. (Default: 0.96).
            inverter_dc_to_ac_size_ratio: A positive number (typically greater than 1) for
                the ratio of the inverter's DC rated size to its AC rated size. Typically,
                inverters are not sized to convert the full DC output under standard
                test conditions (STC) as such conditions rarely occur in reality and
                therefore unnecessarily add to the size/cost of the inverter. For a
                system with a high DC to AC size ratio, during times when the
                DC power output exceeds the inverter's rated DC input size, the inverter
                limits the array's power output by increasing the DC operating voltage,
                which moves the arrays operating point down its current-voltage (I-V)
                curve. The default value of 1.1 is reasonable for most systems. A
                typical range is 1.1 to 1.25, although some large-scale systems have
                ratios of as high as 1.5. The optimal value depends on the system's
                location, array orientation, and module cost. (Default: 1.1).
            output_directory: The directory where the output CSV and images will be
                saved. If None, the output will be saved in the simulation directory.
            ylim: A list of two floats indicating the y-axis limits for the monthly
                PV yield bar chart. If None, the y-axis limits will be determined
                automatically. (Default: None).

        Returns:
            pd.DataFrame: A pandas DataFrame containing the PV yield data and EPW metrics.

        Examples
        --------
        >>> from lbttk.plot.matplotlib import heatmap
        >>> from lbttk.solar.solar import pv_yield
        >>> from ladybug_geometry.geometry3d import Point3D, Face3D, LineSegment3D, Vector3D, Plane
        >>> from honeybee.shade import Shade
        >>> import numpy as np
        >>> import pandas as pd
        >>> from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval

        >>> shd_dynamic_south = Shade(
        >>>     "shade",
        >>>     Face3D.from_extrusion(
        >>>         LineSegment3D.from_end_points(Point3D(-8, -5, 0), Point3D(8, -5, 0)),
        >>>         Vector3D(0, 0, 25),
        >>>     ),
        >>> )
        >>> shd_dynamic_south.properties.energy.transmittance_schedule = ScheduleFixedInterval(
        >>>     identifier="transmissivity", values=abs((np.arange(8760) - 4380) / 4380).tolist()
        >>> )
        >>> shd_opaque_above = Shade(
        >>>     identifier="shade_above",
        >>>     geometry=Face3D.from_regular_polygon(
        >>>         side_count=6, radius=10, base_plane=Plane(o=Point3D(0, 0, 10))
        >>>     ),
        >>> )
        >>> df = pv_yield(
        >>>     epw_file=".\file.epw",
        >>>     context_shades=[shd_dynamic_south, shd_opaque_above],
        >>> )

        >>> heatmap(
        >>>     df.filter(regex="Facility Total Produced Electricity Energy Intensity").squeeze()
        >>> )
    """

    epw_file = Path(epw_file)
    epw = EPW(epw_file)

    # construct the unique hash for this simulation
    h = hashlib.blake2b(digest_size=20, person=b"pv_yield")

    # get all inputs, and hash
    cfg = {}
    for k, v in locals().items():
        if k in ["h", "epw", "cfg"]:
            continue
        elif isinstance(v, (list, tuple)):
            if all(isinstance(i, Shade) for i in v):
                cfg[k] = [i.to_dict() for i in v]
        else:
            cfg[k] = v
    h.update(json.dumps(cfg, sort_keys=True, cls=AllPowerfulEncoder).encode("utf-8"))
    hash_str = h.hexdigest()

    # check if simulation results already exist, and load them instead!
    results_dir = Path(hb_folders.default_simulation_folder) / ".pv_yield_cache"
    results_dir.mkdir(parents=True, exist_ok=True)

    directory = results_dir / hash_str
    results_file = directory / "results.csv"

    if results_file.exists():
        CONSOLE_LOGGER.info(
            f"Loading cached PV yield results for {epw_file.name} ({altitude=}, {azimuth=})"
        )
        df = pd.read_csv(results_file, index_col=0, parse_dates=True, header=[0, 1, 2])
    else:
        OUTPUTS = [
            "Generator Produced DC Electricity Energy",
            "Generator PV Cell Temperature",
            "Plane of Array Irradiance",
            "Shaded Percent",
            "Inverter DC to AC Efficiency",
            "Inverter DC Input Electricity Energy",
            "Inverter AC Output Electricity Energy",
            "Inverter Conversion Loss Energy",
            "Inverter Conversion Loss Decrement Energy",
            "Inverter Thermal Loss Energy",
            "Inverter Ancillary AC Electricity Energy",
            "Electric Load Center Produced Electricity Energy",
            "Electric Load Center Produced Thermal Energy",
            "Facility Net Purchased Electricity Energy",
            "Facility Total Produced Electricity Energy",
        ]

        if altitude < 0:
            raise ValueError("Altitude must be greater than or equal to 0.")

        normal = azimuth_altitude_to_vector(azimuth=azimuth, altitude=altitude)
        plane = Plane(n=normal, o=Point3D())
        face = Face3D.from_regular_polygon(
            side_count=4, radius=np.sqrt(2) / 2, base_plane=plane
        )
        shade = Shade.from_vertices(
            identifier="pv_panel_geometry", vertices=face.vertices, is_detached=True
        )

        # create PV properties and assign to shade
        pv_props = PVProperties(
            identifier="pv_panel",
            rated_efficiency=rated_efficiency,
            active_area_fraction=active_area_fraction,
            module_type=module_type,
            mounting_type=mounting_type,
            system_loss_fraction=system_loss_fraction,
            tracking_ground_coverage_ratio=tracking_ground_coverage_ratio,
        )
        module_type = pv_props.module_type
        shade.properties.energy.pv_properties = pv_props

        shades = [shade]
        if context_shades is not None:
            if not all(isinstance(i, Shade) for i in context_shades):
                raise ValueError("All context_shades must be Shade objects.")
            if not all(
                i.properties.energy.pv_properties is None for i in context_shades
            ):
                raise ValueError("context_shades cannot have PV properties assigned.")
            shades.extend(context_shades)

        # create the model to simulate
        model = Model.from_objects("Generation_Loads", shades)
        model.rooms_to_orphaned()

        # add ground
        soil_construction = opaque_construction_by_identifier("Mud")
        model.properties.energy.generate_ground_room(soil_construction)

        # add inverter efficiency and size
        energy_load_center = ElectricLoadCenter(
            inverter_efficiency=inverter_efficiency,
            inverter_dc_to_ac_size_ratio=inverter_dc_to_ac_size_ratio,
        )
        model.properties.energy.electric_load_center = energy_load_center  # type: ignore

        # process the simulation folder name and the directory
        sch_directory: Path = directory / "schedules"

        # create simulation parameters for the coarsest/fastest E+ sim possible
        _sim_par_ = SimulationParameter()
        _sim_par_.timestep = 6
        _sim_par_.shadow_calculation.solar_distribution = "FullExteriorWithReflections"
        _sim_par_.output.reporting_frequency = "Hourly"
        for output in OUTPUTS:
            _sim_par_.output.add_output(output)
        _sim_par_.output.include_html = False
        _sim_par_.simulation_control.do_zone_sizing = False
        _sim_par_.simulation_control.do_system_sizing = False
        _sim_par_.simulation_control.do_plant_sizing = False

        # create the strings for simulation parameters and model
        ver_str = energyplus_idf_version()
        sim_par_str = _sim_par_.to_idf()
        model_str = model.to.idf(
            model,
            schedule_directory=sch_directory.as_posix(),
            patch_missing_adjacencies=True,
        )
        idf_str = "\n\n".join([ver_str, sim_par_str, model_str])

        # write the final string into an IDF
        idf = directory / "in.idf"
        write_to_file_by_name(directory.as_posix(), "in.idf", idf_str, True)

        CONSOLE_LOGGER.info(
            f"Calculating PV yield for {epw_file.name} ({altitude=}, {azimuth=})"
        )
        # run the IDF through EnergyPlus
        sql, _, _, _, err = run_idf(idf.as_posix(), epw_file.as_posix(), silent=True)
        if sql is None and err is not None:  # something went wrong; parse the errors
            err_obj = Err(err)
            print(err_obj.file_contents)
            for error in err_obj.fatal_errors:
                raise Exception(error)

        # parse the result sql and get the monthly data collections
        sql_obj = SQLiteResult(sql)
        collections = []
        for output in OUTPUTS:
            for col in sql_obj.data_collections_by_output_name(output):
                col: HourlyContinuousCollection
                if isinstance(col.header.data_type, Energy):
                    col = col.to_unit("Wh").normalize_by_area(
                        area=shade.area, area_unit="m2"
                    )
                col.header.metadata["time-zone"] = epw.location.time_zone
                collections.append(col)

        # add metadata to collections
        for col in collections:
            col.header.metadata["pv_rated_efficiency"] = rated_efficiency
            col.header.metadata["pv_active_area_fraction"] = active_area_fraction
            col.header.metadata["pv_module_type"] = module_type
            col.header.metadata["pv_mounting_type"] = mounting_type
            col.header.metadata["pv_system_loss_fraction"] = system_loss_fraction
            col.header.metadata["pv_tracking_ground_coverage_ratio"] = (
                tracking_ground_coverage_ratio
            )
            col.header.metadata["pv_inverter_efficiency"] = inverter_efficiency
            col.header.metadata["pv_inverter_dc_to_ac_size_ratio"] = (
                inverter_dc_to_ac_size_ratio
            )
            col.header.metadata["pv_azimuth"] = azimuth
            col.header.metadata["pv_altitude"] = altitude

        # add epw metrics
        collections.extend(
            [
                epw.dry_bulb_temperature,
                epw.global_horizontal_radiation,
                epw.direct_normal_radiation,
                epw.diffuse_horizontal_radiation,
            ]
        )

        # convert collections to pandas DataFrame
        df = pd.concat([to_pandas(i) for i in collections], axis=1).sort_index(axis=1)

        df.to_csv(results_file, index=True, header=True)

        with open(directory / "config.json", "w") as f:
            json.dump(cfg, f, indent=4, sort_keys=True, cls=AllPowerfulEncoder)

    return df

def plot_pv_yield_temperature_relationship(
    epw_file: Path,
    azimuth: float = 180,
    altitude: float = 0,
    context_shades: Optional[list[Shade]] = [],
    rated_efficiency: float = 0.15,
    active_area_fraction: float = 0.9,
    module_type: Optional[Any] = None,
    mounting_type: Literal["FixedOpenRack", "FixedRoofMounted", "OneAxis", "OneAxisBacktracking", "TwoAxis"] = "FixedOpenRack",
    system_loss_fraction: float = 0.14,
    tracking_ground_coverage_ratio: float = 0.4,
    inverter_efficiency: float = 0.96,
    inverter_dc_to_ac_size_ratio: float = 1.1,
    ax: Optional[Axes] = None,
) -> Axes:
    
    df = pv_yield(
        epw_file=epw_file,
        azimuth=azimuth,
        altitude=altitude,
        context_shades=context_shades,
        rated_efficiency=rated_efficiency,
        active_area_fraction=active_area_fraction,
        module_type=module_type,
        mounting_type=mounting_type,
        system_loss_fraction=system_loss_fraction,
        tracking_ground_coverage_ratio=tracking_ground_coverage_ratio,
        inverter_efficiency=inverter_efficiency,
        inverter_dc_to_ac_size_ratio=inverter_dc_to_ac_size_ratio,
    )
    if ax is None:
        ax = plt.gca()
    
    rad_on_panel = df.filter(regex="Plane of Array Irradiance").squeeze()

    panel_temp = df.filter(regex="Generator PV Cell Temperature").squeeze()

    air_temp = df.filter(regex="Dry Bulb Temperature").squeeze()

    pv_y = df.filter(
        regex="Facility Total Produced Electricity Energy Intensity"
    ).squeeze()

    eff = pv_y / rad_on_panel
    ax.scatter(air_temp, eff, c=rad_on_panel, s=2)
    # add colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Plane of Array Irradiance (W/m2)")
    ax.set_ylabel("Whole System Efficiency")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    ax.set_xlabel("Air Temperature (°C)")

    total_ac_produced = df.filter(
        regex="Facility Total Produced Electricity Energy Intensity"
    ).squeeze()  # Wh/m2
    total_dc_produced = df.filter(
        regex="Generator Produced DC Electricity Energy Intensity"
    ).squeeze()  # Wh/m2

    pv_description = (
        f"Total AC: {total_ac_produced.sum() / 1000:.1f}kWh/m2\n"
        f"Total DC: {total_dc_produced.sum() / 1000:.1f}kWh/m2\n"
        f"PV azimuth: {azimuth}$\degree$\n"  # type: ignore
        f"PV altitude: {altitude}$\degree$\n"  # type: ignore
        f"PV efficiency: {rated_efficiency:.1%}\n"
        f"PV active area: {active_area_fraction:.1%}\n"
        f"PV module type: {module_type}\n"
        f"PV mounting type: {mounting_type}\n"
        f"PV system loss: {system_loss_fraction:0.1%}\n"
        f"Inverter Efficiency: {inverter_efficiency:.1%}\n"
    )
    ax.text(
        1,
        0,
        pv_description,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize="xx-small",
    )

    plt.tight_layout()
    return ax
