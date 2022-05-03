from __future__ import annotations
import json
import shutil
from typing import Any, Dict, List
import warnings

from honeybee_extension.results import load_ill, load_pts, load_res, make_annual
from cached_property import cached_property
import numpy as np
import pandas as pd
from external_comfort.encoder import Encoder
from dataclasses import dataclass, field
from ladybug.datacollection import HourlyContinuousCollection
from pathlib import Path
from external_comfort.external_comfort import ExternalComfort, ExternalComfortResult
from external_comfort.moisture import MoistureSource
from external_comfort.typology import Typology, TypologyResult, Shelter
from ladybug_comfort.utci import universal_thermal_climate_index
from scipy.interpolate import interp1d

v_utci = np.vectorize(universal_thermal_climate_index)

class SpatialEncoder(Encoder):
    """A JSON encoder for the Typology and TypologyResult classes."""

    def default(self, obj):
        if isinstance(obj, ExternalComfort):
            return obj.to_dict()
        if isinstance(obj, ExternalComfortResult):
            return obj.to_dict()
        if isinstance(obj, Shelter):
            return obj.to_dict()
        if isinstance(obj, Typology):
            return obj.to_dict()
        if isinstance(obj, TypologyResult):
            return obj.to_dict()
        if isinstance(obj, SpatialComfort):
            return obj.to_dict()
        if isinstance(obj, SpatialComfortResult):
            return obj.to_dict()
        if isinstance(obj, MoistureSource):
            return obj.to_dict()
        return super(SpatialEncoder, self).default(obj)

@dataclass
class SpatialComfort:
    def __init__(self, simulation_directory: Path, external_comfort_result: ExternalComfortResult) -> SpatialComfort:
        self.simulation_directory = Path(simulation_directory)

        # Tidy results folder and check for requisite datasets
        self._simulation_validity()
        self._remove_temp_files()

        self.external_comfort_result = external_comfort_result
        self.unshaded_typology_result = TypologyResult(Typology(name="unshaded", shelters=None), self.external_comfort_result)
        self.shaded_typology_result = TypologyResult(Typology(name="shaded", shelters=[Shelter(porosity=0, altitude_range=[0, 90], azimuth_range=[0, 360])], ), self.external_comfort_result)

        self.moisture_sources = self._load_moisture_sources()
        self.epw = self.external_comfort_result.external_comfort.epw

    def _simulation_validity(self) -> None:

        annual_irradiance_directory = self.simulation_directory / "annual_irradiance"
        if not annual_irradiance_directory.exists():
            raise ValueError(
                f"Annual-irradiance data is not available at {annual_irradiance_directory}."
            )

        sky_view_directory = self.simulation_directory / "sky_view"
        if not (sky_view_directory).exists():
            raise ValueError(f"Sky-view data is not available at {sky_view_directory}.")

    def _remove_temp_files(self) -> None:
        """Remove initial results files from simulated case to save on disk space."""        
        directories = list(self.simulation_directory.glob('**'))
        for dir in directories:
            if dir.name == "initial_results":
                shutil.rmtree(dir)
    
    def _load_moisture_sources(self) -> List[MoistureSource]:
        """Load the water bodies from the simulation directory.

        Returns:
            Dict[str, Any]: A dictionary contining the boundary vertices of any water bodies, and locations of any point-sources.
        """

        moisture_sources_path = (
            self.simulation_directory / "moisture_sources.json"
        )

        try:
            moisture_sources = MoistureSource.from_json(moisture_sources_path)
            
            if len(moisture_sources) == 0:
                raise ValueError(f"No moisture sources found in {moisture_sources_path}")
                
            return moisture_sources
        except FileNotFoundError as e:
            warnings.warn("No moisture_sources.json found in simulation directory - advanced moisture-impacted UTCI not possible.")
            return []


class SpatialComfortResult:
    def __init__(self, spatial_comfort: SpatialComfort) -> SpatialComfortResult:
        self.spatial_comfort = spatial_comfort

    @cached_property
    def dry_bulb_temperature(self) -> List[float]:
        """Return the dry bulb temperature values from the simulation."""
        return np.array(self.spatial_comfort.epw.dry_bulb_temperature.values)
    
    @cached_property
    def relative_humidity(self) -> List[float]:
        """Return the relative humidity values from the simulation."""
        return np.array(self.spatial_comfort.epw.relative_humidity.values)
    
    @cached_property
    def wind_speed(self) -> List[float]:
        """Return the wind speed values from the simulation."""
        return np.array(self.spatial_comfort.epw.wind_speed.values)
    
    @cached_property
    def wind_direction(self) -> List[float]:
        """Return the wind direction values from the simulation."""
        return np.array(self.spatial_comfort.epw.wind_direction.values)
    
    @cached_property
    def points(self) -> pd.DataFrame:
        """Return the points results from the simulation directory, and create the H5 file to store them as compressed objects if not already done.

        Returns:
            pd.DataFrame: A dataframe with the points locations.
        """

        points_path = self.spatial_comfort.simulation_directory / "points.h5"

        if points_path.exists():
            print(f"- Loading points data from {self.spatial_comfort.simulation_directory.name}")
            return pd.read_hdf(points_path, "df")
        
        print(f"- Processing points data for {self.spatial_comfort.simulation_directory.name}")
        points_files = list(
            (
                self.spatial_comfort.simulation_directory
                / "sky_view"
                / "model"
                / "grid"
            ).glob("*.pts")
        )
        points = load_pts(points_files).astype(np.float16)
        points.to_hdf(points_path, "df", complevel=9, complib="blosc")
        return points

    @cached_property
    def total_irradiance(self) -> pd.DataFrame:
        """Get the total irradiance from the simulation directory.

        Returns:
            pd.DataFrame: A dataframe with the total irradiance.
        """

        total_irradiance_path = (
            self.spatial_comfort.simulation_directory / "total_irradiance.h5"
        )
        
        if total_irradiance_path.exists():
            print(f"- Loading irradiance data from {self.spatial_comfort.simulation_directory.name}")
            return pd.read_hdf(total_irradiance_path, "df")
        
        print(f"- Processing irradiance data for {self.spatial_comfort.simulation_directory.name}")
        ill_files = list(
            (
                self.spatial_comfort.simulation_directory
                / "annual_irradiance"
                / "results"
                / "total"
            ).glob("*.ill")
        )
        total_irradiance = make_annual(load_ill(ill_files)).fillna(0).astype(np.float16)
        total_irradiance.to_hdf(
            total_irradiance_path, "df", complevel=9, complib="blosc"
        )
        return total_irradiance

    @cached_property
    def sky_view(self) -> pd.DataFrame:
        """Get the sky view from the simulation directory.

        Returns:
            pd.DataFrame: The sky view dataframe.
        """

        sky_view_path = self.spatial_comfort.simulation_directory / "sky_view.h5"

        if sky_view_path.exists():
            print(f"- Loading sky-view data from {self.spatial_comfort.simulation_directory.name}")
            return pd.read_hdf(sky_view_path, "df")
        
        print(f"- Processing sky-view data for {self.spatial_comfort.simulation_directory.name}")
        res_files = list(
            (
                self.spatial_comfort.simulation_directory
                / "sky_view"
                / "results"
            ).glob("*.res")
        )
        sky_view = load_res(res_files).astype(np.float16)
        sky_view.to_hdf(sky_view_path, "df", complevel=9, complib="blosc")
        return sky_view

    @staticmethod
    def _interpolate_between_unshaded_shaded(
        unshaded: HourlyContinuousCollection,
        shaded: HourlyContinuousCollection,
        total_irradiance: pd.DataFrame,
        sky_view: pd.DataFrame,
        sun_up_bool: np.ndarray(dtype=bool),
    ) -> pd.DataFrame:
        """INterpolate between the unshaded and shaded input values, using the total irradiance and sky view as proportional values for each point.

        Args:
            unshaded (HourlyContinuousCollection): A collection of hourly values for the unshaded case.
            shaded (HourlyContinuousCollection): A collection of hourly values for the shaded case.
            total_irradiance (pd.DataFrame): A dataframe with the total irradiance for each point.
            sky_view (pd.DataFrame): A dataframe with the sky view for each point.
            sun_up_bool (np.ndarray): A list if booleans stating whether the sun is up.

        Returns:
            pd.DataFrame: _description_
        """
        y_original = np.stack([shaded.values, unshaded.values], axis=1)
        new_min = y_original[sun_up_bool].min(axis=1)
        new_max = y_original[sun_up_bool].max(axis=1)

        # DAYTIME
        irradiance_grp = total_irradiance[sun_up_bool].groupby(
            total_irradiance.columns.get_level_values(0), axis=1
        )
        daytimes = []
        for grid in sky_view.columns:
            irradiance_range = np.vstack(
                [irradiance_grp.min()[grid], irradiance_grp.max()[grid]]
            ).T
            old_min = irradiance_range.min(axis=1)
            old_max = irradiance_range.max(axis=1)
            old_value = total_irradiance[grid][sun_up_bool].values
            with np.errstate(divide="ignore", invalid="ignore"):
                new_value = ((old_value.T - old_min) / (old_max - old_min)) * (
                    new_max - new_min
                ) + new_min
            daytimes.append(pd.DataFrame(new_value.T))

        daytime = pd.concat(daytimes, axis=1)
        daytime.index = total_irradiance.index[sun_up_bool]
        daytime.columns = total_irradiance.columns

        # NIGHTTIME
        x_original = [0, 100]
        nighttime = []
        for grid in sky_view.columns:
            nighttime.append(
                pd.DataFrame(
                    interp1d(x_original, y_original[~sun_up_bool])(sky_view[grid])
                ).dropna(axis=1)
            )
        nighttime = pd.concat(nighttime, axis=1)
        nighttime.index = total_irradiance.index[~sun_up_bool]
        nighttime.columns = total_irradiance.columns

        interpolated_result = (
            pd.concat([nighttime, daytime], axis=0)
            .sort_index()
            .interpolate()
            .ewm(span=1.5)
            .mean()
        )

        return interpolated_result
    
    @staticmethod
    def _angle_from_north(vector: List[float]) -> float:
        """For an X, Y vector, determine the clockwise angle to north at [0, 1].

        Args:
            vector (List[float]): A vector of length 2.

        Returns:
            float: The angle between vector and north in degrees clockwise from [0, 1].
        """        
        north = [0, 1]
        angle1 = np.arctan2(*north[::-1])
        angle2 = np.arctan2(*vector[::-1])
        return np.rad2deg((angle1 - angle2) % (2 * np.pi))

    @cached_property
    def mean_radiant_temperature_matrix(self) -> pd.DataFrame:
        """Determine the mean radiant temperature based on a shaded/unshaded ground surface, and the annual spatial incident total radiation.

        Returns:
            pd.DataFrame: A dataframe with the mean radiant temperature.
        """

        mrt_path = (
            self.spatial_comfort.simulation_directory
            / "mean_radiant_temperature.h5"
        )
                    
        if mrt_path.exists():
            print(f"- Loading mean-radiant-temperature data from {self.spatial_comfort.simulation_directory.name}")
            return pd.read_hdf(mrt_path, "df")

        print(f"- Processing mean-radiant-temperature data for {self.spatial_comfort.simulation_directory.name}")
        mrt = self._interpolate_between_unshaded_shaded(
            self.spatial_comfort.unshaded_typology_result.mean_radiant_temperature,
            self.spatial_comfort.shaded_typology_result.mean_radiant_temperature,
            self.total_irradiance,
            self.sky_view,
            np.array(self.spatial_comfort.epw.global_horizontal_radiation.values) > 0,
        ).astype(np.float16)
        mrt.to_hdf(mrt_path, "df", complevel=9, complib="blosc")
        return mrt

    @cached_property
    def moisture_matrix(self) -> pd.DataFrame:
        raise NotImplementedError()
    
    @cached_property
    def dry_bulb_temperature_matrix(self) -> pd.DataFrame:
        raise NotImplementedError()
    
    @cached_property
    def relative_humidity_matrix(self) -> pd.DataFrame:
        raise NotImplementedError()
    
    @cached_property
    def wind_speed_matrix(self) -> pd.DataFrame:
        raise NotImplementedError()
