from __future__ import annotations

import calendar
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from external_comfort.moisture import MoistureSource
from honeybee_extension.results import load_ill, load_pts, load_res, make_annual
from ladybug_comfort.utci import universal_thermal_climate_index
from ladybug.epw import AnalysisPeriod, HourlyContinuousCollection
from ladybug_extension.analysis_period import describe_analysis_period
from ladybug_extension.datacollection import to_array, to_series
from matplotlib import pyplot as plt
from external_comfort.moisture import evaporative_cooling_effect
from PIL import Image
from scipy.interpolate import interp1d

from external_comfort.encoder import Encoder
from external_comfort.external_comfort import ExternalComfort, ExternalComfortResult
from external_comfort.plot import (
    UTCI_BOUNDARYNORM,
    UTCI_COLORMAP,
    UTCI_LEVELS,
    Triangulation,
    create_triangulation,
    plot_spatial,
)
from external_comfort.shelter import Shelter
from external_comfort.typology import Typology, TypologyResult


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
        return super(SpatialEncoder, self).default(obj)


@dataclass
class SpatialComfort:
    simulation_directory: Path = field(init=True, repr=True)
    external_comfort_result: ExternalComfortResult = field(init=True, repr=True)

    unshaded: TypologyResult = field(init=False, repr=False)
    shaded: TypologyResult = field(init=False, repr=False)

    def __post_init__(self) -> SpatialComfort:
        object.__setattr__(
            self, "simulation_directory", Path(self.simulation_directory)
        )
        self._simulation_validity()

        for k, v in self._typology_result().items():
            object.__setattr__(self, k, v)

    def _simulation_validity(self) -> None:

        if (self.simulation_directory is None) or (self.simulation_directory == ""):
            raise ValueError("Simulation directory is not set.")

        annual_irradiance_directory = self.simulation_directory / "annual_irradiance"
        if not annual_irradiance_directory.exists():
            raise ValueError(
                f"Annual-irradiance data is not available at {annual_irradiance_directory}."
            )

        sky_view_directory = self.simulation_directory / "sky_view"
        if not (sky_view_directory).exists():
            raise ValueError(f"Sky-view data is not available at {sky_view_directory}.")

        water_sources_json = self.simulation_directory / "water_sources.json"
        if not (water_sources_json).exists():
            raise ValueError(
                f"Water-source data is not available at {water_sources_json}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        d = {
            "simulation_directory": self.simulation_directory,
            "shaded_ground_temperature": self.shaded.external_comfort_result.shaded_below_temperature,
            "shaded_universal_thermal_climate_index": self.shaded.universal_thermal_climate_index,
            "shaded_mean_radiant_temperature": self.shaded.mean_radiant_temperature,
            "unshaded_ground_temperature": self.unshaded.external_comfort_result.unshaded_below_temperature,
            "unshaded_universal_thermal_climate_index": self.unshaded.universal_thermal_climate_index,
            "unshaded_mean_radiant_temperature": self.unshaded.mean_radiant_temperature,
        }
        return d

    def to_json(self) -> Path:
        """Write the content of this object to a JSON file

        Returns:
            Path: The path to the newly created JSON file.
        """

        file_path: Path = self.simulation_directory / "spatial_comfort.json"
        file_path.parent.mkdir(exist_ok=True, parents=True)

        with open(file_path, "w") as fp:
            json.dump(self.to_dict(), fp, cls=SpatialEncoder, indent=4)

        return file_path

    def _typology_result(self) -> Dict[str, TypologyResult]:
        unshaded = TypologyResult(
            Typology(name="unshaded", shelters=None), self.external_comfort_result
        )

        shaded = TypologyResult(
            Typology(
                name="shaded",
                shelters=[
                    Shelter(porosity=0, altitude_range=[0, 90], azimuth_range=[0, 360])
                ],
            ),
            self.external_comfort_result,
        )
        return {"unshaded": unshaded, "shaded": shaded}


@dataclass
class SpatialComfortResult:
    spatial_comfort: SpatialComfort = field(init=True, repr=True)

    points: pd.DataFrame = field(init=False, repr=False)
    sky_view: pd.DataFrame = field(init=False, repr=False)
    total_irradiance: pd.DataFrame = field(init=False, repr=False)

    water_sources: Dict[str, Any] = field(init=False, repr=False)
    triangulation: Dict[str, Triangulation] = field(init=False, repr=False)

    def __post_init__(self) -> SpatialComfortResult:

        # Write input spatial metrics to JSON
        self.spatial_comfort.to_json()

        object.__setattr__(self, "points", self._points())
        object.__setattr__(self, "sky_view", self._sky_view())
        object.__setattr__(self, "total_irradiance", self._total_irradiance())

        object.__setattr__(
            self, "ground_surface_temperature", self._ground_surface_temperature()
        )
        object.__setattr__(
            self, "mean_radiant_temperature", self._mean_radiant_temperature()
        )
        object.__setattr__(
            self,
            "universal_thermal_climate_index_simple",
            self._universal_thermal_climate_index_simple(),
        )
        try:
            object.__setattr__(self, "moisture_sources", self._water_sources())
            object.__setattr__(
                self,
                "universal_thermal_climate_index_complex",
                self._universal_thermal_climate_index_complex(),
            )
        except Exception as e:
            print(f"Complex UTCI calcualtion not possible because {e}")
        
        object.__setattr__(self, "triangulation", self._triangulation())

        # self._generic_output()

    def _points(self) -> pd.DataFrame:
        """Return the points results from the simulation directory, and create the H5 file to store them as compressed objects if not already done.

        Returns:
            pd.DataFrame: A dataframe with the points locations.
        """

        points_path = self.spatial_comfort.simulation_directory / "points.h5"

        try:
            return self.points
        except AttributeError:
            if points_path.exists():
                print(f"- Loading points data from {self.spatial_comfort.simulation_directory.name}")
                return pd.read_hdf(points_path, "df")
            else:
                print(f"- Processing points data for {self.spatial_comfort.simulation_directory.name}")
                points_files = list(
                    (
                        self.spatial_comfort.simulation_directory
                        / "sky_view"
                        / "model"
                        / "grid"
                    ).glob("*.pts")
                )
                points = load_pts(points_files)
                points.to_hdf(points_path, "df", complevel=9, complib="blosc")
            return points

    def _total_irradiance(self) -> pd.DataFrame:
        """Get the total irradiance from the simulation directory.

        Returns:
            pd.DataFrame: A dataframe with the total irradiance.
        """
        try:
            return self.total_irradiance
        except AttributeError:
            total_irradiance_path = (
                self.spatial_comfort.simulation_directory / "total_irradiance.h5"
            )

            if total_irradiance_path.exists():
                print(f"- Loading irradiance data from {self.spatial_comfort.simulation_directory.name}")
                return pd.read_hdf(total_irradiance_path, "df")
            else:
                print(f"- Processing irradiance data for {self.spatial_comfort.simulation_directory.name}")
                ill_files = list(
                    (
                        self.spatial_comfort.simulation_directory
                        / "annual_irradiance"
                        / "results"
                        / "total"
                    ).glob("*.ill")
                )
                total_irradiance = make_annual(load_ill(ill_files)).fillna(0)
                total_irradiance.to_hdf(
                    total_irradiance_path, "df", complevel=9, complib="blosc"
                )
            return total_irradiance

    def _sky_view(self) -> pd.DataFrame:
        """Get the sky view from the simulation directory.

        Returns:
            pd.DataFrame: The sky view dataframe.
        """

        try:
            return self.sky_view
        except AttributeError:
            sky_view_path = self.spatial_comfort.simulation_directory / "sky_view.h5"

            if sky_view_path.exists():
                print(f"- Loading sky-view data from {self.spatial_comfort.simulation_directory.name}")
                return pd.read_hdf(sky_view_path, "df")
            else:
                print(f"- Processing sky-view data for {self.spatial_comfort.simulation_directory.name}")
                res_files = list(
                    (
                        self.spatial_comfort.simulation_directory
                        / "sky_view"
                        / "results"
                    ).glob("*.res")
                )
                sky_view = load_res(res_files)
                sky_view.to_hdf(sky_view_path, "df", complevel=9, complib="blosc")
            return sky_view

    def _water_sources(self) -> List[MoistureSource]:
        """Load the water bodies from the simulation directory.

        Returns:
            Dict[str, Any]: A dictionary contining the boundary vertices of any water bodies, and locations of any point-sources.
        """

        water_sources_path = (
            self.spatial_comfort.simulation_directory / "water_sources.json"
        )

        return MoistureSource.from_json(water_sources_path)

    def _dry_bulb_temperature_no_moisture(self) -> pd.Series:
        """Get the dry bulb temperature from the input EPW file.

        Returns:
            pd.Series: A series with the dry bulb temperature directly from the EPW input file.
        """
        return to_series(
            self.spatial_comfort.external_comfort_result.external_comfort.epw.dry_bulb_temperature
        )

    def _relative_humidity_no_moisture(self) -> pd.Series:
        """Get the relative humidity from the input EPW file.

        Returns:
            pd.Series: A series with the dry bulb temperature directly from the EPW input file.
        """
        return to_series(
            self.spatial_comfort.external_comfort_result.external_comfort.epw.relative_humidity
        )

    def _wind_speed(self) -> pd.Series:
        """Get the wind_speed from the input EPW file.

        Returns:
            pd.Series: A series with the wind-speed directly from the EPW input file.
        """
        return to_series(
            self.spatial_comfort.external_comfort_result.external_comfort.epw.wind_speed
        )

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

    def _mean_radiant_temperature(self) -> pd.DataFrame:
        """Determine the mean radiant temperature based on a shaded/unshed ground surface, and the annual spatial incident total radiation.

        Returns:
            pd.DataFrame: A dataframe with the mean radiant temperature.
        """
        try:
            return self.mean_radiant_temperature
        except AttributeError:
            mrt_path = (
                self.spatial_comfort.simulation_directory
                / "mean_radiant_temperature.h5"
            )

            if mrt_path.exists():
                print(
                    f"- Loading mean-radiant-temperature data from {self.spatial_comfort.simulation_directory.name}"
                )
                return pd.read_hdf(mrt_path, "df")

            print(f"- Processing mean-radiant-temperature data for {self.spatial_comfort.simulation_directory.name}")

            mrt = self._interpolate_between_unshaded_shaded(
                self.spatial_comfort.unshaded.mean_radiant_temperature,
                self.spatial_comfort.shaded.mean_radiant_temperature,
                self.total_irradiance,
                self.sky_view,
                to_array(
                    self.spatial_comfort.external_comfort_result.external_comfort.epw.global_horizontal_radiation
                )
                > 0,
            )
            mrt.to_hdf(mrt_path, "df", complevel=9, complib="blosc")
            return mrt

    def _ground_surface_temperature(self) -> pd.DataFrame:
        """Determine the ground surface temperature based on a shaded/unshed ground surface, and the annual spatial incident total radiation.

        Returns:
            pd.DataFrame: A dataframe with the ground surface temperature.
        """
        try:
            return self.ground_surface_temperature
        except AttributeError:
            gnd_srf_path = (
                self.spatial_comfort.simulation_directory
                / "ground_surface_temperature.h5"
            )

            if gnd_srf_path.exists():
                print(
                    f"- Loading ground_surface_temperature data from {self.spatial_comfort.simulation_directory.name}"
                )
                return pd.read_hdf(gnd_srf_path, "df")

            print(f"- Processing ground_surface_temperature data for {self.spatial_comfort.simulation_directory.name}")

            gnd_srf = self._interpolate_between_unshaded_shaded(
                self.spatial_comfort.shaded.ground_surface_temperature,
                self.spatial_comfort.unshaded.ground_surface_temperature,
                self.total_irradiance,
                self.sky_view,
                to_array(
                    self.spatial_comfort.external_comfort_result.external_comfort.epw.global_horizontal_radiation
                )
                > 0,
            )
            gnd_srf.to_hdf(gnd_srf_path, "df", complevel=9, complib="blosc")
            return gnd_srf

    def _universal_thermal_climate_index_simple(self) -> pd.DataFrame:
        """Using the simplified method (not accounting for wind direction, speed or moisture addition), calculate the UTCI.

        Returns:
            pd.DataFrame: A dataframe with the UTCI for each hour of the year, for each point.
        """
        try:
            return self.universal_thermal_climate_index_simple
        except AttributeError:
            utci_path = (
                self.spatial_comfort.simulation_directory
                / "universal_thermal_climate_index_simple.h5"
            )

            if utci_path.exists():
                print(
                    f"- Loading universal thermal climate index data from {self.spatial_comfort.simulation_directory.name}"
                )
                return pd.read_hdf(utci_path, "df")

            print(f"- Processing universal thermal climate index data for {self.spatial_comfort.simulation_directory.name}")

            utci = self._interpolate_between_unshaded_shaded(
                self.spatial_comfort.unshaded.universal_thermal_climate_index,
                self.spatial_comfort.shaded.universal_thermal_climate_index,
                self.total_irradiance,
                self.sky_view,
                to_array(
                    self.spatial_comfort.external_comfort_result.external_comfort.epw.global_horizontal_radiation
                )
                > 0,
            )
            utci.to_hdf(utci_path, "df", complevel=9, complib="blosc")
            return utci

    def _spatial_moisture_matrix(self) -> pd.DataFrame:

        save_path = self.spatial_comfort.simulation_directory / "moisture_matrix.h5"
        if save_path.exists():
            print(f"- Loading moisture matrix data from {self.spatial_comfort.simulation_directory.name}")
            return pd.read_hdf(save_path, "df")

        # Create lookup for unique wind-speed/direction combinations for year
        ws = to_series(self.spatial_comfort.external_comfort_result.external_comfort.epw.wind_speed)
        wd = to_series(self.spatial_comfort.external_comfort_result.external_comfort.epw.wind_direction)
        wind_speed_direction = pd.concat([ws, wd], axis=1).values
        wind_speed_direction_unique = np.unique(wind_speed_direction, axis=0)

        moisture_sources = self._water_sources()

        all_points_df = self.points.droplevel([0], axis=1)
        all_pts = all_points_df[["x", "y"]].values

        with np.errstate(divide='ignore', invalid='ignore'):
            angle_matrices = []
            distance_matrices = []
            for n_ms, moisture_source in enumerate(moisture_sources[0:]):
                src_pts = all_pts[moisture_source.point_indices][0:]  # <- Limit this value to reduce points to test - useful for debugging

                # create distance and angle matrices
                normal_vector = np.array([0, -1])
                vector_matrix = np.array([all_pts - sp for sp in src_pts])
                distance_matrix = np.linalg.norm(vector_matrix, axis=2)
                vector_matrix_unit = np.moveaxis([i.T / distance_matrix for i in vector_matrix.T], 0, -1)
                angle_matrix_temp = np.nan_to_num(np.degrees(np.arccos(np.clip(np.dot(normal_vector, np.swapaxes(vector_matrix_unit, -1, -2)), -1.0, 1.0))))
                angle_matrix = np.where(vector_matrix[:, :, 0] > 0, 360 - angle_matrix_temp, angle_matrix_temp)
                distance_matrices.append(distance_matrix.astype(np.float16, copy=False))
                angle_matrices.append(angle_matrix.astype(np.float16, copy=False))

            angle_matrices = np.array(angle_matrices, dtype=object)
            distance_matrices = np.array(distance_matrices, dtype=object)

            # Construct a table containing all possible combinations of speed/direction to then pick from for the annual matrix
            lookup = []
            for n_ws_wd, (wind_speed, wind_direction) in enumerate(wind_speed_direction_unique[0:]):
                print(f" - {n_ws_wd/len(wind_speed_direction_unique):0.3%}", end="\r")
                # Get distance mask - describing where each point is within a certain distance of the moisture source point, based on wind speed
                distance_mask = [dm < wind_speed * 10 for dm in distance_matrices]

                # Get angle mask - describing where each point is "downwind" from each moisture source point
                plume_buffer = 5  # the angle (in degrees) either side of the downwind direction within which to capture downwind points
                angle_mask = []
                for am in angle_matrices:
                    if wind_direction > (360 - plume_buffer):
                        angle_mask.append(np.any([
                            (am > wind_direction - plume_buffer),
                            (am < plume_buffer - 360 - wind_direction),
                        ], axis=0))
                    elif wind_direction < (0 + plume_buffer):
                        angle_mask.append(np.any([
                            (am < wind_direction + plume_buffer), 
                            (am > 360 + wind_direction - plume_buffer)
                        ], axis=0))
                    else:
                        angle_mask.append(np.all([
                            (am < wind_direction + plume_buffer), 
                            (am > wind_direction - plume_buffer)
                        ], axis=0))

                # Get combined mask for impacted points
                mask = [np.all([am, dm], axis=0) for am, dm in zip(*[angle_mask, distance_mask])]

                moisture_matrices = []
                for n_ms, moisture_source in enumerate(moisture_sources[0:]):
                    moisture_matrices.append(
                        np.amax(
                            np.where(mask[n_ms], np.clip(moisture_source.magnitude / distance_matrices[n_ms], 0, moisture_source.magnitude), 0),
                            axis=0
                        )
                    )
                lookup.append(np.amax(moisture_matrices, axis=0))

            # Construct annual matrix
            idx = [np.all(wind_speed_direction_unique == ws_wd, axis=1).argmax() for ws_wd in wind_speed_direction]
            annual_moisture_matrix = np.array(lookup)[idx]

            # Ensure hours where wind speed == 0 use the raw moisture index values
            # This part is a hack to get things working properly! It can be made much better/more efficient
            mags = []
            for ms in moisture_sources:
                moisture_pt_bool = np.isin(range(len(all_pts)), ms.point_indices)
                mags.append(np.where(moisture_pt_bool, ms.magnitude, 0))
            val_at_zero = np.array(mags).max(axis=0)

            new_matrix = []
            for hr in range(len(annual_moisture_matrix)):
                if ws[hr] == 0:
                    new_matrix.append(val_at_zero)
                else:
                    new_matrix.append(annual_moisture_matrix[hr])
        
        df = pd.DataFrame(np.array(new_matrix), index=ws.index).round(3)

        df.to_hdf(save_path, "df", complevel=9, complib="blosc")

        return df

    def _dbt_rh_ws_matrices(self) -> List[pd.DataFrame]:

        dbt_save_path = self.spatial_comfort.simulation_directory / "dry_bulb_temperature_matrix.h5"
        rh_save_path = self.spatial_comfort.simulation_directory / "relative_humidity_matrix.h5"
        ws_save_path = self.spatial_comfort.simulation_directory / "wind_speed_matrix.h5"
        if dbt_save_path.exists() & rh_save_path.exists() & ws_save_path.exists():
            print(f"- Matrix data already exists, and will be loaded instead of calculated")

            return [pd.read_hdf(dbt_save_path, "df"), pd.read_hdf(rh_save_path, "df"), pd.read_hdf(ws_save_path, "df")]

        moisture_matrix = self._spatial_moisture_matrix()
        temp = []
        for n, (dbt, rh, atm) in enumerate(list(zip(*[self.spatial_comfort.external_comfort_result.external_comfort.epw.dry_bulb_temperature, self.spatial_comfort.external_comfort_result.external_comfort.epw.relative_humidity, self.spatial_comfort.external_comfort_result.external_comfort.epw.atmospheric_station_pressure]))[0:]):
            print(f"- Calculating DBT/RH hourly spatial values - {n/len(moisture_matrix):0.2%}", end="\r")
            temp.append(evaporative_cooling_effect(dbt, rh, moisture_matrix.values[n], atm))
        dbt_mtx, rh_mtx = np.swapaxes(temp, 0, 1)

        dbt = pd.DataFrame(dbt_mtx, index=moisture_matrix.index)
        rh = pd.DataFrame(rh_mtx, index=moisture_matrix.index)

        dbt.to_hdf(dbt_save_path, "df", complib="blosc", complevel=9)
        rh.to_hdf(rh_save_path, "df", complib="blosc", complevel=9)

        ws = to_series(self.spatial_comfort.external_comfort_result.external_comfort.epw.wind_speed)
        ws = pd.DataFrame(np.repeat([ws.values.astype(np.float16)], [len(moisture_matrix.values.T)], axis=0).T, index=moisture_matrix.index)
        ws.to_hdf(ws_save_path, "df", complib="blosc", complevel=9)

        return [dbt, rh, ws]

    def _universal_thermal_climate_index_complex(self) -> pd.DataFrame:

        save_path = self.spatial_comfort.simulation_directory / "universal_thermal_climate_index_complex.h5"
        if save_path.exists():
            print(f"- Loading UTCI from {save_path}")
            return pd.read_hdf(save_path, "df")
       
        uu = np.vectorize(universal_thermal_climate_index)

        dbt, rh, ws = self._dbt_rh_ws_matrices()
        mrt = self._mean_radiant_temperature()

        # TODO - MAKE THIS MULTITHREADED AS IT WOULD BE SOOOOO MUCH FASTER!!!!!!!
        utcis = []
        for n in range(len(dbt)):
            print(f"{n/len(dbt):0.2%}", end="\r")
            utcis.append(uu(dbt.values[n], mrt.values[n], ws.values[n], rh.values[n]))
        
        utci = pd.DataFrame(np.array(utcis), index=dbt.index)

        utci.to_hdf(save_path, "df", complib="blosc", complevel=9)

        return utci

    def _x_lims(self) -> List[float]:
        """Return the x-axis limits for the plot."""
        maxima = (
            self.points.groupby(self.points.columns.get_level_values(1), axis=1)
            .max()
            .max()
        )
        minima = (
            self.points.groupby(self.points.columns.get_level_values(1), axis=1)
            .min()
            .min()
        )
        return [minima.x, maxima.x]

    def _y_lims(self) -> List[float]:
        """Return the y-axis limits for the plot."""
        maxima = (
            self.points.groupby(self.points.columns.get_level_values(1), axis=1)
            .max()
            .max()
        )
        minima = (
            self.points.groupby(self.points.columns.get_level_values(1), axis=1)
            .min()
            .min()
        )
        return [minima.y, maxima.y]

    def _gnd_temp_lims(self) -> List[float]:
        """Return the ground surface temperature value limits for the plot."""
        maxima = self.ground_surface_temperature.unstack().max()
        minima = self.ground_surface_temperature.unstack().min()
        return [minima, maxima]

    def _mrt_lims(self) -> List[float]:
        """Return the MRT value limits for the plot."""
        maxima = self.mean_radiant_temperature.unstack().max()
        minima = self.mean_radiant_temperature.unstack().min()
        return [minima, maxima]

    def _utci_lims(self) -> List[float]:
        """Return the UTCI value limits for the plot."""
        maxima = self.universal_thermal_climate_index_simple.unstack().max()
        minima = self.universal_thermal_climate_index_simple.unstack().min()
        return [minima, maxima]

    def _grid_names(self) -> List[str]:
        """Return the names of the grids for this spatial comfort result."""
        return self.points.columns.get_level_values(0).unique().tolist()

    def _triangulation(self) -> Dict[str, Triangulation]:
        """Triangulate the x, y sensor locations for this spatial case.

        Returns:
            Dict[str, Triangulation]: A dictionary with grid names as keys and triangulations as values.
        """
        d = {}
        for grid_name in self._grid_names():
            xs = self.points.loc[:, (grid_name, "x")].dropna().values
            ys = self.points.loc[:, (grid_name, "y")].dropna().values
            d[grid_name] = create_triangulation(xs, ys)
        return d

    def _plot_sky_view(self) -> Path:
        """Return the path to the sky view plot."""

        min_val = 0
        max_val = 100
        step = 5
        levels = np.arange(min_val, max_val + step, step)

        zs = [
            self._sky_view()[grid_name].dropna().values
            for grid_name in self._grid_names()
        ]

        fig = plot_spatial(
            triangulations=self.triangulation.values(),
            values=zs,
            levels=levels,
            colormap="Spectral_r",
            extend="neither",
            xlims=self._x_lims(),
            ylims=self._y_lims(),
            colorbar_label="Proportion of sky visible (%)",
            title=f"Proportion of sky visible",
        )

        save_path = self.spatial_comfort.simulation_directory / "sky_view.png"

        fig.savefig(save_path, dpi=200, bbox_inches="tight")

        return save_path

    def _plot_utci(self, month: int, hour: int) -> Path:
        """Return the path to the UTCI plot."""

        if not month in range(1, 13, 1):
            raise ValueError(f"Month must be between 1 and 12 inclusive, got {month}")
        if not hour in range(0, 24, 1):
            raise ValueError(f"Hour must be between 0 and 23 inclusive, got {hour}")

        # Group data to give a typical month-hour value
        zs = (
            self._universal_thermal_climate_index_simple()
            .groupby(
                [
                    self._universal_thermal_climate_index_simple().index.month,
                    self._universal_thermal_climate_index_simple().index.hour,
                ],
                axis=0,
            )
            .mean()
            .loc[month, hour]
        )
        zs = [zs.unstack().T[gn].dropna().values.tolist() for gn in self._grid_names()]

        fig = plot_spatial(
            triangulations=self.triangulation.values(),
            values=zs,
            levels=UTCI_LEVELS,
            colormap=UTCI_COLORMAP,
            extend="both",
            norm=UTCI_BOUNDARYNORM,
            xlims=self._x_lims(),
            ylims=self._y_lims(),
            colorbar_label="Universal thermal climate index (°C)",
            title=f"{calendar.month_abbr[month]} {hour:02d}:00",
        )

        save_path = (
            self.spatial_comfort.simulation_directory
            / f"universal_thermal_climate_index_{month:02d}_{hour:02d}.png"
        )

        fig.savefig(save_path, dpi=200, bbox_inches="tight")

        return save_path

    def _plot_mrt(self, month: int, hour: int) -> Path:
        """Return the path to the MRT plot."""

        if not month in range(1, 13, 1):
            raise ValueError(f"Month must be between 1 and 12 inclusive, got {month}")
        if not hour in range(0, 24, 1):
            raise ValueError(f"Hour must be between 0 and 23 inclusive, got {hour}")

        min_val, max_val = self._mrt_lims()
        levels = np.linspace(np.floor(min_val), np.ceil(max_val), 21)

        # Group data to give a typical month-hour value
        zs = (
            self._mean_radiant_temperature()
            .groupby(
                [
                    self._mean_radiant_temperature().index.month,
                    self._mean_radiant_temperature().index.hour,
                ],
                axis=0,
            )
            .mean()
            .loc[month, hour]
        )
        zs = [zs.unstack().T[gn].dropna().values.tolist() for gn in self._grid_names()]

        fig = plot_spatial(
            triangulations=self.triangulation.values(),
            values=zs,
            levels=levels,
            colormap="inferno",
            extend="both",
            xlims=self._x_lims(),
            ylims=self._y_lims(),
            colorbar_label="Mean radiant temperature (°C)",
            title=f"{calendar.month_abbr[month]} {hour:02d}:00",
        )

        save_path = (
            self.spatial_comfort.simulation_directory
            / f"mean_radiant_temperature_{month:02d}_{hour:02d}.png"
        )

        fig.savefig(save_path, dpi=200, bbox_inches="tight")

        return save_path

    def _plot_utci_comfortable_hours(self, analysis_period: AnalysisPeriod) -> Path:
        """Return the path to the comfortable-hours plot."""

        # Filter for the analysis period
        zs_temp = self._universal_thermal_climate_index_simple().iloc[
            list(analysis_period.hoys_int), :
        ]

        comfortable = ((zs_temp >= 9) & (zs_temp <= 26)).sum()

        # Calculate % hours comfortable
        zs = (
            comfortable
            / len(analysis_period.hoys_int)
            * 100
        )
        zs = [zs.unstack().T[gn].dropna().values.tolist() for gn in self._grid_names()]

        fig = plot_spatial(
            triangulations=self.triangulation.values(),
            values=zs,
            levels=21,
            colormap="magma_r",
            extend="both",
            xlims=self._x_lims(),
            ylims=self._y_lims(),
            colorbar_label=f"% time comfortable (out of {len(analysis_period.hoys_int)} hours)",
            title=f"Time comfortable (9°C-26°C UTCI) for {describe_analysis_period(analysis_period, False)}",
        )

        save_path = (
            self.spatial_comfort.simulation_directory
            / f"time_comfortable_{describe_analysis_period(analysis_period, True)}.png"
        )

        fig.savefig(save_path, dpi=200, bbox_inches="tight")

        return save_path

    def _generic_output(self) -> List[Path]:
        """Return a list of paths to the generic outputs."""

        output = []

        # Sky view plot
        print("- Plotting sky view...")
        output.append(self._plot_sky_view())
        plt.close("all")

        # UTCI comfortable hours
        print("- Plotting UTCI comfortable hours (annual)...")
        output.append(self._plot_utci_comfortable_hours(AnalysisPeriod()))
        plt.close("all")
        print("- Plotting UTCI comfortable hours (08:00-18:00)...")
        output.append(
            self._plot_utci_comfortable_hours(AnalysisPeriod(st_hour=8, end_hour=18))
        )
        plt.close("all")
        for month in range(1, 13, 1):
            print(
                f"- Plotting UTCI comfortable hours (08:00-18:00) for {calendar.month_abbr[month]}..."
            )
            output.append(
                self._plot_utci_comfortable_hours(
                    AnalysisPeriod(st_month=month, end_month=month)
                )
            )
            plt.close("all")

        # UTCI typical days animation
        for month in [12, 3, 6]:
            print(f"- Plotting spatial UTCI for {calendar.month_abbr[month]}...")
            temp_output = []
            for hour in range(24):
                temp_output.append(self._plot_utci(month, hour))
                plt.close("all")
            
            # images = [Image.open(to) for to in temp_output]
            # print(f"- Creating animated spatial UTCI for {calendar.month_abbr[month]}...")
            # images[0].save(
            #     self.spatial_comfort.simulation_directory
            #     / f"universal_thermal_climate_index_{month:02d}.gif",
            #     save_all=True,
            #     append_images=images[1:],
            #     optimize=True,
            #     duration=333,
            #     loop=0,
            # )

        return output
