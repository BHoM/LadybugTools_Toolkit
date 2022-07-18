# from __future__ import annotations

# import calendar
# import shutil
# import warnings
# from dataclasses import dataclass
# from datetime import datetime
# from pathlib import Path
# from typing import Dict, List

# import numpy as np
# import pandas as pd
# from cached_property import cached_property
# from honeybee.model import Model
# from ladybug.analysisperiod import AnalysisPeriod
# from ladybug.datacollection import HourlyContinuousCollection
# from ladybug_comfort.utci import universal_thermal_climate_index
# from matplotlib import pyplot as plt
# from matplotlib.figure import Figure
# from matplotlib.tri.triangulation import Triangulation
# from PIL import Image
# from python_toolkit.plot.chart import spatial_heatmap, timeseries_heatmap
# from python_toolkit.plot.color import colormap_sequential
# from python_toolkit.plot.triangulate import create_triangulation
# from scipy.interpolate import interp1d

# from ...honeybee_extension.results import load_ill, load_pts, load_res, make_annual
# from ...ladybug_extension.analysis_period import describe as describe_analysis_period
# from ...ladybug_extension.datacollection import from_series, to_series
# from ..external_comfort import ExternalComfort
# from ..moisture import Moisture, evaporative_cooling_effect
# from ..plot import plot_utci_heatmap_histogram, utci_comparison_diurnal
# from ..plot.colormaps import UTCI_BOUNDARYNORM, UTCI_COLORMAP, UTCI_LEVELS
# from ..shelter import Shelter
# from ..typology import Typology, TypologyResult

# v_utci = np.vectorize(universal_thermal_climate_index)


# @dataclass
# class SpatialComfort:
#     def __init__(
#         self, simulation_directory: Path, external_comfort_result: ExternalComfort
#     ) -> SpatialComfort:
#         """The first stage in the creation of a SpatialComfortResult, where inputs are given to assess whether calculation of a results object is possible.

#         Args:
#             simulation_directory (Path): A directory path containing SkyView and Annual Irradiance simulation results.
#             external_comfort_result (ExternalComfortResult): A results object containing pre-simulated MRT values.

#         Returns:
#             SpatialComfort: A SpatialComfort object.
#         """
#         self.simulation_directory = Path(simulation_directory)

#         # Tidy results folder and check for requisite datasets
#         self._simulation_validity()
#         self._remove_temp_files()

#         self.external_comfort_result = external_comfort_result
#         self.unshaded_typology_result = TypologyResult(
#             Typology(name="unshaded", shelters=None), self.external_comfort_result
#         )
#         self.shaded_typology_result = TypologyResult(
#             Typology(
#                 name="shaded",
#                 shelters=[
#                     Shelter(porosity=0, altitude_range=[0, 90], azimuth_range=[0, 360])
#                 ],
#             ),
#             self.external_comfort_result,
#         )

#         self.moisture_sources = self._load_moisture_sources()
#         self.epw = self.external_comfort_result.external_comfort.epw

#     def _simulation_validity(self) -> None:

#         annual_irradiance_directory = self.simulation_directory / "annual_irradiance"
#         if (
#             not annual_irradiance_directory.exists()
#             or len(list((annual_irradiance_directory / "results").glob("**/*.ill")))
#             == 0
#         ):
#             raise ValueError(
#                 f"Annual-irradiance data is not available in {annual_irradiance_directory}."
#             )

#         sky_view_directory = self.simulation_directory / "sky_view"
#         if (
#             not (sky_view_directory).exists()
#             or len(list((sky_view_directory / "results").glob("**/*.res"))) == 0
#         ):
#             raise ValueError(f"Sky-view data is not available in {sky_view_directory}.")

#         res_files = list((sky_view_directory / "results").glob("*.res"))
#         if len(res_files) != 1:
#             raise ValueError(
#                 f"This process is only possible for a single Analysis Grid - multiple files found {[i.stem for i in res_files]}."
#             )

#     def _remove_temp_files(self) -> None:
#         """Remove initial results files from simulated case to save on disk space."""
#         directories = list(self.simulation_directory.glob("**"))
#         for directory in directories:
#             if directory.name == "initial_results":
#                 shutil.rmtree(directory)

#     def _load_moisture_sources(self) -> List[Moisture]:
#         """Load the water bodies from the simulation directory.

#         Returns:
#             Dict[str, Any]: A dictionary contining the boundary vertices of any water bodies, and locations of any point-sources.
#         """

#         moisture_sources_path = self.simulation_directory / "moisture_sources.json"

#         try:
#             moisture_sources = Moisture.from_json(moisture_sources_path)

#             if len(moisture_sources) == 0:
#                 raise ValueError(
#                     f"No moisture sources found in {moisture_sources_path}"
#                 )

#             return moisture_sources
#         except FileNotFoundError:
#             warnings.warn(
#                 "No moisture_sources.json found in simulation directory - advanced moisture-impacted UTCI not possible."
#             )
#             return []

#     @cached_property
#     def model(self) -> Model:
#         """Load the honeybee model from the simulation directory."""
#         print(
#             f"Loading the model for {self.simulation_directory.stem}, this can take a while ..."
#         )
#         return Model.from_hbjson(
#             self.simulation_directory / f"{self.simulation_directory.stem}.hbjson"
#         )


# class SpatialComfortResult:
#     def __init__(self, spatial_comfort: SpatialComfort) -> SpatialComfortResult:
#         self.spatial_comfort = spatial_comfort

#     @cached_property
#     def dry_bulb_temperature(self) -> List[float]:
#         """Return the hourly dry bulb temperature values from the simulation weatherfile."""
#         return np.array(self.spatial_comfort.epw.dry_bulb_temperature.values)

#     @cached_property
#     def atmospheric_station_pressure(self) -> List[float]:
#         """Return the hourly atmospheric station pressure values from the simulation weatherfile."""
#         return np.array(self.spatial_comfort.epw.atmospheric_station_pressure.values)

#     @cached_property
#     def relative_humidity(self) -> List[float]:
#         """Return the hourly relative humidity values from the simulation weatherfile."""
#         return np.array(self.spatial_comfort.epw.relative_humidity.values)

#     @cached_property
#     def wind_speed(self) -> List[float]:
#         """Return the hourly wind speed values from the simulation weatherfile."""
#         return np.array(self.spatial_comfort.epw.wind_speed.values)

#     @cached_property
#     def wind_direction(self) -> List[float]:
#         """Return the hourly wind direction values from the simulation weatherfile."""
#         return np.array(self.spatial_comfort.epw.wind_direction.values)

#     @cached_property
#     def points(self) -> pd.DataFrame:
#         """Return the points results from the simulation directory, and create the H5 file to store them as compressed objects if not already done.

#         Returns:
#             pd.DataFrame: A dataframe with the points locations.
#         """

#         points_path = self.spatial_comfort.simulation_directory / "points.h5"

#         if points_path.exists():
#             print(
#                 f"- Loading points data from {self.spatial_comfort.simulation_directory.name}"
#             )
#             return pd.read_hdf(points_path, "df")

#         print(
#             f"- Processing points data for {self.spatial_comfort.simulation_directory.name}"
#         )
#         points_files = list(
#             (
#                 self.spatial_comfort.simulation_directory
#                 / "sky_view"
#                 / "model"
#                 / "grid"
#             ).glob("*.pts")
#         )
#         points = load_pts(points_files).astype(np.float16)
#         points.to_hdf(points_path, "df", complevel=9, complib="blosc")
#         return points

#     @cached_property
#     def total_irradiance_matrix(self) -> pd.DataFrame:
#         """Get the total irradiance from the simulation directory.

#         Returns:
#             pd.DataFrame: A dataframe with the total irradiance.
#         """

#         total_irradiance_path = (
#             self.spatial_comfort.simulation_directory / "total_irradiance_matrix.h5"
#         )

#         if total_irradiance_path.exists():
#             print(
#                 f"- Loading irradiance data from {self.spatial_comfort.simulation_directory.name}"
#             )
#             return pd.read_hdf(total_irradiance_path, "df")

#         print(
#             f"- Processing irradiance data for {self.spatial_comfort.simulation_directory.name}"
#         )
#         ill_files = list(
#             (
#                 self.spatial_comfort.simulation_directory
#                 / "annual_irradiance"
#                 / "results"
#                 / "total"
#             ).glob("*.ill")
#         )
#         total_irradiance = make_annual(load_ill(ill_files)).fillna(0).astype(np.float16)
#         # The "clip" below ensure the value range is between
#         total_irradiance.clip(lower=0).to_hdf(
#             total_irradiance_path, "df", complevel=9, complib="blosc"
#         )
#         return total_irradiance

#     @cached_property
#     def sky_view(self) -> pd.DataFrame:
#         """Get the sky view from the simulation directory.

#         Returns:
#             pd.DataFrame: The sky view dataframe.
#         """

#         sky_view_path = self.spatial_comfort.simulation_directory / "sky_view.h5"

#         if sky_view_path.exists():
#             print(
#                 f"- Loading sky-view data from {self.spatial_comfort.simulation_directory.name}"
#             )
#             return pd.read_hdf(sky_view_path, "df")

#         print(
#             f"- Processing sky-view data for {self.spatial_comfort.simulation_directory.name}"
#         )
#         res_files = list(
#             (self.spatial_comfort.simulation_directory / "sky_view" / "results").glob(
#                 "*.res"
#             )
#         )
#         sky_view = load_res(res_files).clip(lower=0, upper=100).astype(np.float16)
#         sky_view.to_hdf(sky_view_path, "df", complevel=9, complib="blosc")
#         return sky_view

#     @staticmethod
#     def _distribute(
#         magnitude: float,
#         distance: float,
#         max_distance: float,
#         curve_type: str = "linear",
#     ) -> float:
#         """Calculate the "decayed" value distributed from 0 to a given distance up to the maximum distance.

#         Args:
#             magnitude (float): The value to be distributed.
#             distance (float): A distance at which to return the magnitude.
#             max_distance (float): The maximum distance to which magnitude is to be distributed.
#             curve_type (str, optional): A type of distribution (the shape of the distribution profile). Defaults to "linear".

#         Returns:
#             float: The magnitude at the given distance.
#         """
#         distance = np.interp(distance, [0, max_distance], [0, 1])

#         if curve_type == "linear":
#             return (1 - distance) * magnitude
#         elif curve_type == "parabolic":
#             return (-(distance**2) + 1) * magnitude
#         elif curve_type == "sigmoid":
#             return (1 - (0.5 * (np.sin(distance * np.pi - np.pi / 2) + 1))) * magnitude
#         else:
#             raise ValueError(f"Unknown curve type: {curve_type}")

#     @staticmethod
#     def _interpolate_between_unshaded_shaded(
#         unshaded: HourlyContinuousCollection,
#         shaded: HourlyContinuousCollection,
#         total_irradiance: pd.DataFrame,
#         sky_view: pd.DataFrame,
#         sun_up_bool: List[List[bool]],
#     ) -> pd.DataFrame:
#         """Interpolate between the unshaded and shaded input values, using the total irradiance and sky view as proportional values for each point.

#         Args:
#             unshaded (HourlyContinuousCollection): A collection of hourly values for the unshaded case.
#             shaded (HourlyContinuousCollection): A collection of hourly values for the shaded case.
#             total_irradiance (pd.DataFrame): A dataframe with the total irradiance for each point for each hour.
#             sky_view (pd.DataFrame): A dataframe with the sky view for each point.
#             sun_up_bool (np.ndarray): A list if booleans stating whether the sun is up.

#         Returns:
#             pd.DataFrame: _description_
#         """
#         y_original = np.stack([shaded.values, unshaded.values], axis=1)
#         new_min = y_original[sun_up_bool].min(axis=1)
#         new_max = y_original[sun_up_bool].max(axis=1)

#         # DAYTIME
#         irradiance_grp = total_irradiance[sun_up_bool].groupby(
#             total_irradiance.columns.get_level_values(0), axis=1
#         )
#         daytimes = []
#         for grid in sky_view.columns:
#             irradiance_range = np.vstack(
#                 [irradiance_grp.min()[grid], irradiance_grp.max()[grid]]
#             ).T
#             old_min = irradiance_range.min(axis=1)
#             old_max = irradiance_range.max(axis=1)
#             old_value = total_irradiance[grid][sun_up_bool].values
#             with np.errstate(divide="ignore", invalid="ignore"):
#                 new_value = ((old_value.T - old_min) / (old_max - old_min)) * (
#                     new_max - new_min
#                 ) + new_min
#             daytimes.append(pd.DataFrame(new_value.T))

#         daytime = pd.concat(daytimes, axis=1)
#         daytime.index = total_irradiance.index[sun_up_bool]
#         daytime.columns = total_irradiance.columns

#         # NIGHTTIME
#         x_original = [0, 100]
#         nighttime = []
#         for grid in sky_view.columns:
#             nighttime.append(
#                 pd.DataFrame(
#                     interp1d(x_original, y_original[~sun_up_bool])(sky_view[grid])
#                 ).dropna(axis=1)
#             )
#         nighttime = pd.concat(nighttime, axis=1)
#         nighttime.index = total_irradiance.index[~sun_up_bool]
#         nighttime.columns = total_irradiance.columns

#         interpolated_result = (
#             pd.concat([nighttime, daytime], axis=0)
#             .sort_index()
#             .interpolate()
#             .ewm(span=1.5)
#             .mean()
#         )

#         return interpolated_result

#     @staticmethod
#     def _angle_from_north(vector: List[float]) -> float:
#         """For an X, Y vector, determine the clockwise angle to north at [0, 1].

#         Args:
#             vector (List[float]): A vector of length 2.

#         Returns:
#             float: The angle between vector and north in degrees clockwise from [0, 1].
#         """
#         north = [0, 1]
#         angle1 = np.arctan2(*north[::-1])
#         angle2 = np.arctan2(*vector[::-1])
#         return np.rad2deg((angle1 - angle2) % (2 * np.pi))

#     @cached_property
#     def mean_radiant_temperature_matrix(self) -> pd.DataFrame:
#         """Determine the mean radiant temperature based on a shaded/unshaded ground surface, and the annual spatial incident total radiation.

#         Returns:
#             pd.DataFrame: A dataframe with the mean radiant temperature.
#         """

#         mrt_path = (
#             self.spatial_comfort.simulation_directory
#             / "mean_radiant_temperature_matrix.h5"
#         )

#         if mrt_path.exists():
#             print(
#                 f"- Loading mean-radiant-temperature data from {self.spatial_comfort.simulation_directory.name}"
#             )
#             return pd.read_hdf(mrt_path, "df")

#         print(
#             f"- Processing mean-radiant-temperature data for {self.spatial_comfort.simulation_directory.name}"
#         )
#         mrt = self._interpolate_between_unshaded_shaded(
#             self.spatial_comfort.unshaded_typology_result.mean_radiant_temperature,
#             self.spatial_comfort.shaded_typology_result.mean_radiant_temperature,
#             self.total_irradiance_matrix,
#             self.sky_view,
#             np.array(self.spatial_comfort.epw.global_horizontal_radiation.values) > 0,
#         ).astype(np.float16)
#         mrt.to_hdf(mrt_path, "df", complevel=9, complib="blosc")
#         return mrt

#     @cached_property
#     def points_xy(self) -> List[List[float]]:
#         """Return the x and y coordinates of the points in the grid."""
#         reshaped_points = (
#             self.points.unstack()
#             .reset_index()
#             .pivot(index=["level_0", "level_2"], values=0, columns="level_1")
#             .dropna()
#         )
#         reshaped_points.columns.name = None
#         reshaped_points.index.names = (None, None)
#         return reshaped_points[["x", "y"]].values

#     @cached_property
#     def wind_speed_direction(self) -> List[List[float]]:
#         """Return a list of all wind speeds and directions"""
#         return np.stack([self.wind_speed, self.wind_direction], axis=1)

#     @cached_property
#     def wind_speed_direction_unique(self) -> List[List[float]]:
#         """Return a list of every unique wind speed and direction combination."""
#         return np.unique(self.wind_speed_direction, axis=0)

#     @cached_property
#     def wind_speed_direction_indices(self) -> List[int]:
#         """Return a list of indices for every unique wind speed and direction combination, to map these back to the original hourly set of wind speeds and directions."""
#         return np.array(
#             [
#                 np.all(self.wind_speed_direction_unique == ws_wd, axis=1).argmax()
#                 for ws_wd in self.wind_speed_direction
#             ]
#         )

#     @cached_property
#     def moisture_source_points(self) -> List[List[List[float]]]:
#         """Return a list of all moisture source points, for each moisture source found in the simulation directory."""
#         return [i.points_xy for i in self.spatial_comfort.moisture_sources]

#     @cached_property
#     def moisture_source_magnitudes(self) -> List[float]:
#         """Return a list of all moisture source magnitudes, for each moisture source found in the simulation directory."""
#         return [i.magnitude for i in self.spatial_comfort.moisture_sources]

#     @cached_property
#     def moisture_source_vectors(self) -> List[List[List[List[float]]]]:
#         """Return a list of all moisture source vectors (between source point and sensor points), for each moisture source found in the simulation directory."""
#         return [
#             np.subtract(i, self.points_xy[:, None]) for i in self.moisture_source_points
#         ]

#     @cached_property
#     def moisture_source_distances(self) -> List[List[List[List[float]]]]:
#         """Return a list of all moisture source distances (between source point and sensor points), for each moisture source found in the simulation directory."""
#         return [np.linalg.norm(i, axis=2) for i in self.moisture_source_vectors]

#     @cached_property
#     def moisture_source_angles(self) -> List[List[List[float]]]:
#         """Return a list of all moisture source relative angles to North (between source point and sensor points), for each moisture source found in the simulation directory."""
#         return [self._angle_from_north(i.T).T for i in self.moisture_source_vectors]

#     @cached_property
#     def moisture_source_matrix_unique(self) -> List[List[List[float]]]:
#         """Create a matrix of moisture source evaporative cooling effectivenessess, for each moisture source found in the simulation directory."""

#         distance_around_sources_in_calm_wind = 5
#         wind_speed_multiplier_for_distances = 10
#         angle_width = 12

#         moisture_source_matrices = []
#         for moisture_source_idx, moisture_source in enumerate(
#             self.spatial_comfort.moisture_sources[0:]
#         ):  # <<<< TESTING LIMIT

#             save_path = (
#                 self.spatial_comfort.simulation_directory
#                 / f"{moisture_source.pathsafe_id}.npy"
#             )
#             if save_path.exists():
#                 print(
#                     f"- Loading moisture matrix lookup for {moisture_source}", end="\n"
#                 )
#                 moisture_matrix = np.load(save_path)
#                 moisture_source_matrices.append(moisture_matrix)
#                 continue

#             print(f"- Generating moisture matrix index for {moisture_source}", end="\n")
#             moisture_matrix = []
#             for w_idx, (ws, wd) in enumerate(
#                 self.wind_speed_direction_unique[0:]
#             ):  # <<<< TESTING LIMIT
#                 print(
#                     f"- Calculating {(w_idx + 1)/len(self.wind_speed_direction_unique):0.3%}",
#                     end="\r",
#                 )

#                 # CREATE MOISTURE PROFILE AROUND EACH POINT (based on distance from source and wind speed * multiplier)
#                 if ws == 0:
#                     magnitude_matrix_local = self._distribute(
#                         self.spatial_comfort.moisture_sources[
#                             moisture_source_idx
#                         ].magnitude,
#                         self.moisture_source_distances[moisture_source_idx],
#                         distance_around_sources_in_calm_wind,
#                         "parabolic",
#                     )
#                 else:
#                     magnitude_matrix_local = self._distribute(
#                         self.spatial_comfort.moisture_sources[
#                             moisture_source_idx
#                         ].magnitude,
#                         self.moisture_source_distances[moisture_source_idx],
#                         ws * wind_speed_multiplier_for_distances,
#                         "parabolic",
#                     )

#                 # ANGLE MASK
#                 plume_buffer = angle_width / 2
#                 if ws == 0:
#                     # Where no wind is present, use all directions, and filter using the DIRECTION MASK
#                     angle_mask_local = np.ones_like(
#                         self.moisture_source_angles[moisture_source_idx], dtype=bool
#                     )
#                 elif wd > (360 - plume_buffer):
#                     angle_mask_local = np.any(
#                         [
#                             (
#                                 self.moisture_source_angles[moisture_source_idx]
#                                 > wd - plume_buffer
#                             ),
#                             (
#                                 self.moisture_source_angles[moisture_source_idx]
#                                 < wd - 360 - wd
#                             ),
#                         ],
#                         axis=0,
#                     )
#                 elif wd < (0 + plume_buffer):
#                     angle_mask_local = np.any(
#                         [
#                             (
#                                 self.moisture_source_angles[moisture_source_idx]
#                                 < wd + plume_buffer
#                             ),
#                             (
#                                 self.moisture_source_angles[moisture_source_idx]
#                                 > 360 + wd - plume_buffer
#                             ),
#                         ],
#                         axis=0,
#                     )
#                 else:
#                     angle_mask_local = np.all(
#                         [
#                             (
#                                 self.moisture_source_angles[moisture_source_idx]
#                                 < wd + plume_buffer
#                             ),
#                             (
#                                 self.moisture_source_angles[moisture_source_idx]
#                                 > wd - plume_buffer
#                             ),
#                         ],
#                         axis=0,
#                     )

#                 # POINT-IN-TIME MOISTURE MATRIX
#                 moisture_matrix.append(
#                     np.amax(
#                         np.where(angle_mask_local, magnitude_matrix_local, 0), axis=1
#                     )
#                 )

#             np.save(save_path, moisture_matrix)

#             moisture_source_matrices.append(np.array(moisture_matrix))
#             print("\n")

#         return np.amax(moisture_source_matrices, axis=0)

#     @cached_property
#     def moisture_matrix(self) -> pd.DataFrame:
#         """Create an annual hourly spatial matrix of moisture source evaporative cooling effectivenessess."""

#         mtx_path = self.spatial_comfort.simulation_directory / "moisture_matrix.h5"

#         if mtx_path.exists():
#             print(
#                 f"- Loading moisture data from {self.spatial_comfort.simulation_directory.name}"
#             )
#             return pd.read_hdf(mtx_path, "df")

#         print(
#             f"- Processing moisture matrix data for {self.spatial_comfort.simulation_directory.name}"
#         )
#         mtx = pd.DataFrame(
#             self.moisture_source_matrix_unique[self.wind_speed_direction_indices],
#             index=self.mean_radiant_temperature_matrix.index,
#             columns=self.mean_radiant_temperature_matrix.columns,
#         ).astype(np.float16)

#         mtx.to_hdf(mtx_path, "df", complevel=9, complib="blosc")
#         return mtx

#     @cached_property
#     def dry_bulb_temperature_matrix(self) -> pd.DataFrame:
#         """Return an annual hourly spatial matrix of dry bulb temperature."""
#         save_path = (
#             self.spatial_comfort.simulation_directory / "dry_bulb_temperature_matrix.h5"
#         )

#         if save_path.exists():
#             print("- Loading dry bulb temperature matrix")
#             return pd.read_hdf(save_path, "df")
#         elif len(self.spatial_comfort.moisture_sources) == 0:
#             dbt = pd.DataFrame(
#                 np.repeat(
#                     [self.dry_bulb_temperature],
#                     [len(self.mean_radiant_temperature_matrix.values.T)],
#                     axis=0,
#                 ).T,
#                 index=self.mean_radiant_temperature_matrix.index,
#                 columns=self.mean_radiant_temperature_matrix.columns,
#             )
#             dbt.to_hdf(save_path, "df", complib="blosc", complevel=9)
#         else:
#             return self.spatial_moisture_properties()["dbt"]

#     def spatial_moisture_properties(self) -> Dict[str, pd.DataFrame]:
#         """Calculate the annual hourly spatial matrices of dry bulb temperature and relative humidity."""

#         dbt_save_path = (
#             self.spatial_comfort.simulation_directory / "dry_bulb_temperature_matrix.h5"
#         )
#         rh_save_path = (
#             self.spatial_comfort.simulation_directory / "relative_humidity_matrix.h5"
#         )

#         if dbt_save_path.exists() and rh_save_path.exists():
#             print("- Loading DBT/RH matrices")
#             dbt = pd.read_hdf(dbt_save_path, "df")
#             rh = pd.read_hdf(rh_save_path, "df")
#             return {"dbt": dbt, "rh": rh}
#         elif len(self.spatial_comfort.moisture_sources) == 0:
#             dbt = pd.DataFrame(
#                 np.repeat(
#                     [self.dry_bulb_temperature],
#                     [len(self.mean_radiant_temperature_matrix.values.T)],
#                     axis=0,
#                 ).T,
#                 index=self.mean_radiant_temperature_matrix.index,
#                 columns=self.mean_radiant_temperature_matrix.columns,
#             )
#             rh = pd.DataFrame(
#                 np.repeat(
#                     [self.relative_humidity],
#                     [len(self.mean_radiant_temperature_matrix.values.T)],
#                     axis=0,
#                 ).T,
#                 index=self.mean_radiant_temperature_matrix.index,
#                 columns=self.mean_radiant_temperature_matrix.columns,
#             )
#         else:
#             temp = []
#             for n, (dbt, rh, ms, atm) in enumerate(
#                 list(
#                     zip(
#                         *[
#                             self.dry_bulb_temperature,
#                             self.relative_humidity,
#                             self.moisture_matrix.values,
#                             self.atmospheric_station_pressure,
#                         ]
#                     )
#                 )[0:]
#             ):
#                 print(
#                     f"- Calculating DBT/RH matrices - {n/len(self.moisture_matrix.values):0.2%}",
#                     end="\r",
#                 )
#                 temp.append(evaporative_cooling_effect(dbt, rh, ms, atm))
#             dbt_mtx, rh_mtx = np.swapaxes(temp, 0, 1)

#             dbt = pd.DataFrame(
#                 dbt_mtx,
#                 index=self.moisture_matrix.index,
#                 columns=self.moisture_matrix.columns,
#             )
#             rh = pd.DataFrame(
#                 rh_mtx,
#                 index=self.moisture_matrix.index,
#                 columns=self.moisture_matrix.columns,
#             )

#         dbt.to_hdf(dbt_save_path, "df", complib="blosc", complevel=9)
#         rh.to_hdf(rh_save_path, "df", complib="blosc", complevel=9)

#         return {"dbt": dbt, "rh": rh}

#     @cached_property
#     def relative_humidity_matrix(self) -> pd.DataFrame:
#         """Return an annual hourly spatial matrix of relative humidity."""

#         save_path = (
#             self.spatial_comfort.simulation_directory / "relative_humidity_matrix.h5"
#         )

#         if save_path.exists():
#             print("- Loading relative humidity matrix")
#             return pd.read_hdf(save_path, "df")
#         elif len(self.spatial_comfort.moisture_sources) == 0:
#             rh = pd.DataFrame(
#                 np.repeat(
#                     [self.relative_humidity],
#                     [len(self.mean_radiant_temperature_matrix.values.T)],
#                     axis=0,
#                 ).T,
#                 index=self.mean_radiant_temperature_matrix.index,
#                 columns=self.mean_radiant_temperature_matrix.columns,
#             )
#             rh.to_hdf(save_path, "df", complib="blosc", complevel=9)
#         else:
#             return self.spatial_moisture_properties()["rh"]

#     @cached_property
#     def wind_speed_matrix(self) -> pd.DataFrame:
#         """Return an annual hourly spatial matrix of wind speed."""
#         save_path = self.spatial_comfort.simulation_directory / "wind_speed_matrix.h5"

#         ws = pd.DataFrame(
#             np.repeat(
#                 [self.wind_speed],
#                 [len(self.mean_radiant_temperature_matrix.values.T)],
#                 axis=0,
#             ).T,
#             index=self.mean_radiant_temperature_matrix.index,
#             columns=self.mean_radiant_temperature_matrix.columns,
#         )
#         ws.to_hdf(save_path, "df", complib="blosc", complevel=9)

#         return ws

#     @cached_property
#     def universal_thermal_climate_index_matrix(self) -> pd.DataFrame:
#         """Calculate the universal thermal climate index based on matrices of DBT, MRT, WS and RH.

#         Returns:
#             pd.DataFrame: A dataframe with the universal thermal climate index.
#         """

#         save_path = (
#             self.spatial_comfort.simulation_directory
#             / "universal_thermal_climate_index_matrix.h5"
#         )

#         if save_path.exists():
#             print(
#                 f"- Loading universal thermal climate index data from {self.spatial_comfort.simulation_directory.name}"
#             )
#             return pd.read_hdf(save_path, "df")

#         if len(self.spatial_comfort.moisture_sources) == 0:
#             print(
#                 f"- Processing universal thermal climate index data for {self.spatial_comfort.simulation_directory.name}, using interpolation method"
#             )
#             utci = self._interpolate_between_unshaded_shaded(
#                 self.spatial_comfort.unshaded_typology_result.universal_thermal_climate_index,
#                 self.spatial_comfort.shaded_typology_result.universal_thermal_climate_index,
#                 self.total_irradiance_matrix,
#                 self.sky_view,
#                 np.array(self.spatial_comfort.epw.global_horizontal_radiation.values)
#                 > 0,
#             ).astype(np.float16)
#             utci.to_hdf(save_path, "df", complevel=9, complib="blosc")
#             return utci

#         print(
#             f"- Processing universal thermal climate index data for {self.spatial_comfort.simulation_directory.name}, using moisture-matrix method"
#         )
#         utcis = []
#         for n, (dbt, mrt, ws, rh) in enumerate(
#             list(
#                 zip(
#                     *[
#                         self.dry_bulb_temperature_matrix.values,
#                         self.mean_radiant_temperature_matrix.values,
#                         self.wind_speed_matrix.values,
#                         self.relative_humidity_matrix.values,
#                     ]
#                 )
#             )
#         ):
#             print(
#                 f"- Calculating UTCI matrix - {n/len(self.dry_bulb_temperature):0.2%}",
#                 end="\r",
#             )
#             utcis.append(v_utci(dbt, mrt, ws, rh))

#         utci = pd.DataFrame(
#             np.array(utcis),
#             index=self.mean_radiant_temperature_matrix.index,
#             columns=self.mean_radiant_temperature_matrix.columns,
#         ).astype(np.float16)
#         utci.to_hdf(save_path, "df", complevel=9, complib="blosc")
#         return utci

#     @cached_property
#     def _triangulation(self) -> Triangulation:
#         """Return a triangulation of the spatial grid."""
#         x, y = self.points_xy.T
#         return create_triangulation(x, y)

#     @property
#     def plot_directory(self) -> Path:
#         """Return the path to the plot directory."""
#         plot_dir = self.spatial_comfort.simulation_directory / "plots"
#         plot_dir.mkdir(exist_ok=True, parents=True)
#         return plot_dir

#     def plot_utci_comfortable_hours(
#         self, analysis_period: AnalysisPeriod, hours: bool = False
#     ) -> Path:
#         """Return the path to the comfortable-hours plot."""

#         save_path = (
#             self.plot_directory
#             / f"time_comfortable_{'hours' if hours else 'percentage'}_{describe_analysis_period(analysis_period, True)}.png"
#         )
#         # if save_path.exists():
#         #     return save_path
#         print(f"- Plotting {save_path.stem}")

#         x, y = self.points_xy.T
#         # Filter for the analysis period
#         z_temp = self.universal_thermal_climate_index_matrix.iloc[
#             list(analysis_period.hoys_int), :
#         ]
#         z = ((z_temp >= 9) & (z_temp <= 26)).sum().values

#         if not hours:
#             z = z / len(analysis_period.hoys_int) * 100

#         fig = spatial_heatmap(
#             triangulations=[self._triangulation],
#             values=[z],
#             levels=np.linspace(10, 90, 101),
#             cmap="magma_r",
#             extend="both",
#             xlims=[x.min(), x.max()],
#             ylims=[y.min(), y.max()],
#             colorbar_label=f"Hours comfortable (out of a possible {len(analysis_period.hoys_int)})"
#             if hours
#             else f"% time comfortable (out of {len(analysis_period.hoys_int)} hours)",
#             title=f"Time comfortable (9°C-26°C UTCI) for {describe_analysis_period(analysis_period, False)}",
#         )

#         fig.savefig(save_path, dpi=200, bbox_inches="tight")

#         return save_path

#     def plot_all_utci_comfortable_hours(self) -> List[Path]:
#         """Return the paths to the default comfortable-hours plots."""

#         output = []
#         for month in range(1, 13, 1):
#             for st_hour, end_hour in list(zip(*[[0, 8], [23, 18]])):
#                 output.append(
#                     self.plot_utci_comfortable_hours(
#                         AnalysisPeriod(
#                             st_month=month,
#                             end_month=month,
#                             st_hour=st_hour,
#                             end_hour=end_hour,
#                         )
#                     )
#                 )
#                 plt.close("all")
#         return output

#     def plot_typical_utci(self, month: int, hour: int) -> Path:
#         """Return the path to the UTCI plot."""

#         if not month in range(1, 13, 1):
#             raise ValueError(f"Month must be between 1 and 12 inclusive, got {month}")
#         if not hour in range(0, 24, 1):
#             raise ValueError(f"Hour must be between 0 and 23 inclusive, got {hour}")

#         save_path = (
#             self.plot_directory
#             / f"universal_thermal_climate_index_{month:02d}_{hour:02d}.png"
#         )
#         # if save_path.exists():
#         #     return save_path
#         print(f"- Plotting {save_path.stem}")

#         x, y = self.points_xy.T

#         # Filter for the analysis period
#         z = (
#             self.universal_thermal_climate_index_matrix.groupby(
#                 [
#                     self.universal_thermal_climate_index_matrix.index.month,
#                     self.universal_thermal_climate_index_matrix.index.hour,
#                 ],
#                 axis=0,
#             )
#             .mean()
#             .loc[month, hour]
#         ).values

#         fig = spatial_heatmap(
#             triangulations=[self._triangulation],
#             values=[z],
#             levels=UTCI_LEVELS,
#             cmap=UTCI_COLORMAP,
#             extend="both",
#             norm=UTCI_BOUNDARYNORM,
#             xlims=[x.min(), x.max()],
#             ylims=[y.min(), y.max()],
#             colorbar_label="Universal thermal climate index (°C)",
#             title=f"{calendar.month_abbr[month]} {hour:02d}:00",
#         )

#         fig.savefig(save_path, dpi=200, bbox_inches="tight")

#         return save_path

#     def plot_typical_mrt(self, month: int, hour: int) -> Path:
#         """Return the path to the MRT plot."""

#         if not month in range(1, 13, 1):
#             raise ValueError(f"Month must be between 1 and 12 inclusive, got {month}")
#         if not hour in range(0, 24, 1):
#             raise ValueError(f"Hour must be between 0 and 23 inclusive, got {hour}")

#         save_path = (
#             self.plot_directory / f"mean_radiant_temperature_{month:02d}_{hour:02d}.png"
#         )
#         # if save_path.exists():
#         #     return save_path
#         print(f"- Plotting {save_path.stem}")

#         x, y = self.points_xy.T

#         # get the value limits for all MRTs
#         zmin, zmax = (
#             self.mean_radiant_temperature_matrix.min().min(),
#             self.mean_radiant_temperature_matrix.max().max(),
#         )

#         # Filter for the analysis period
#         z = (
#             self.mean_radiant_temperature_matrix.groupby(
#                 [
#                     self.mean_radiant_temperature_matrix.index.month,
#                     self.mean_radiant_temperature_matrix.index.hour,
#                 ],
#                 axis=0,
#             )
#             .mean()
#             .loc[month, hour]
#         ).values

#         fig = spatial_heatmap(
#             triangulations=[self._triangulation],
#             values=[z],
#             levels=np.linspace(zmin, zmax, 101),
#             cmap="inferno",
#             extend="both",
#             xlims=[x.min(), x.max()],
#             ylims=[y.min(), y.max()],
#             colorbar_label="Mean radiant temperature (°C)",
#             title=f"{calendar.month_abbr[month]} {hour:02d}:00",
#         )

#         fig.savefig(save_path, dpi=200, bbox_inches="tight")

#         return save_path

#     def plot_sky_view(self) -> Path:
#         """Return the path to the sky view plot."""

#         save_path = self.plot_directory / "sky_view.png"
#         # if save_path.exists():
#         #     return save_path
#         print(f"- Plotting {save_path.stem}")

#         min_val = 0
#         max_val = 100
#         step = 5
#         levels = np.arange(min_val, max_val + step, step)

#         x, y = self.points_xy.T
#         z = self.sky_view.iloc[:, 0].values

#         fig = spatial_heatmap(
#             triangulations=[self._triangulation],
#             values=[z],
#             levels=levels,
#             cmap="Spectral_r",
#             extend="neither",
#             xlims=[x.min(), x.max()],
#             ylims=[y.min(), y.max()],
#             colorbar_label="Proportion of sky visible (%)",
#             title="Proportion of sky visible",
#         )

#         fig.savefig(save_path, dpi=200, bbox_inches="tight")

#         return save_path

#     def generic_output(self) -> List[Path]:
#         """Shortcut to running all methods and generating the associated output plots.

#         Returns:
#             List[Path]: A list of all generated plots.
#         """
#         output = []

#         # Sky view plot
#         output.append(self.plot_sky_view())
#         plt.close("all")

#         # UTCI comfortable hours
#         output.append(self.plot_utci_comfortable_hours(AnalysisPeriod()))
#         plt.close("all")
#         output.append(
#             self.plot_utci_comfortable_hours(AnalysisPeriod(st_hour=8, end_hour=18))
#         )
#         plt.close("all")
#         for month in range(1, 13, 1):
#             output.append(
#                 self.plot_utci_comfortable_hours(
#                     AnalysisPeriod(st_month=month, end_month=month)
#                 )
#             )
#             plt.close("all")

#         # UTCI typical days animation
#         for month in [12, 3, 6]:
#             temp_output = []
#             for hour in range(24):
#                 temp_output.append(self.plot_typical_utci(month, hour))
#                 plt.close("all")

#             images = [Image.open(to) for to in temp_output]
#             background = Image.new("RGBA", images[0].size, (255, 255, 255))
#             images = [Image.alpha_composite(background, i) for i in images]
#             print(
#                 f"- Creating animated spatial UTCI for {calendar.month_abbr[month]}..."
#             )
#             images[0].save(
#                 self.plot_directory
#                 / f"universal_thermal_climate_index_{month:02d}.gif",
#                 save_all=True,
#                 append_images=images[1:],
#                 optimize=True,
#                 duration=333,
#                 loop=0,
#             )

#     def plot_wireframe(
#         self, include_points: bool = True, highlight_points: Dict[str, int] = None
#     ) -> Figure:
#         """A method included here to quickly plot a wireframe of the model."""
#         from collections import defaultdict

#         from shapely.geometry import MultiPolygon, Polygon
#         from shapely.ops import unary_union

#         groups = defaultdict(list)
#         for obj in self.spatial_comfort.model.faces:
#             groups[str(obj)].append(obj)

#         mps = []
#         for k, v in groups.items():
#             try:
#                 mps.append(
#                     MultiPolygon(
#                         unary_union(
#                             [
#                                 Polygon([i.to_array() for i in face.vertices])
#                                 for face in v
#                             ]
#                         )
#                     )
#                 )
#             except Exception:
#                 print(f"Failed to create multipolygon for {k}")

#         fig, ax = plt.subplots(1, 1, figsize=(30, 25))
#         ax.set_aspect("equal")
#         ax.axis("off")

#         for mp in mps:
#             for geom in mp.geoms:
#                 xs, ys = geom.exterior.xy
#                 ax.fill(xs, ys, alpha=0.5, fc="none", ec="k", fill=False)

#         x, y = self.points_xy.T
#         if include_points:
#             ax.scatter(x, y, s=0.05)
#             for n, xy in enumerate(self.points_xy):
#                 if n % 30 == 0:
#                     ax.annotate(
#                         n,
#                         xy=xy,
#                         textcoords="data",
#                         fontsize="xx-small",
#                         ha="center",
#                         va="center",
#                         c="r",
#                         rotation=45,
#                     )

#         if highlight_points is not None:
#             for k, v in highlight_points.items():
#                 for n, xy in enumerate(self.points_xy):
#                     if n != v:
#                         pass
#                     else:
#                         ax.scatter(x[n], y[n], s=10, c="red")
#                         ax.annotate(
#                             k,
#                             xy=xy,
#                             textcoords="data",
#                             fontsize="xx-large",
#                             ha="left",
#                             va="bottom",
#                             c="k",
#                             rotation=0,
#                         )

#         plt.tight_layout()

#         return fig

#     def point_summary_str(self, point_index: int, name: str = None) -> str:
#         """Create a summary string descirbing the difference between the selected point and an "openfield" condition.

#         Returns:
#             str: The summary.
#         """

#         if name is None:
#             name = f"Sensor #{point_index}"

#         up_hours_bool = (
#             to_series(
#                 self.spatial_comfort.external_comfort_result.external_comfort.epw.global_horizontal_radiation
#             )
#             > 0
#         ).values
#         daytime_hours = up_hours_bool.sum()
#         start_hour = 9
#         end_hour = 21

#         openf = to_series(
#             self.spatial_comfort.unshaded_typology_result.universal_thermal_climate_index
#         )
#         annual_comf_hours_baseline = ((openf >= 9) & (openf <= 26)).sum()
#         heat_stress_hours = (openf > 26).sum()
#         cold_stress_hours = (openf < 9).sum()
#         daytime_comf_hours_baseline = (
#             (openf >= 9) & (openf <= 26) & (up_hours_bool)
#         ).sum()
#         outstr = []
#         outstr.append("Openfield:")
#         outstr.append(
#             f"- No thermal stress is expected for {annual_comf_hours_baseline} hours in the year ({annual_comf_hours_baseline/8760:0.1%}). Heat stress is expected for {heat_stress_hours} hours ({heat_stress_hours/8760:0.1%}) and cold stress for {cold_stress_hours} hours ({cold_stress_hours/8760:0.1%})."
#         )
#         outstr.append(
#             f"- When considering only the hours between {start_hour:02d}:00 and {end_hour:02d}:00, this changes to {daytime_comf_hours_baseline} comfortable hours in the year ({daytime_comf_hours_baseline/daytime_hours:0.1%})."
#         )
#         outstr.append("")

#         tgt = self.universal_thermal_climate_index_matrix.iloc[:, point_index]
#         tgt.index = openf.index
#         difference = tgt - openf
#         peak_reduction_time = pd.to_datetime(difference.idxmin())

#         annual_comf_hours = ((tgt >= 9) & (tgt <= 26)).sum()
#         heat_stress_hours = (tgt > 26).sum()
#         cold_stress_hours = (tgt < 9).sum()
#         daytime_comf_hours = ((tgt >= 9) & (tgt <= 26) & (up_hours_bool)).sum()
#         diffr = (
#             annual_comf_hours - annual_comf_hours_baseline
#         ) / annual_comf_hours_baseline
#         diffr_daytime = (
#             daytime_comf_hours - daytime_comf_hours_baseline
#         ) / daytime_comf_hours_baseline

#         outstr.append(f"{name}:")
#         outstr.append(
#             f"- No thermal stress is expected for {annual_comf_hours} hours in the year ({annual_comf_hours/8760:0.1%}). Heat stress is expected for {heat_stress_hours} hours ({heat_stress_hours/8760:0.1%}) and cold stress for {cold_stress_hours} hours ({cold_stress_hours/8760:0.1%})."
#         )
#         outstr.append(
#             f"- When considering only the hours between {start_hour:02d}:00 and {end_hour:02d}:00, this changes to {daytime_comf_hours} comfortable hours in the year ({daytime_comf_hours/daytime_hours:0.1%})."
#         )
#         outstr.append(
#             f"- Compared to the baseline \"Openfield\" condition this represents an {diffr:0.1%} {'increase' if diffr > 0 else 'decrease'} in comfortable hours in this location annually, and a {diffr_daytime:0.1%} {'increase' if diffr_daytime > 0 else 'decrease'} in comfortable hours between {start_hour:02d}:00 and {end_hour:02d}:00."
#         )
#         outstr.append(
#             f"- A peak UTCI reduction in this location is expected to be around {difference.min():0.1f}°C, occurring on {peak_reduction_time:%B %d} at {peak_reduction_time:%H:%M}."
#         )
#         outstr.append("")

#         return "\n".join(outstr)

#     def point_summary_plots(self, point_index: int, name: str = None) -> List[Path]:
#         """Create summary plots for the selected point."""

#         openfield_utci = (
#             self.spatial_comfort.unshaded_typology_result.universal_thermal_climate_index
#         )
#         point_utci = self.universal_thermal_climate_index_matrix.iloc[:, point_index]
#         point_utci.name = "Universal Thermal Climate Index (C)"
#         point_utci = from_series(point_utci)

#         openfield_fig_path = self.plot_directory / "utci_openfield.png"
#         openfield_fig = plot_utci_heatmap_histogram(openfield_utci, title="Openfield")
#         openfield_fig.savefig(
#             openfield_fig_path, dpi=300, transparent=True, bbox_inches="tight"
#         )

#         if name is None:
#             name = f"Sensor #{point_index}"

#         point_fig_path = self.plot_directory / f"utci_{name}.png"
#         point_fig = plot_utci_heatmap_histogram(point_utci, title=name)
#         point_fig.savefig(
#             point_fig_path, dpi=300, transparent=True, bbox_inches="tight"
#         )

#         # comparison
#         cmap = colormap_sequential(["#00A9E0", "w", "#702F8A"])
#         diff_fig_path = self.plot_directory / f"utci_{name}_difference.png"
#         diff_fig = timeseries_heatmap(
#             point_utci - openfield_utci,
#             cmap=cmap,
#             title=f"Difference between Openfield UTCI and {name}",
#             vlims=[-10, 10],
#         )
#         diff_fig.savefig(diff_fig_path, dpi=300, transparent=True, bbox_inches="tight")

#         # diurnal
#         diurnal_fig_path = self.plot_directory / f"utci_{name}_diurnal.png"
#         diurnal_fig = utci_comparison_diurnal(
#             [openfield_utci, point_utci], ["Openfield", name]
#         )
#         diurnal_fig.savefig(
#             diurnal_fig_path, dpi=300, transparent=True, bbox_inches="tight"
#         )

#         return [openfield_fig_path, point_fig_path, diff_fig_path, diurnal_fig_path]
