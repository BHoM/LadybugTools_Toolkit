from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from scipy.interpolate import interp1d
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from honeybee_extension.results import load_ill, load_pts, load_res, make_annual
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug_extension.datacollection import from_json, to_array, to_json, to_series
from scipy import interpolate

from external_comfort.material import MATERIALS
from external_comfort.external_comfort import ExternalComfort, ExternalComfortResult
from external_comfort.shelter import Shelter
from external_comfort.typology import Typology, TypologyResult


@dataclass
class SpatialComfort:
    simulation_directory: Path = field(init=True, repr=True)
    epw: EPW = field(init=True, repr=True)
    ground_material: _EnergyMaterialOpaqueBase = field(init=True, repr=True, default=MATERIALS["ConcreteHeavyweight"])
    shade_material: _EnergyMaterialOpaqueBase = field(init=True, repr=True, default=MATERIALS["Fabric"])

    def __post_init__(self) -> SpatialComfort:
        object.__setattr__(self, "simulation_directory", Path(self.simulation_directory))
        self._simulation_validity()

    def _simulation_validity(self) -> None:


        if (self.simulation_directory is None) or (self.simulation_directory == ""):
            raise ValueError("Simulation directory is not set.")

        annual_irradiance_directory = self.simulation_directory / "annual_irradiance"
        if not annual_irradiance_directory.exists():
            raise ValueError(f"Annual-irradiance data is not available at {annual_irradiance_directory}.")
        
        sky_view_directory = self.simulation_directory / "sky_view"
        if not (sky_view_directory).exists():
            raise ValueError(f"Sky-view data is not available at {sky_view_directory}.")

@dataclass
class SpatialComfortResult:
    spatial_comfort: SpatialComfort = field(init=True, repr=True)

    points: pd.DataFrame = field(init=False, repr=False)
    sky_view: pd.DataFrame = field(init=False, repr=False)
    total_irradiance: pd.DataFrame = field(init=False, repr=False)

    shaded_utci: pd.Series = field(init=False, repr=False)
    unshaded_utci: pd.Series = field(init=False, repr=False)
    shaded_gnd: pd.Series = field(init=False, repr=False)
    unshaded_gnd: pd.Series = field(init=False, repr=False)
    shaded_mrt: pd.Series = field(init=False, repr=False)
    unshaded_mrt: pd.Series = field(init=False, repr=False)
    
    _sun_up_bool: np.ndarray(float) = field(init=False, repr=False)
    

    def __post_init__(self) -> SpatialComfortResult:
        
        object.__setattr__(self, "points", self._points())
        object.__setattr__(self, "sky_view", self._sky_view())
        object.__setattr__(self, "total_irradiance", self._total_irradiance())

        # Run external comfort simulation
        external_comfort_result_dict = self._external_comfort_result()
        for k, v in external_comfort_result_dict.items():
            object.__setattr__(self, k, v)

        # TODO - post processing of the case following data generation

        # Post-process spatial comfort
        # object.__setattr__(self, "_sun_up_bool", to_array(self.spatial_comfort.epw.global_horizontal_radiation) > 0)


    def _points(self) -> pd.DataFrame:
        """Return the points results from the simulation directory, and create the H5 file to store them as compressed objects if not already done.

        Returns:
            pd.DataFrame: A dataframe with the points locations.
        """        

        points_path = self.spatial_comfort.simulation_directory / "points.h5"

        if points_path.exists():
            return pd.read_hdf(points_path, "df")
        else:
            points_files = list((self.spatial_comfort.simulation_directory / "sky_view" / "model" / "grid").glob("*.pts"))
            points = load_pts(points_files)
            points.to_hdf(points_path, "df", complevel=9, complib="blosc")
        return points
    
    def _total_irradiance(self) -> pd.DataFrame:
        """Get the total irradiance from the simulation directory.

        Returns:
            pd.DataFrame: A dataframe with the total irradiance.
        """
        total_irradiance_path = self.spatial_comfort.simulation_directory / "total_irradiance.h5"

        if total_irradiance_path.exists():
            return pd.read_hdf(total_irradiance_path, "df")
        else:
            ill_files = list(
                (
                    self.spatial_comfort.simulation_directory / "annual_irradiance" / "results" / "total"
                ).glob("*.ill")
            )
            total_irradiance = make_annual(
                load_ill(ill_files)
            ).fillna(0)
            total_irradiance.to_hdf(
                total_irradiance_path, "df", complevel=9, complib="blosc"
            )
        return total_irradiance
 
    def _sky_view(self) -> pd.DataFrame:
        """Get the sky view from the simulation directory.

        Returns:
            pd.DataFrame: The sky view dataframe.
        """        

        sky_view_path = self.spatial_comfort.simulation_directory / "sky_view.h5"
        
        if sky_view_path.exists():
            return pd.read_hdf(sky_view_path, "df")
        else:
            res_files = list((self.spatial_comfort.simulation_directory / "sky_view" / "results").glob("*.res"))
            sky_view = load_res(res_files)
            sky_view.to_hdf(sky_view_path, "df", complevel=9, complib="blosc")
        return sky_view

    def _external_comfort_result(self) -> Dict[str, pd.Series]:
        
        shaded_utci_path = self.spatial_comfort.simulation_directory / "shaded.utci"
        unshaded_utci_path = self.spatial_comfort.simulation_directory / "unshaded.utci"
        shaded_gnd_path = self.spatial_comfort.simulation_directory / "shaded.gnd"
        unshaded_gnd_path = self.spatial_comfort.simulation_directory / "unshaded.gnd"
        shaded_mrt_path = self.spatial_comfort.simulation_directory / "shaded.mrt"
        unshaded_mrt_path = self.spatial_comfort.simulation_directory / "unshaded.mrt"

        if not all([p.exists() for p in [shaded_utci_path, unshaded_utci_path, shaded_gnd_path, unshaded_gnd_path, shaded_mrt_path, unshaded_mrt_path]]):
            external_comfort_result = ExternalComfortResult(external_comfort=ExternalComfort(self.spatial_comfort.epw, self.spatial_comfort.ground_material, self.spatial_comfort.shade_material))

            unshaded = TypologyResult(Typology(name="Unshaded", shelters=None), external_comfort_result)
            shaded = TypologyResult(Typology(name="Shaded", shelters=[Shelter(porosity=0, altitude_range=[0, 90], azimuth_range=[0, 360])]), external_comfort_result)

            shaded_utci = to_series(shaded.universal_thermal_climate_index)
            unshaded_utci = to_series(unshaded.universal_thermal_climate_index)
            shaded_gnd = to_series(external_comfort_result.shaded_below_temperature)
            unshaded_gnd = to_series(external_comfort_result.unshaded_below_temperature)
            shaded_mrt = to_series(shaded.mean_radiant_temperature)
            unshaded_mrt = to_series(unshaded.mean_radiant_temperature)

            shaded_utci.to_csv(shaded_utci_path)
            unshaded_utci.to_csv(unshaded_utci_path)
            shaded_gnd.to_csv(shaded_gnd_path)
            unshaded_gnd.to_csv(unshaded_gnd_path)
            shaded_mrt.to_csv(shaded_mrt_path)
            unshaded_mrt.to_csv(unshaded_mrt_path)

            d = {
                "shaded_utci": shaded_utci,
                "unshaded_utci": unshaded_utci,
                "shaded_gnd": shaded_gnd,
                "unshaded_gnd": unshaded_gnd,
                "shaded_mrt": shaded_mrt,
                "unshaded_mrt": unshaded_mrt,
            }
            return d
        else:
            d = {
                "shaded_utci": pd.read_csv(shaded_utci_path, index_col=0, header=0, parse_dates=True),
                "unshaded_utci": pd.read_csv(unshaded_utci_path, index_col=0, header=0, parse_dates=True),
                "shaded_gnd": pd.read_csv(shaded_gnd_path, index_col=0, header=0, parse_dates=True),
                "unshaded_gnd": pd.read_csv(unshaded_gnd_path, index_col=0, header=0, parse_dates=True),
                "shaded_mrt": pd.read_csv(shaded_mrt_path, index_col=0, header=0, parse_dates=True),
                "unshaded_mrt": pd.read_csv(unshaded_mrt_path, index_col=0, header=0, parse_dates=True),
            }
            return d


    #     self._spatial_mrt: pd.DataFrame = None
    #     self._spatial_utci: pd.DataFrame = None
    #     self._spatial_gnd: pd.DataFrame = None


    # @property
    # def shaded_utci(self) -> pd.Series:
    #     shaded_utci_path = self.simulation_directory / "shaded.utci"
        
    #     if self._shaded_utci is not None:
    #         return self._shaded_utci
        
    #     if not shaded_utci_path.exists():
    #         self.__run_openfield()
        
    #     self._shaded_utci = pd.read_csv(shaded_utci_path, index_col=0, header=0, parse_dates=True)

    #     return self._shaded_utci
    
    # @property
    # def unshaded_utci(self) -> pd.Series:
    #     unshaded_utci_path = self.simulation_directory / "unshaded.utci"
        
    #     if self._unshaded_utci is not None:
    #         return self._unshaded_utci
        
    #     if not unshaded_utci_path.exists():
    #         self.__run_openfield()
        
    #     self._unshaded_utci = pd.read_csv(unshaded_utci_path, index_col=0, header=0, parse_dates=True)

    #     return self._unshaded_utci

    # @property
    # def shaded_mrt(self) -> pd.Series:
    #     shaded_mrt_path = self.simulation_directory / "shaded.mrt"
        
    #     if self._shaded_mrt is not None:
    #         return self._shaded_mrt
        
    #     if not shaded_mrt_path.exists():
    #         self.__run_openfield()
        
    #     self._shaded_mrt = pd.read_csv(shaded_mrt_path, index_col=0, header=0, parse_dates=True)

    #     return self._shaded_mrt
    
    # @property
    # def unshaded_mrt(self) -> pd.Series:
    #     unshaded_mrt_path = self.simulation_directory / "unshaded.mrt"
        
    #     if self._unshaded_mrt is not None:
    #         return self._unshaded_mrt
        
    #     if not unshaded_mrt_path.exists():
    #         self.__run_openfield()
        
    #     self._unshaded_mrt = pd.read_csv(unshaded_mrt_path, index_col=0, header=0, parse_dates=True)

    #     return self._unshaded_mrt

    # @property
    # def shaded_gnd(self) -> pd.Series:
    #     shaded_gnd_path = self.simulation_directory / "shaded.gnd"
        
    #     if self._shaded_gnd is not None:
    #         return self._shaded_gnd
        
    #     if not shaded_gnd_path.exists():
    #         self.__run_openfield()
        
    #     self._shaded_gnd = pd.read_csv(shaded_gnd_path, index_col=0, header=0, parse_dates=True)

    #     return self._shaded_gnd
    
    # @property
    # def unshaded_gnd(self) -> pd.Series:
    #     unshaded_gnd_path = self.simulation_directory / "unshaded.gnd"
        
    #     if self._unshaded_gnd is not None:
    #         return self._unshaded_gnd
        
    #     if not unshaded_gnd_path.exists():
    #         self.__run_openfield()
        
    #     self._unshaded_gnd = pd.read_csv(unshaded_gnd_path, index_col=0, header=0, parse_dates=True)

    #     return self._unshaded_gnd
    
    # @property
    # def dry_bulb_temperature(self) -> pd.Series:
    #     return to_series(self.epw.dry_bulb_temperature)
    
    # @property
    # def wind_speed(self) -> pd.Series:
    #     return to_series(self.epw.wind_speed)
    
    # @property
    # def wind_direction(self) -> pd.Series:
    #     return to_series(self.epw.wind_direction)

    # def __run_openfield(self) -> None:
    #     """Run the Openfield process to generate the shaded and unshaded MRT, surface temperature and UTCI values."""
        
    #     openfield = Openfield(
    #         self.epw, self.ground_material, self.shade_material, run=True
    #     )

    #     unshaded_typology = Typology(
    #         openfield,
    #         name="Openfield",
    #         evaporative_cooling_effectiveness=0,
    #         shelter=Shelter(
    #             altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1
    #         ),
    #     )
        
    #     shaded_typology = Typology(
    #         openfield,
    #         name="Enclosed",
    #         evaporative_cooling_effectiveness=0,
    #         shelter=Shelter(
    #             altitude_range=[0, 90], azimuth_range=[0, 360], porosity=0
    #         ),
    #     )

    #     to_series(openfield.unshaded_below_temperature).to_csv(self.simulation_directory / "unshaded.gnd")
    #     to_series(openfield.shaded_below_temperature).to_csv(self.simulation_directory / "shaded.gnd")

    #     to_series(openfield.unshaded_mean_radiant_temperature).to_csv(self.simulation_directory / "unshaded.mrt")
    #     to_series(openfield.shaded_mean_radiant_temperature).to_csv(self.simulation_directory / "shaded.mrt")

    #     to_series(unshaded_typology.effective_utci()).to_csv(self.simulation_directory / "unshaded.utci")
    #     to_series(shaded_typology.effective_utci()).to_csv(self.simulation_directory / "shaded.utci")

    # def spatial_mrt(self) -> pd.DataFrame:
    #     """Return the annual MRT values for each point in the simulation."""

    #     spatial_mrt_path = self.simulation_directory / "spatial_mrt.h5"

    #     if self._spatial_mrt is not None:
    #         return self._spatial_mrt
        
    #     if not spatial_mrt_path.exists():
    #         y_original = np.stack([self.shaded_mrt.squeeze().values, self.unshaded_mrt.squeeze().values], axis=1)

    #         # DAYTIME
    #         irradiance_grp = self.total_irradiance[self.sun_up_bool].groupby(self.total_irradiance.columns.get_level_values(0), axis=1)
    #         daytimes = []
    #         for grid in self.sky_view.columns:
    #             irradiance_range = np.vstack([irradiance_grp.min()[grid], irradiance_grp.max()[grid]]).T
    #             new_min = y_original[self.sun_up_bool].min(axis=1)
    #             new_max = y_original[self.sun_up_bool].max(axis=1)
    #             old_min = irradiance_range.min(axis=1)
    #             old_max = irradiance_range.max(axis=1)
    #             old_value = self.total_irradiance[grid][self.sun_up_bool].values
    #             new_value = ((old_value.T - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    #             daytimes.append(pd.DataFrame(new_value.T))

    #         daytime = pd.concat(daytimes, axis=1)
    #         daytime.index = self.total_irradiance.index[self.sun_up_bool]
    #         daytime.columns = self.total_irradiance.columns

    #         # NIGHTTIME
    #         x_original = [0, 100]
    #         nighttime = []
    #         for grid in self.sky_view.columns:
    #             print(f"{grid} - Interpolating nighttime MRT values")
    #             nighttime.append(pd.DataFrame(interp1d(x_original, y_original[~self.sun_up_bool])(self.sky_view[grid])).dropna(axis=1))
    #         nighttime = pd.concat(nighttime, axis=1)
    #         nighttime.index = self.total_irradiance.index[~self.sun_up_bool]
    #         nighttime.columns = self.total_irradiance.columns

    #         self._spatial_mrt = pd.concat([nighttime, daytime], axis=0).sort_index().interpolate().ewm(span=1.5).mean()
    #         self._spatial_mrt.to_hdf(spatial_mrt_path, "df", complevel=9, complib="blosc")

    #         return self._spatial_mrt
    #     else:
    #         self._spatial_mrt = pd.read_hdf(spatial_mrt_path, "df")
    #         return self._spatial_mrt
        
    # def spatial_utci(self) -> pd.DataFrame:
    #     """Return the annual UTCI values for each point in the simulation."""

    #     spatial_utci_path = self.simulation_directory / "spatial_utci.h5"

    #     if self._spatial_utci is not None:
    #         return self._spatial_utci
        
    #     if not spatial_utci_path.exists():
    #         y_original = np.stack([self.shaded_utci.squeeze().values, self.unshaded_utci.squeeze().values], axis=1)

    #         # DAYTIME
    #         irradiance_grp = self.total_irradiance[self.sun_up_bool].groupby(self.total_irradiance.columns.get_level_values(0), axis=1)
    #         daytimes = []
    #         for grid in self.sky_view.columns:
    #             irradiance_range = np.vstack([irradiance_grp.min()[grid], irradiance_grp.max()[grid]]).T
    #             new_min = y_original[self.sun_up_bool].min(axis=1)
    #             new_max = y_original[self.sun_up_bool].max(axis=1)
    #             old_min = irradiance_range.min(axis=1)
    #             old_max = irradiance_range.max(axis=1)
    #             old_value = self.total_irradiance[grid][self.sun_up_bool].values
    #             new_value = ((old_value.T - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    #             daytimes.append(pd.DataFrame(new_value.T))

    #         daytime = pd.concat(daytimes, axis=1)
    #         daytime.index = self.total_irradiance.index[self.sun_up_bool]
    #         daytime.columns = self.total_irradiance.columns

    #         # NIGHTTIME
    #         x_original = [0, 100]
    #         nighttime = []
    #         for grid in self.sky_view.columns:
    #             print(f"{grid} - Interpolating nighttime utci values")
    #             nighttime.append(pd.DataFrame(interp1d(x_original, y_original[~self.sun_up_bool])(self.sky_view[grid])).dropna(axis=1))
    #         nighttime = pd.concat(nighttime, axis=1)
    #         nighttime.index = self.total_irradiance.index[~self.sun_up_bool]
    #         nighttime.columns = self.total_irradiance.columns

    #         self._spatial_utci = pd.concat([nighttime, daytime], axis=0).sort_index().interpolate().ewm(span=1.5).mean()
    #         self._spatial_utci.to_hdf(spatial_utci_path, "df", complevel=9, complib="blosc")

    #         return self._spatial_utci
    #     else:
    #         self._spatial_utci = pd.read_hdf(spatial_utci_path, "df")
    #         return self._spatial_utci
    
    # def spatial_gnd(self) -> pd.DataFrame:
    #     """Return the annual Ground Temperature values for each point in the simulation."""

    #     spatial_gnd_path = self.simulation_directory / "spatial_gnd.h5"

    #     if self._spatial_gnd is not None:
    #         return self._spatial_gnd
        
    #     if not spatial_gnd_path.exists():
    #         y_original = np.stack([self.shaded_gnd.squeeze().values, self.unshaded_gnd.squeeze().values], axis=1)

    #         # DAYTIME
    #         irradiance_grp = self.total_irradiance[self.sun_up_bool].groupby(self.total_irradiance.columns.get_level_values(0), axis=1)
    #         daytimes = []
    #         for grid in self.sky_view.columns:
    #             irradiance_range = np.vstack([irradiance_grp.min()[grid], irradiance_grp.max()[grid]]).T
    #             new_min = y_original[self.sun_up_bool].min(axis=1)
    #             new_max = y_original[self.sun_up_bool].max(axis=1)
    #             old_min = irradiance_range.min(axis=1)
    #             old_max = irradiance_range.max(axis=1)
    #             old_value = self.total_irradiance[grid][self.sun_up_bool].values
    #             new_value = ((old_value.T - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    #             daytimes.append(pd.DataFrame(new_value.T))

    #         daytime = pd.concat(daytimes, axis=1)
    #         daytime.index = self.total_irradiance.index[self.sun_up_bool]
    #         daytime.columns = self.total_irradiance.columns

    #         # NIGHTTIME
    #         x_original = [0, 100]
    #         nighttime = []
    #         for grid in self.sky_view.columns:
    #             print(f"{grid} - Interpolating nighttime gnd values")
    #             nighttime.append(pd.DataFrame(interp1d(x_original, y_original[~self.sun_up_bool])(self.sky_view[grid])).dropna(axis=1))
    #         nighttime = pd.concat(nighttime, axis=1)
    #         nighttime.index = self.total_irradiance.index[~self.sun_up_bool]
    #         nighttime.columns = self.total_irradiance.columns

    #         self._spatial_gnd = pd.concat([nighttime, daytime], axis=0).sort_index().interpolate().ewm(span=1.5).mean()
    #         self._spatial_gnd.to_hdf(spatial_gnd_path, "df", complevel=9, complib="blosc")

    #         return self._spatial_gnd
    #     else:
    #         self._spatial_gnd = pd.read_hdf(spatial_gnd_path, "df")
    #         return self._spatial_gnd

    # @property
    # def xlim(self) -> List[float]:
    #     """Return the x-axis limits for the plot."""
    #     maxima = self.points.groupby(self.points.columns.get_level_values(1), axis=1).max().max()
    #     minima = self.points.groupby(self.points.columns.get_level_values(1), axis=1).min().min()
    #     return [minima.x, maxima.x]
    
    # @property
    # def ylim(self) -> List[float]:
    #     """Return the y-axis limits for the plot."""
    #     maxima = self.points.groupby(self.points.columns.get_level_values(1), axis=1).max().max()
    #     minima = self.points.groupby(self.points.columns.get_level_values(1), axis=1).min().min()
    #     return [minima.y, maxima.y]

    # # TODO - add method to figure out appropriate trimesh alpha value




