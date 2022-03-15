from __future__ import annotations

import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from pathlib import Path
from scipy.interpolate import interp1d
from typing import List, Union

import numpy as np
import pandas as pd
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from honeybee_extension.results import _make_annual, load_ill, load_pts, load_res
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug_extension.datacollection import from_json, to_array, to_json, to_series
from scipy import interpolate

from external_comfort.material import material_from_string
from external_comfort.openfield import Openfield
from external_comfort.shelter import Shelter
from external_comfort.typology import Typology


class SpatialComfort:
    def __init__(
        self,
        simulation_directory: Union[Path, str],
        epw: Union[str, EPW],
        ground_material: Union[str, _EnergyMaterialOpaqueBase] = material_from_string(
            "CONCRETE_HEAVYWEIGHT"
        ),
        shade_material: Union[str, _EnergyMaterialOpaqueBase] = material_from_string(
            "FABRIC"
        ),
    ) -> SpatialComfort:
        """
        Initialize a SpatialComfort object.

        Args:
            simulation_directory: Path to the simulation directory containing pre-simulated annual-irradiance and sky-view results.
            epw: Path to the epw file associated with the simulation.
            ground_material: Material of the ground. This should be representative of the majority ground-type in the simulation
            shade_material: Material of the shade. This should be representative of the majority shade-type in the simulation
        
        Returns:
            SpatialComfort object
        """
        self.simulation_directory = Path(simulation_directory)
        self.ground_material = (
            material_from_string(ground_material)
            if isinstance(ground_material, str)
            else ground_material
        )
        self.shade_material = (
            material_from_string(shade_material)
            if isinstance(shade_material, str)
            else shade_material
        )
        self.epw = EPW(epw) if isinstance(epw, str) else epw
        self.sun_up_bool = to_array(self.epw.global_horizontal_radiation) > 0

        self._total_irradiance: pd.DataFrame = None
        self._sky_view: pd.DataFrame = None
        self._points: pd.DataFrame = None

        self._shaded_utci: pd.Series = None
        self._unshaded_utci: pd.Series = None
        self._shaded_gnd: pd.Series = None
        self._unshaded_gnd: pd.Series = None
        self._shaded_mrt: pd.Series = None
        self._unshaded_mrt: pd.Series = None

        self._spatial_mrt: pd.DataFrame = None
        self._spatial_utci: pd.DataFrame = None
        self._spatial_gnd: pd.DataFrame = None

    @property
    def total_irradiance(self) -> pd.DataFrame:
        """Return the irradiance results from the simulation directory, and load them into the object if they're not already loaded."""

        total_irradiance_path = self.simulation_directory / "total_irradiance.h5"

        if self._total_irradiance is not None:
            return self._total_irradiance

        if total_irradiance_path.exists():
            self._total_irradiance = pd.read_hdf(total_irradiance_path, "df")
        else:
            ill_files = list(
                (
                    self.simulation_directory / "annual_irradiance" / "results" / "total"
                ).glob("*.ill")
            )
            self._total_irradiance = _make_annual(
                load_ill(ill_files)
            ).fillna(0)
            self._total_irradiance.to_hdf(
                total_irradiance_path, "df", complevel=9, complib="blosc"
            )
        return self._total_irradiance
    
    @property
    def sky_view(self) -> pd.DataFrame:
        """Return the sky view results from the simulation directory, and load them into the object if they're not already loaded."""

        sky_view_path = self.simulation_directory / "sky_view.h5"

        if self._sky_view is not None:
            return self._sky_view
        
        if sky_view_path.exists():
            self._sky_view = pd.read_hdf(sky_view_path, "df")
        else:
            if self._sky_view is None:
                res_files = list((self.simulation_directory / "sky_view" / "results").glob("*.res"))
                self._sky_view = load_res(res_files)
                self._sky_view.to_hdf(sky_view_path, "df", complevel=9, complib="blosc")

        return self._sky_view

    @property
    def points(self) -> pd.DataFrame:
        """Return the points results from the simulation directory, and load them into the object if they're not already loaded."""

        points_path = self.simulation_directory / "points.h5"

        if self._points is not None:
            return self._points

        if points_path.exists():
            self._points = pd.read_hdf(points_path, "df")
        else:
            points_files = list((self.simulation_directory / "sky_view" / "model" / "grid").glob("*.pts"))
            self._points = load_pts(points_files)
            self._points.to_hdf(points_path, "df", complevel=9, complib="blosc")
        return self._points

    @property
    def shaded_utci(self) -> pd.Series:
        shaded_utci_path = self.simulation_directory / "shaded.utci"
        
        if self._shaded_utci is not None:
            return self._shaded_utci
        
        if not shaded_utci_path.exists():
            self.__run_openfield()
        
        self._shaded_utci = pd.read_csv(shaded_utci_path, index_col=0, header=0, parse_dates=True)

        return self._shaded_utci
    
    @property
    def unshaded_utci(self) -> pd.Series:
        unshaded_utci_path = self.simulation_directory / "unshaded.utci"
        
        if self._unshaded_utci is not None:
            return self._unshaded_utci
        
        if not unshaded_utci_path.exists():
            self.__run_openfield()
        
        self._unshaded_utci = pd.read_csv(unshaded_utci_path, index_col=0, header=0, parse_dates=True)

        return self._unshaded_utci

    @property
    def shaded_mrt(self) -> pd.Series:
        shaded_mrt_path = self.simulation_directory / "shaded.mrt"
        
        if self._shaded_mrt is not None:
            return self._shaded_mrt
        
        if not shaded_mrt_path.exists():
            self.__run_openfield()
        
        self._shaded_mrt = pd.read_csv(shaded_mrt_path, index_col=0, header=0, parse_dates=True)

        return self._shaded_mrt
    
    @property
    def unshaded_mrt(self) -> pd.Series:
        unshaded_mrt_path = self.simulation_directory / "unshaded.mrt"
        
        if self._unshaded_mrt is not None:
            return self._unshaded_mrt
        
        if not unshaded_mrt_path.exists():
            self.__run_openfield()
        
        self._unshaded_mrt = pd.read_csv(unshaded_mrt_path, index_col=0, header=0, parse_dates=True)

        return self._unshaded_mrt

    @property
    def shaded_gnd(self) -> pd.Series:
        shaded_gnd_path = self.simulation_directory / "shaded.gnd"
        
        if self._shaded_gnd is not None:
            return self._shaded_gnd
        
        if not shaded_gnd_path.exists():
            self.__run_openfield()
        
        self._shaded_gnd = pd.read_csv(shaded_gnd_path, index_col=0, header=0, parse_dates=True)

        return self._shaded_gnd
    
    @property
    def unshaded_gnd(self) -> pd.Series:
        unshaded_gnd_path = self.simulation_directory / "unshaded.gnd"
        
        if self._unshaded_gnd is not None:
            return self._unshaded_gnd
        
        if not unshaded_gnd_path.exists():
            self.__run_openfield()
        
        self._unshaded_gnd = pd.read_csv(unshaded_gnd_path, index_col=0, header=0, parse_dates=True)

        return self._unshaded_gnd
    
    @property
    def dry_bulb_temperature(self) -> pd.Series:
        return to_series(self.epw.dry_bulb_temperature)
    
    @property
    def wind_speed(self) -> pd.Series:
        return to_series(self.epw.wind_speed)
    
    @property
    def wind_direction(self) -> pd.Series:
        return to_series(self.epw.wind_direction)

    def __run_openfield(self) -> None:
        """Run the Openfield process to generate the shaded and unshaded MRT, surface temperature and UTCI values."""
        
        openfield = Openfield(
            self.epw, self.ground_material, self.shade_material, run=True
        )

        unshaded_typology = Typology(
            openfield,
            name="Openfield",
            evaporative_cooling_effectiveness=0,
            shelter=Shelter(
                altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1
            ),
        )
        
        shaded_typology = Typology(
            openfield,
            name="Enclosed",
            evaporative_cooling_effectiveness=0,
            shelter=Shelter(
                altitude_range=[0, 90], azimuth_range=[0, 360], porosity=0
            ),
        )

        to_series(openfield.unshaded_below_temperature).to_csv(self.simulation_directory / "unshaded.gnd")
        to_series(openfield.shaded_below_temperature).to_csv(self.simulation_directory / "shaded.gnd")

        to_series(openfield.unshaded_mean_radiant_temperature).to_csv(self.simulation_directory / "unshaded.mrt")
        to_series(openfield.shaded_mean_radiant_temperature).to_csv(self.simulation_directory / "shaded.mrt")

        to_series(unshaded_typology.effective_utci()).to_csv(self.simulation_directory / "unshaded.utci")
        to_series(shaded_typology.effective_utci()).to_csv(self.simulation_directory / "shaded.utci")

    def spatial_mrt(self) -> pd.DataFrame:
        """Return the annual MRT values for each point in the simulation."""

        spatial_mrt_path = self.simulation_directory / "spatial_mrt.h5"

        if self._spatial_mrt is not None:
            return self._spatial_mrt
        
        if not spatial_mrt_path.exists():
            y_original = np.stack([self.shaded_mrt.squeeze().values, self.unshaded_mrt.squeeze().values], axis=1)

            # DAYTIME
            irradiance_grp = self.total_irradiance[self.sun_up_bool].groupby(self.total_irradiance.columns.get_level_values(0), axis=1)
            daytimes = []
            for grid in self.sky_view.columns:
                irradiance_range = np.vstack([irradiance_grp.min()[grid], irradiance_grp.max()[grid]]).T
                new_min = y_original[self.sun_up_bool].min(axis=1)
                new_max = y_original[self.sun_up_bool].max(axis=1)
                old_min = irradiance_range.min(axis=1)
                old_max = irradiance_range.max(axis=1)
                old_value = self.total_irradiance[grid][self.sun_up_bool].values
                new_value = ((old_value.T - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
                daytimes.append(pd.DataFrame(new_value.T))

            daytime = pd.concat(daytimes, axis=1)
            daytime.index = self.total_irradiance.index[self.sun_up_bool]
            daytime.columns = self.total_irradiance.columns

            # NIGHTTIME
            x_original = [0, 100]
            nighttime = []
            for grid in self.sky_view.columns:
                print(f"{grid} - Interpolating nighttime MRT values")
                nighttime.append(pd.DataFrame(interp1d(x_original, y_original[~self.sun_up_bool])(self.sky_view[grid])).dropna(axis=1))
            nighttime = pd.concat(nighttime, axis=1)
            nighttime.index = self.total_irradiance.index[~self.sun_up_bool]
            nighttime.columns = self.total_irradiance.columns

            self._spatial_mrt = pd.concat([nighttime, daytime], axis=0).sort_index().interpolate().ewm(span=1.5).mean()
            self._spatial_mrt.to_hdf(spatial_mrt_path, "df", complevel=9, complib="blosc")

            return self._spatial_mrt
        else:
            self._spatial_mrt = pd.read_hdf(spatial_mrt_path, "df")
            return self._spatial_mrt
        
    def spatial_utci(self) -> pd.DataFrame:
        """Return the annual UTCI values for each point in the simulation."""

        spatial_utci_path = self.simulation_directory / "spatial_utci.h5"

        if self._spatial_utci is not None:
            return self._spatial_utci
        
        if not spatial_utci_path.exists():
            y_original = np.stack([self.shaded_utci.squeeze().values, self.unshaded_utci.squeeze().values], axis=1)

            # DAYTIME
            irradiance_grp = self.total_irradiance[self.sun_up_bool].groupby(self.total_irradiance.columns.get_level_values(0), axis=1)
            daytimes = []
            for grid in self.sky_view.columns:
                irradiance_range = np.vstack([irradiance_grp.min()[grid], irradiance_grp.max()[grid]]).T
                new_min = y_original[self.sun_up_bool].min(axis=1)
                new_max = y_original[self.sun_up_bool].max(axis=1)
                old_min = irradiance_range.min(axis=1)
                old_max = irradiance_range.max(axis=1)
                old_value = self.total_irradiance[grid][self.sun_up_bool].values
                new_value = ((old_value.T - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
                daytimes.append(pd.DataFrame(new_value.T))

            daytime = pd.concat(daytimes, axis=1)
            daytime.index = self.total_irradiance.index[self.sun_up_bool]
            daytime.columns = self.total_irradiance.columns

            # NIGHTTIME
            x_original = [0, 100]
            nighttime = []
            for grid in self.sky_view.columns:
                print(f"{grid} - Interpolating nighttime utci values")
                nighttime.append(pd.DataFrame(interp1d(x_original, y_original[~self.sun_up_bool])(self.sky_view[grid])).dropna(axis=1))
            nighttime = pd.concat(nighttime, axis=1)
            nighttime.index = self.total_irradiance.index[~self.sun_up_bool]
            nighttime.columns = self.total_irradiance.columns

            self._spatial_utci = pd.concat([nighttime, daytime], axis=0).sort_index().interpolate().ewm(span=1.5).mean()
            self._spatial_utci.to_hdf(spatial_utci_path, "df", complevel=9, complib="blosc")

            return self._spatial_utci
        else:
            self._spatial_utci = pd.read_hdf(spatial_utci_path, "df")
            return self._spatial_utci
    
    def spatial_gnd(self) -> pd.DataFrame:
        """Return the annual Ground Temperature values for each point in the simulation."""

        spatial_gnd_path = self.simulation_directory / "spatial_gnd.h5"

        if self._spatial_gnd is not None:
            return self._spatial_gnd
        
        if not spatial_gnd_path.exists():
            y_original = np.stack([self.shaded_gnd.squeeze().values, self.unshaded_gnd.squeeze().values], axis=1)

            # DAYTIME
            irradiance_grp = self.total_irradiance[self.sun_up_bool].groupby(self.total_irradiance.columns.get_level_values(0), axis=1)
            daytimes = []
            for grid in self.sky_view.columns:
                irradiance_range = np.vstack([irradiance_grp.min()[grid], irradiance_grp.max()[grid]]).T
                new_min = y_original[self.sun_up_bool].min(axis=1)
                new_max = y_original[self.sun_up_bool].max(axis=1)
                old_min = irradiance_range.min(axis=1)
                old_max = irradiance_range.max(axis=1)
                old_value = self.total_irradiance[grid][self.sun_up_bool].values
                new_value = ((old_value.T - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
                daytimes.append(pd.DataFrame(new_value.T))

            daytime = pd.concat(daytimes, axis=1)
            daytime.index = self.total_irradiance.index[self.sun_up_bool]
            daytime.columns = self.total_irradiance.columns

            # NIGHTTIME
            x_original = [0, 100]
            nighttime = []
            for grid in self.sky_view.columns:
                print(f"{grid} - Interpolating nighttime gnd values")
                nighttime.append(pd.DataFrame(interp1d(x_original, y_original[~self.sun_up_bool])(self.sky_view[grid])).dropna(axis=1))
            nighttime = pd.concat(nighttime, axis=1)
            nighttime.index = self.total_irradiance.index[~self.sun_up_bool]
            nighttime.columns = self.total_irradiance.columns

            self._spatial_gnd = pd.concat([nighttime, daytime], axis=0).sort_index().interpolate().ewm(span=1.5).mean()
            self._spatial_gnd.to_hdf(spatial_gnd_path, "df", complevel=9, complib="blosc")

            return self._spatial_gnd
        else:
            self._spatial_gnd = pd.read_hdf(spatial_gnd_path, "df")
            return self._spatial_gnd

    @property
    def xlim(self) -> List[float]:
        """Return the x-axis limits for the plot."""
        maxima = self.points.groupby(self.points.columns.get_level_values(1), axis=1).max().max()
        minima = self.points.groupby(self.points.columns.get_level_values(1), axis=1).min().min()
        return [minima.x, maxima.x]
    
    @property
    def ylim(self) -> List[float]:
        """Return the y-axis limits for the plot."""
        maxima = self.points.groupby(self.points.columns.get_level_values(1), axis=1).max().max()
        minima = self.points.groupby(self.points.columns.get_level_values(1), axis=1).min().min()
        return [minima.y, maxima.y]

    # TODO - add method to figure out appropriate trimesh alpha value



if __name__ == "__main__":
    ec = SpatialComfort(
        simulation_directory=r"C:\Users\tgerrish\simulation\LadybugTools_ToolkitExternalThermalComfort",
        epw=r"C:\Users\tgerrish\BuroHappold\Sustainability and Physics - epws\GBR_London.Gatwick.037760_IWEC.epw",
    )
    print(ec.utci)
