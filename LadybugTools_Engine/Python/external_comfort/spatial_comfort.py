from __future__ import annotations

import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from pathlib import Path
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

        self.shaded_utci: HourlyContinuousCollection = None
        self.unshaded_utci: HourlyContinuousCollection = None
        self.shaded_ground_surface_temperature: HourlyContinuousCollection = None
        self.unshaded_ground_surface_temperature: HourlyContinuousCollection = None
        self.shaded_mean_radiant_temperature: HourlyContinuousCollection = None
        self.unshaded_mean_radiant_temperature: HourlyContinuousCollection = None

        self._run_openfield()

        self.total_irradiance: pd.DataFrame = None
        self.sky_view: pd.DataFrame = None
        self.points: pd.DataFrame = None

        self._load_simulation_results()

        self.utci = self._spatial_utci_approx()
        self.mrt = self._spatial_mrt_approx()
        self.ground_surface_temperature = self._spatial_gnd_srf_approx()

    def _load_simulation_results(self) -> None:
        """Load all simulation results from within the target simulation directory, and save these as compressed h5 files.
        """
        total_irradiance_path = self.simulation_directory / "total_irradiance.h5"
        sky_view_path = self.simulation_directory / "sky_view.h5"
        points_path = self.simulation_directory / "points.h5"

        if total_irradiance_path.exists():
            self.total_irradiance = pd.read_hdf(total_irradiance_path, "df")
        if sky_view_path.exists():
            self.sky_view = pd.read_hdf(sky_view_path, "df")
        if points_path.exists():
            self.points = pd.read_hdf(points_path, "df")

        if self.total_irradiance is None:
            self.total_irradiance = _make_annual(
                load_ill(self._find_irradiance_files())
            ).fillna(0)
            self.total_irradiance.to_hdf(
                total_irradiance_path, "df", complevel=9, complib="blosc"
            )

        if self.sky_view is None:
            self.sky_view = load_res(self._find_sky_view_files())
            self.sky_view.to_hdf(sky_view_path, "df", complevel=9, complib="blosc")

        if self.points is None:
            self.points = load_pts(self._find_points_files())
            self.points.to_hdf(points_path, "df", complevel=9, complib="blosc")

        return None

    def _run_openfield(self) -> None:
        """Run the Openfield process to generate the shaded and unshaded MRT, surface temperature and UTCI values."""

        shaded_utci_path = self.simulation_directory / "shaded_utci.json"
        unshaded_utci_path = self.simulation_directory / "unshaded_utci.json"

        shaded_ground_surface_temperature_path = (
            self.simulation_directory / "shaded_ground_surface_temperature.json"
        )
        unshaded_ground_surface_temperature_path = (
            self.simulation_directory / "unshaded_ground_surface_temperature.json"
        )

        shaded_mean_radiant_temperature_path = (
            self.simulation_directory / "shaded_mean_radiant_temperature.json"
        )
        unshaded_mean_radiant_temperature_path = (
            self.simulation_directory / "unshaded_mean_radiant_temperature.json"
        )

        if shaded_utci_path.exists():
            print(f"Loading {shaded_utci_path}")
            self.shaded_utci = from_json(shaded_utci_path)
        if unshaded_utci_path.exists():
            print(f"Loading {unshaded_utci_path}")
            self.unshaded_utci = from_json(unshaded_utci_path)
        if shaded_ground_surface_temperature_path.exists():
            print(f"Loading {shaded_ground_surface_temperature_path}")
            self.shaded_ground_surface_temperature = from_json(
                shaded_ground_surface_temperature_path
            )
        if unshaded_ground_surface_temperature_path.exists():
            print(f"Loading {unshaded_ground_surface_temperature_path}")
            self.unshaded_ground_surface_temperature = from_json(
                unshaded_ground_surface_temperature_path
            )
        if shaded_mean_radiant_temperature_path.exists():
            print(f"Loading {shaded_mean_radiant_temperature_path}")
            self.shaded_mean_radiant_temperature = from_json(
                shaded_mean_radiant_temperature_path
            )
        if unshaded_mean_radiant_temperature_path.exists():
            print(f"Loading {unshaded_mean_radiant_temperature_path}")
            self.unshaded_mean_radiant_temperature = from_json(
                unshaded_mean_radiant_temperature_path
            )

        if (
            (not self.shaded_utci)
            and (not self.unshaded_utci)
            and (not self.shaded_ground_surface_temperature)
            and (not self.unshaded_ground_surface_temperature)
            and (not self.shaded_mean_radiant_temperature)
            and (not self.unshaded_mean_radiant_temperature)
        ):
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

            self.shaded_utci = shaded_typology._universal_thermal_climate_index()
            self.unshaded_utci = unshaded_typology._universal_thermal_climate_index()

            self.shaded_ground_surface_temperature = openfield.shaded_below_temperature
            self.unshaded_ground_surface_temperature = (
                openfield.unshaded_below_temperature
            )

            self.shaded_mean_radiant_temperature = (
                openfield.shaded_mean_radiant_temperature
            )
            self.unshaded_mean_radiant_temperature = (
                openfield.unshaded_mean_radiant_temperature
            )

            to_json(self.shaded_utci, shaded_utci_path)
            to_json(self.unshaded_utci, unshaded_utci_path)
            to_json(
                self.shaded_ground_surface_temperature,
                shaded_ground_surface_temperature_path,
            )
            to_json(
                self.unshaded_ground_surface_temperature,
                unshaded_ground_surface_temperature_path,
            )
            to_json(
                self.shaded_mean_radiant_temperature,
                shaded_mean_radiant_temperature_path,
            )
            to_json(
                self.unshaded_mean_radiant_temperature,
                unshaded_mean_radiant_temperature_path,
            )

        return None

    def _find_points_files(self) -> List[Path]:
        """Find the points files in the simulation directory."""
        return list(
            (self.simulation_directory / "sky_view" / "model" / "grid").glob("*.pts")
        )

    def _find_irradiance_files(self) -> List[Path]:
        """Find the irradiance files in the simulation directory."""
        return list(
            (
                self.simulation_directory / "annual_irradiance" / "results" / "total"
            ).glob("*.ill")
        )

    def _find_sky_view_files(self) -> List[Path]:
        """Find the sky view files in the simulation directory."""
        return list((self.simulation_directory / "sky_view" / "results").glob("*.res"))

    def _spatial_utci_approx(self) -> pd.DataFrame:
        """Approximate the UTCI values for the full geometric simulation."""
        spatial_utci_path = self.simulation_directory / "utci.h5"
        if spatial_utci_path.exists():
            print(f"Loading {spatial_utci_path}")
            return pd.read_hdf(spatial_utci_path, "df")

        grp = self.total_irradiance.groupby(
            self.total_irradiance.columns.get_level_values(0), axis=1
        )

        _min_rad = grp.min().min(axis=1)
        _max_rad = grp.max().max(axis=1)

        shaded_utci = to_series(self.shaded_utci)
        unshaded_utci = to_series(self.unshaded_utci)

        xnew = self.total_irradiance.values
        x = np.stack([_min_rad.values, _max_rad.values], axis=1)

        y_day = np.stack([shaded_utci.values, unshaded_utci.values], axis=1)
        y_night = np.stack([unshaded_utci.values, shaded_utci.values], axis=1)

        daytime_vals = []
        nighttime_vals = []
        for i in range(8760):
            print(f"Spatial UTCI approx: [{i:04d}/8760]")
            daytime_vals.append(interpolate.interp1d(x[i], y_day[i])(xnew[i]))
            nighttime_vals.append(interpolate.interp1d(x[i], y_night[i])(xnew[i]))

        daytime = pd.DataFrame(
            daytime_vals,
            index=self.total_irradiance.index,
            columns=self.total_irradiance.columns,
        )
        nighttime = pd.DataFrame(
            nighttime_vals,
            index=self.total_irradiance.index,
            columns=self.total_irradiance.columns,
        )

        utci_approx = daytime.where(
            np.tile(self.sun_up_bool, [daytime.shape[1], 1]).T, nighttime
        )

        utci_approx.to_hdf(spatial_utci_path, "df", complevel=9, complib="blosc")

        return utci_approx

    def _spatial_mrt_approx(self) -> pd.DataFrame:
        """Approximate the mean radiant temperature values for the full geometric simulation."""
        spatial_mrt_path = self.simulation_directory / "mrt.h5"
        if spatial_mrt_path.exists():
            print(f"Loading {spatial_mrt_path}")
            return pd.read_hdf(spatial_mrt_path, "df")

        grp = self.total_irradiance.groupby(
            self.total_irradiance.columns.get_level_values(0), axis=1
        )

        _min_rad = grp.min().min(axis=1)
        _max_rad = grp.max().max(axis=1)

        shaded_mrt = to_series(self.shaded_mean_radiant_temperature)
        unshaded_mrt = to_series(self.unshaded_mean_radiant_temperature)

        xnew = self.total_irradiance.values
        x = np.stack([_min_rad.values, _max_rad.values], axis=1)

        y_day = np.stack([shaded_mrt.values, unshaded_mrt.values], axis=1)
        y_night = np.stack([unshaded_mrt.values, shaded_mrt.values], axis=1)

        daytime_vals = []
        nighttime_vals = []
        for i in range(8760):
            print(f"Spatial MRT approx: [{i:04d}/8760]")
            daytime_vals.append(interpolate.interp1d(x[i], y_day[i])(xnew[i]))
            nighttime_vals.append(interpolate.interp1d(x[i], y_night[i])(xnew[i]))

        daytime = pd.DataFrame(
            daytime_vals,
            index=self.total_irradiance.index,
            columns=self.total_irradiance.columns,
        )
        nighttime = pd.DataFrame(
            nighttime_vals,
            index=self.total_irradiance.index,
            columns=self.total_irradiance.columns,
        )

        mrt_approx = daytime.where(
            np.tile(self.sun_up_bool, [daytime.shape[1], 1]).T, nighttime
        )

        mrt_approx.to_hdf(spatial_mrt_path, "df", complevel=9, complib="blosc")

        return mrt_approx

    def _spatial_gnd_srf_approx(self) -> pd.DataFrame:
        """Approximate the ground surface temperature values for the full geometric simulation."""
        spatial_gnd_srf_path = self.simulation_directory / "gnd_srf.h5"
        if spatial_gnd_srf_path.exists():
            print(f"Loading {spatial_gnd_srf_path}")
            return pd.read_hdf(spatial_gnd_srf_path, "df")

        grp = self.total_irradiance.groupby(
            self.total_irradiance.columns.get_level_values(0), axis=1
        )

        _min_rad = grp.min().min(axis=1)
        _max_rad = grp.max().max(axis=1)

        shaded_gnd_srf = to_series(self.shaded_ground_surface_temperature)
        unshaded_gnd_srf = to_series(self.unshaded_ground_surface_temperature)

        xnew = self.total_irradiance.values
        x = np.stack([_min_rad.values, _max_rad.values], axis=1)

        y_day = np.stack([shaded_gnd_srf.values, unshaded_gnd_srf.values], axis=1)
        y_night = np.stack([unshaded_gnd_srf.values, shaded_gnd_srf.values], axis=1)

        daytime_vals = []
        nighttime_vals = []
        for i in range(8760):
            print(f"Spatial Ground Srf Temp approx: [{i:04d}/8760]")
            daytime_vals.append(interpolate.interp1d(x[i], y_day[i])(xnew[i]))
            nighttime_vals.append(interpolate.interp1d(x[i], y_night[i])(xnew[i]))

        daytime = pd.DataFrame(
            daytime_vals,
            index=self.total_irradiance.index,
            columns=self.total_irradiance.columns,
        )
        nighttime = pd.DataFrame(
            nighttime_vals,
            index=self.total_irradiance.index,
            columns=self.total_irradiance.columns,
        )

        gnd_srf_approx = daytime.where(
            np.tile(self.sun_up_bool, [daytime.shape[1], 1]).T, nighttime
        )

        gnd_srf_approx.to_hdf(spatial_gnd_srf_path, "df", complevel=9, complib="blosc")

        return gnd_srf_approx

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
        simulation_directory=r"C:\Users\tgerrish\simulation\RootBridges",
        epw=r"C:\Users\tgerrish\BuroHappold\Sustainability and Physics - epws\GBR_LONDON-HEATHROW-AP_037720_IW2.epw",
    )
    print(ec.utci)
