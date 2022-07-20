from __future__ import annotations

from pathlib import Path

import pandas as pd
from cached_property import cached_property
from LadybugTools_Engine.Python.src.ladybugtools_toolkit.external_comfort.spatial.load_universal_thermal_climate_index_interpolated import (
    load_universal_thermal_climate_index_interpolated,
)
from ladybugtools_toolkit.external_comfort.external_comfort import ExternalComfort
from ladybugtools_toolkit.external_comfort.spatial.load_diffuse_irradiance import (
    load_diffuse_irradiance,
)
from ladybugtools_toolkit.external_comfort.spatial.load_direct_irradiance import (
    load_direct_irradiance,
)
from ladybugtools_toolkit.external_comfort.spatial.load_mean_radiant_temperature_interpolated import (
    load_mean_radiant_temperature_interpolated,
)
from ladybugtools_toolkit.external_comfort.spatial.load_points import load_points
from ladybugtools_toolkit.external_comfort.spatial.load_sky_view import load_sky_view
from ladybugtools_toolkit.external_comfort.spatial.load_total_irradiance import (
    load_total_irradiance,
)
from ladybugtools_toolkit.external_comfort.spatial.proximity_decay import (
    proximity_decay,
)
from ladybugtools_toolkit.external_comfort.spatial.spatial_comfort_possible import (
    spatial_comfort_possible,
)
from ladybugtools_toolkit.external_comfort.thermal_comfort.universal_thermal_climate_index import (
    universal_thermal_climate_index,
)


class SpatialComfort:
    def __init__(
        self, simulation_directory: Path, external_comfort: ExternalComfort
    ) -> SpatialComfort:
        """A SpatialComfort object, used to calculate spatial UTCI.

        Args:
            simulation_directory (Path): A directory path containing SkyView and Annual
                Irradiance simulation results.
            external_comfort_result (ExternalComfortResult): A results object containing
                pre-simulated MRT values.

        Returns:
            SpatialComfort: A SpatialComfort object.
        """

        self.simulation_directory = Path(simulation_directory)
        self.external_comfort = external_comfort

        # check that spatial-comfort is possible for given simulation_directory
        spatial_comfort_possible(self.simulation_directory)

        # calculate baseline UTCI for shaded/unshaded
        self._unshaded_utci = universal_thermal_climate_index(
            self.external_comfort.simulation_result.epw.dry_bulb_temperature,
            self.external_comfort.simulation_result.epw.relative_humidity,
            self.external_comfort.simulation_result.unshaded_mean_radiant_temperature,
            self.external_comfort.simulation_result.epw.wind_speed,
        )
        self._shaded_utci = universal_thermal_climate_index(
            self.external_comfort.simulation_result.epw.dry_bulb_temperature,
            self.external_comfort.simulation_result.epw.relative_humidity,
            self.external_comfort.simulation_result.shaded_mean_radiant_temperature,
            self.external_comfort.simulation_result.epw.wind_speed,
        )

    @cached_property
    def points(self) -> pd.DataFrame:
        return load_points(self.simulation_directory)

    @cached_property
    def total_irradiance(self) -> pd.DataFrame:
        return load_total_irradiance(self.simulation_directory)

    @cached_property
    def direct_irradiance(self) -> pd.DataFrame:
        return load_direct_irradiance(self.simulation_directory)

    @cached_property
    def diffuse_irradiance(self) -> pd.DataFrame:
        return load_diffuse_irradiance(
            self.simulation_directory,
            self.total_irradiance,
            self.direct_irradiance,
        )

    @cached_property
    def sky_view(self) -> pd.Series:
        return load_sky_view(self.simulation_directory)

    @cached_property
    def universal_thermal_climate_index_interpolated(self) -> pd.DataFrame:

        return load_universal_thermal_climate_index_interpolated(
            self.simulation_directory,
            self._unshaded_utci,
            self._shaded_utci,
            self.total_irradiance,
            self.sky_view,
            self.external_comfort.simulation_result.epw,
        )

    @cached_property
    def mean_radiant_temperature_interpolated(self) -> pd.DataFrame:
        return load_mean_radiant_temperature_interpolated(
            self.simulation_directory,
            self.external_comfort.simulation_result.unshaded_mean_radiant_temperature,
            self.external_comfort.simulation_result.shaded_mean_radiant_temperature,
            self.total_irradiance,
            self.sky_view,
            self.external_comfort.simulation_result.epw,
        )
