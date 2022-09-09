from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybugtools_toolkit.external_comfort.simulate.simulation_result import (
    SimulationResult,
)
from ladybugtools_toolkit.external_comfort.thermal_comfort.utci.utci import utci
from ladybugtools_toolkit.external_comfort.typology.typology import Typology
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.ladybug_extension.epw.filename import filename
from ladybugtools_toolkit.ladybug_extension.epw.to_dataframe import to_dataframe
from ladybugtools_toolkit.ladybug_extension.location.to_string import (
    to_string as location_to_string,
)
from ladybugtools_toolkit.plot.colormaps import (
    DBT_COLORMAP,
    MRT_COLORMAP,
    RH_COLORMAP,
    WS_COLORMAP,
)
from ladybugtools_toolkit.plot.timeseries_heatmap import timeseries_heatmap
from ladybugtools_toolkit.plot.utci_day_comfort_metrics import utci_day_comfort_metrics
from ladybugtools_toolkit.plot.utci_distance_to_comfortable import (
    utci_distance_to_comfortable,
)
from ladybugtools_toolkit.plot.utci_heatmap import utci_heatmap
from ladybugtools_toolkit.plot.utci_heatmap_histogram import utci_heatmap_histogram
from matplotlib.figure import Figure


class ExternalComfort:
    """An object containing all inputs and results of an external MRT simulation and resultant
        thermal comfort metrics.

    Args:
        simulation_result (SimulationResult): A set of simulation results.
        typology (Typology): A typology object.

    Returns:
        ExternalComfort: An object containing all inputs and results of an external MRT simulation
        and resultant thermal comfort metrics.
    """

    def __init__(
        self, simulation_result: SimulationResult, typology: Typology
    ) -> ExternalComfort:

        self.simulation_result = simulation_result
        self.typology = typology

        # calculate inputs to thermal comfort calculations
        self.dry_bulb_temperature = self.typology.dry_bulb_temperature(
            self.simulation_result.epw
        )
        self.relative_humidity = self.typology.relative_humidity(
            self.simulation_result.epw
        )
        self.wind_speed = self.typology.wind_speed(self.simulation_result.epw)
        self.mean_radiant_temperature = self.typology.mean_radiant_temperature(
            self.simulation_result
        )

        # calculate UTCI
        self.universal_thermal_climate_index = utci(
            self.dry_bulb_temperature,
            self.relative_humidity,
            self.mean_radiant_temperature,
            self.wind_speed,
        )

        # add typology descriptions to collection metadata
        if typology.sky_exposure() != 1:
            typology_description = f"{self.typology.name} ({self.simulation_result.ground_material.identifier} ground and {self.simulation_result.shade_material.identifier} shade)"
        else:
            typology_description = f"{self.typology.name} ({self.simulation_result.ground_material.identifier} ground)"
        for attr in [
            "dry_bulb_temperature",
            "relative_humidity",
            "wind_speed",
            "mean_radiant_temperature",
            "universal_thermal_climate_index",
        ]:
            old_metadata = getattr(self, attr).header.metadata
            new_metadata = {
                **old_metadata,
                **{"typology": typology_description},
            }
            setattr(getattr(self, attr).header, "metadata", new_metadata)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.simulation_result.identifier} - {self.typology.name})"

    def to_dict(self) -> Dict[str, HourlyContinuousCollection]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        variables = [
            "universal_thermal_climate_index",
            "dry_bulb_temperature",
            "relative_humidity",
            "mean_radiant_temperature",
            "wind_speed",
        ]

        return {
            **{var: getattr(self, var) for var in variables},
            **self.typology.to_dict(),
            **self.simulation_result.to_dict(),
        }

    def to_dataframe(
        self, include_epw: bool = False, include_simulation_results: bool = False
    ) -> pd.DataFrame:
        """Create a Pandas DataFrame from this object.

        Args:
            include_epw (bool, optional): Set to True to include the dataframe for the EPW file
                also.
            include_simulation_results(bool, optional): Set to True to include the dataframe for
                the simulation results also.

        Returns:
            pd.DataFrame: A Pandas DataFrame with this objects properties.
        """

        df = pd.DataFrame()

        if include_simulation_results:
            df = pd.concat(
                [df, self.simulation_result.to_dataframe(include_epw)], axis=1
            )
        elif include_epw:
            df = pd.concat([df, to_dataframe(self.simulation_result.epw)], axis=1)

        variables = [
            "universal_thermal_climate_index",
            "dry_bulb_temperature",
            "relative_humidity",
            "mean_radiant_temperature",
            "wind_speed",
        ]
        obj_series = []
        for var in variables:
            _ = to_series(getattr(self, var))
            obj_series.append(
                _.rename(
                    (
                        f"{filename(self.simulation_result.epw)}",
                        f"{var} - {self.simulation_result.ground_material.display_name} ground, {self.simulation_result.shade_material.display_name} shade - {self.typology.name}",
                    )
                )
            )
        df = pd.concat([df, pd.concat(obj_series, axis=1)], axis=1)

        return df

    @property
    def plot_title_string(self) -> str:
        """Return the description of this result suitable for use in plotting titles."""
        return f"{location_to_string(self.simulation_result.epw.location)}\n{self.simulation_result.ground_material.display_name} ground, {self.simulation_result.shade_material.display_name} shade\n{self.typology.name}"

    def plot_utci_day_comfort_metrics(self, month: int = 3, day: int = 21) -> Figure:
        """Plot a single day UTCI and composite components

        Args:
            month (int, optional): The month to plot. Defaults to 3.
            day (int, optional): The day to plot. Defaults to 21.

        Returns:
            Figure: A figure showing UTCI and component parts for the given day.
        """

        return utci_day_comfort_metrics(
            to_series(self.universal_thermal_climate_index),
            to_series(self.dry_bulb_temperature),
            to_series(self.mean_radiant_temperature),
            to_series(self.relative_humidity),
            to_series(self.wind_speed),
            month,
            day,
            self.plot_title_string,
        )

    def plot_utci_heatmap(self) -> Figure:
        """Create a heatmap showing the annual hourly UTCI values associated with this Typology.

        Returns:
            Figure: A matplotlib Figure object.
        """

        return utci_heatmap(
            self.universal_thermal_climate_index,
            self.plot_title_string,
        )

    def plot_utci_heatmap_histogram(self) -> Figure:
        """Create a heatmap showing the annual hourly UTCI values associated with this Typology.

        Returns:
            Figure: A matplotlib Figure object.
        """

        return utci_heatmap_histogram(
            self.universal_thermal_climate_index,
            self.plot_title_string,
        )

    def plot_utci_distance_to_comfortable(
        self,
        comfort_thresholds: Tuple[float] = (9, 26),
        low_limit: float = 15,
        high_limit: float = 25,
    ) -> Figure:
        """Create a heatmap showing the "distance" in C from the "no thermal stress" UTCI comfort
            band.

        Args:
            comfort_thresholds (List[float], optional): The comfortable band of UTCI temperatures.
                Defaults to [9, 26].
            low_limit (float, optional): The distance from the lower edge of the comfort threshold
                to include in the "too cold" part of the heatmap. Defaults to 15.
            high_limit (float, optional): The distance from the upper edge of the comfort threshold
                to include in the "too hot" part of the heatmap. Defaults to 25.

        Returns:
            Figure: A matplotlib Figure object.
        """

        return utci_distance_to_comfortable(
            collection=self.universal_thermal_climate_index,
            title=self.plot_title_string,
            comfort_thresholds=comfort_thresholds,
            low_limit=low_limit,
            high_limit=high_limit,
        )

    def plot_dbt_heatmap(self, vlims: Tuple[float] = None) -> Figure:
        """Create a heatmap showing the annual hourly DBT values associated with this Typology.

        Args:
            vlims (Tuple[float], optional): A list of two values to set the lower and upper limits of the colorbar. Defaults to None.

        Returns:
            Figure: A matplotlib Figure object.
        """

        fig = timeseries_heatmap(
            series=to_series(self.dry_bulb_temperature),
            cmap=DBT_COLORMAP,
            title=self.plot_title_string,
            vlims=vlims,
        )

        return fig

    def plot_rh_heatmap(self, vlims: Tuple[float] = None) -> Figure:
        """Create a heatmap showing the annual hourly RH values associated with this Typology.

        Args:
            vlims (Tuple[float], optional): A list of two values to set the lower and upper limits of the colorbar. Defaults to None.

        Returns:
            Figure: A matplotlib Figure object.
        """

        fig = timeseries_heatmap(
            series=to_series(self.relative_humidity),
            cmap=RH_COLORMAP,
            title=self.plot_title_string,
            vlims=vlims,
        )

        return fig

    def plot_ws_heatmap(self, vlims: Tuple[float] = None) -> Figure:
        """Create a heatmap showing the annual hourly WS values associated with this Typology.

        Args:
            vlims (Tuple[float], optional): A list of two values to set the lower and upper limits of the colorbar. Defaults to None.

        Returns:
            Figure: A matplotlib Figure object.
        """

        fig = timeseries_heatmap(
            series=to_series(self.wind_speed),
            cmap=WS_COLORMAP,
            title=self.plot_title_string,
            vlims=vlims,
        )

        return fig

    def plot_mrt_heatmap(self, vlims: Tuple[float] = None) -> Figure:
        """Create a heatmap showing the annual hourly MRT values associated with this Typology.

        Args:
            vlims (Tuple[float], optional): A list of two values to set the lower and upper limits of the colorbar. Defaults to None.

        Returns:
            Figure: A matplotlib Figure object.
        """

        fig = timeseries_heatmap(
            series=to_series(self.mean_radiant_temperature),
            cmap=MRT_COLORMAP,
            title=self.plot_title_string,
            vlims=vlims,
        )

        return fig
