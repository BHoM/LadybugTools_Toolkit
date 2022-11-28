from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from matplotlib.figure import Figure

from ..bhomutil.bhom_object import BHoMObject, bhom_dict_to_dict, pascalcase
from ..ladybug_extension.datacollection import to_series
from ..ladybug_extension.epw import to_dataframe
from ..ladybug_extension.location import to_string as location_to_string
from ..plot.colormaps import DBT_COLORMAP, MRT_COLORMAP, RH_COLORMAP, WS_COLORMAP
from ..plot.timeseries_heatmap import timeseries_heatmap
from ..plot.utci_day_comfort_metrics import utci_day_comfort_metrics
from ..plot.utci_distance_to_comfortable import utci_distance_to_comfortable
from ..plot.utci_heatmap import utci_heatmap
from ..plot.utci_heatmap_histogram import utci_heatmap_histogram
from .simulate import SimulationResult
from .typology import Typology


@dataclass(init=True, repr=True, eq=True)
class ExternalComfort(BHoMObject):
    """An object containing all inputs and results of an external MRT
        simulation and resultant thermal comfort metrics.

    Args:
        simulation_result (SimulationResult):
            A set of pre-run simulation results.
        typology (Typology):
            A typology object.

    Returns:
        ExternalComfort: An object containing all inputs and results of an external MRT simulation
        and resultant thermal comfort metrics.
    """

    simulation_result: SimulationResult = field(init=True, compare=True, repr=True)
    typology: Typology = field(init=True, compare=True, repr=True)

    dry_bulb_temperature: HourlyContinuousCollection = field(
        init=True, compare=True, repr=False, default=None
    )
    relative_humidity: HourlyContinuousCollection = field(
        init=True, compare=True, repr=False, default=None
    )
    wind_speed: HourlyContinuousCollection = field(
        init=True, compare=True, repr=False, default=None
    )
    mean_radiant_temperature: HourlyContinuousCollection = field(
        init=True, compare=True, repr=False, default=None
    )
    universal_thermal_climate_index: HourlyContinuousCollection = field(
        init=True, compare=True, repr=False, default=None
    )

    _t: str = field(
        init=False,
        compare=True,
        repr=False,
        default="BH.oM.LadybugTools.ExternalComfort",
    )

    def __post_init__(self):

        if not self.simulation_result.is_run():
            self.simulation_result = self.simulation_result.run()

        # calculate metrics
        epw = self.simulation_result.epw

        self.dry_bulb_temperature = (
            self.dry_bulb_temperature
            if isinstance(
                getattr(self, "dry_bulb_temperature"), HourlyContinuousCollection
            )
            else self.typology.dry_bulb_temperature(epw)
        )
        self.relative_humidity = (
            self.relative_humidity
            if isinstance(
                getattr(self, "relative_humidity"), HourlyContinuousCollection
            )
            else self.typology.relative_humidity(epw)
        )
        self.wind_speed = (
            self.wind_speed
            if isinstance(getattr(self, "wind_speed"), HourlyContinuousCollection)
            else self.typology.wind_speed(epw)
        )
        self.mean_radiant_temperature = (
            self.mean_radiant_temperature
            if isinstance(
                getattr(self, "mean_radiant_temperature"), HourlyContinuousCollection
            )
            else self.typology.mean_radiant_temperature(self.simulation_result)
        )
        self.universal_thermal_climate_index = (
            self.universal_thermal_climate_index
            if isinstance(
                getattr(self, "universal_thermal_climate_index"),
                HourlyContinuousCollection,
            )
            else self.typology.universal_thermal_climate_index(self.simulation_result)
        )

        # populate metadata in metrics with current ExternalComfort config
        if self.typology.sky_exposure() != 1:
            typology_description = f"{self.typology.name} ({self.simulation_result.ground_material.to_lbt().identifier} ground and {self.simulation_result.shade_material.to_lbt().identifier} shade)"
        else:
            typology_description = f"{self.typology.name} ({self.simulation_result.ground_material.to_lbt().identifier} ground)"
        for attr in [
            "dry_bulb_temperature",
            "relative_humidity",
            "wind_speed",
            "mean_radiant_temperature",
            "universal_thermal_climate_index",
        ]:
            obj = getattr(self, attr)
            if isinstance(obj, HourlyContinuousCollection):
                old_metadata = obj.header.metadata
                new_metadata = {
                    **old_metadata,
                    **{"typology": typology_description},
                }
                setattr(obj.header, "metadata", new_metadata)

        # wrap methods within this class
        super().__post_init__()

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> SimulationResult:
        """Create this object from a dictionary."""

        sanitised_dict = bhom_dict_to_dict(dictionary)
        sanitised_dict.pop("_t", None)

        # handle object conversions
        if isinstance(sanitised_dict["simulation_result"], dict):
            sanitised_dict["simulation_result"] = SimulationResult.from_dict(
                sanitised_dict["simulation_result"]
            )
        if isinstance(sanitised_dict["typology"], dict):
            sanitised_dict["typology"] = Typology.from_dict(sanitised_dict["typology"])

        for calculated_result in [
            "dry_bulb_temperature",
            "relative_humidity",
            "wind_speed",
            "mean_radiant_temperature",
            "universal_thermal_climate_index",
        ]:
            if isinstance(sanitised_dict[calculated_result], dict):
                if "type" in sanitised_dict[calculated_result].keys():
                    sanitised_dict[
                        calculated_result
                    ] = HourlyContinuousCollection.from_dict(
                        sanitised_dict[calculated_result]
                    )
            else:
                sanitised_dict[calculated_result] = None

        return cls(
            simulation_result=sanitised_dict["simulation_result"],
            typology=sanitised_dict["typology"],
            dry_bulb_temperature=sanitised_dict["dry_bulb_temperature"],
            relative_humidity=sanitised_dict["relative_humidity"],
            wind_speed=sanitised_dict["wind_speed"],
            mean_radiant_temperature=sanitised_dict["mean_radiant_temperature"],
            universal_thermal_climate_index=sanitised_dict[
                "universal_thermal_climate_index"
            ],
        )

    @classmethod
    def from_json(cls, json_string: str) -> SimulationResult:
        """Create this object from a JSON string."""

        dictionary = json.loads(json_string)

        return cls.from_dict(dictionary)

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
        dfs = []

        if include_epw:
            dfs.append(to_dataframe(self.simulation_result.epw))

        if include_simulation_results:
            dfs.append(self.simulation_result.to_dataframe())

        variables = [
            "universal_thermal_climate_index",
            "dry_bulb_temperature",
            "relative_humidity",
            "mean_radiant_temperature",
            "wind_speed",
        ]
        for var in variables:
            s = to_series(getattr(self, var))
            s.rename(
                (
                    f"{self.simulation_result.identifier} - {self.typology.name}",
                    pascalcase(var),
                    s.name,
                ),
                inplace=True,
            )
            dfs.append(s)

        return pd.concat(dfs, axis=1)

    @property
    def plot_title_string(self) -> str:
        """Return the description of this result suitable for use in plotting titles."""
        return f"{location_to_string(self.simulation_result.epw.location)}\n{self.simulation_result.ground_material.to_lbt().display_name} ground, {self.simulation_result.shade_material.to_lbt().display_name} shade\n{self.typology.name}"

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
