from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.sunpath import Sunpath
from ladybug_comfort.collection.pmv import PMV
from ladybug_comfort.collection.utci import UTCI
from matplotlib.figure import Figure
from python_toolkit.plot.chart import timeseries_heatmap

from ...external_comfort import ExternalComfortResult
from ...ladybug_extension.datacollection import from_series, to_series
from ...ladybug_extension.location import to_string
from ..encoder import Encoder
from ..moisture import evaporative_cooling_effect_collection
from ..plot import plot_typology_day, plot_utci_heatmap_histogram
from ..plot.colormaps import (
    DBT_COLORMAP,
    MRT_COLORMAP,
    RH_COLORMAP,
    UTCI_BOUNDARYNORM,
    UTCI_COLORMAP,
    WS_COLORMAP,
)
from .typology import Typology


@dataclass(frozen=True)
class TypologyResult:
    typology: Typology = field(init=True, repr=True)
    external_comfort_result: ExternalComfortResult = field(init=True, repr=True)

    dry_bulb_temperature: HourlyContinuousCollection = field(init=False, repr=False)
    relative_humidity: HourlyContinuousCollection = field(init=False, repr=False)
    wind_speed: HourlyContinuousCollection = field(init=False, repr=False)
    mean_radiant_temperature: HourlyContinuousCollection = field(init=False, repr=False)
    ground_surface_temperature: HourlyContinuousCollection = field(
        init=False, repr=False
    )
    universal_thermal_climate_index: HourlyContinuousCollection = field(
        init=False, repr=False
    )
    standard_effective_temperature: HourlyContinuousCollection = field(
        init=False, repr=False
    )

    def __post_init__(self) -> TypologyResult:
        print(f"- Calculating {self}")
        dbt, rh = evaporative_cooling_effect_collection(
            self.external_comfort_result.external_comfort.epw,
            self.typology.evaporative_cooling_effectiveness,
        )
        object.__setattr__(self, "dry_bulb_temperature", dbt)
        object.__setattr__(self, "relative_humidity", rh)
        object.__setattr__(self, "wind_speed", self._wind_speed())
        object.__setattr__(
            self, "ground_surface_temperature", self._ground_surface_temperature()
        )
        object.__setattr__(
            self, "mean_radiant_temperature", self._mean_radiant_temperature()
        )
        object.__setattr__(
            self,
            "universal_thermal_climate_index",
            self._universal_thermal_climate_index(),
        )
        # object.__setattr__(
        #     self,
        #     "standard_effective_temperature",
        #     self._standard_effective_temperature(),
        # )

    def _wind_speed(self) -> HourlyContinuousCollection:
        if len(self.typology.shelters) == 0:
            return (
                self.external_comfort_result.external_comfort.epw.wind_speed
                * self.typology.wind_speed_multiplier
            )
        else:
            collections = []
            for shelter in self.typology.shelters:
                collections.append(
                    to_series(
                        shelter.effective_wind_speed(
                            self.external_comfort_result.external_comfort.epw
                        )
                        * self.typology.wind_speed_multiplier
                    )
                )
            return from_series(
                pd.concat(collections, axis=1).min(axis=1).rename("Wind Speed (m/s)")
            )

    def _ground_surface_temperature(self) -> HourlyContinuousCollection:
        """Calculate the ground surface temperature based on the external comfort result and typology set-up.

        Returns:
            HourlyContinuousCollection: An HourlyContinuousCollection of ground surface temperature.
        """

        sky_visible = self._sky_visibility()
        sky_blocked = 1 - sky_visible

        return (
            self.external_comfort_result.unshaded_below_temperature * sky_visible
        ) + (self.external_comfort_result.shaded_below_temperature * sky_blocked)

    def _sky_visibility(self) -> float:
        """Calculate the proportion of sky visible from a typology with any nuber of shelters.

        Returns:
            float: The proportion of sky visible from the typology.
        """
        unsheltered_proportion = 1
        sheltered_proportion = 0
        for shelter in self.typology.shelters:
            occ = shelter.sky_occluded
            unsheltered_proportion -= occ
            sheltered_proportion += occ * shelter.porosity
        return sheltered_proportion + unsheltered_proportion

    def _annual_hourly_sun_exposure(self) -> List[float]:
        """Return NaN if sun below horizon, and a value between 0-1 for sun-hidden to sun-exposed.

        Returns:
            List[float]: A list of sun visibility values for each hour of the year.
        """

        sunpath = Sunpath.from_location(
            self.external_comfort_result.external_comfort.epw.location
        )
        suns = [sunpath.calculate_sun_from_hoy(i) for i in range(8760)]
        sun_is_up = np.array([True if sun.altitude > 0 else False for sun in suns])

        nans = np.empty(
            len(self.external_comfort_result.external_comfort.epw.dry_bulb_temperature)
        )
        nans[:] = np.NaN

        if len(self.typology.shelters) == 0:
            return np.where(sun_is_up, 1, nans)

        blocked = []
        for shelter in self.typology.shelters:
            shelter_blocking = shelter.sun_blocked(suns)
            temp = np.where(shelter_blocking, shelter.porosity, nans)
            temp = np.where(np.logical_and(np.isnan(temp), sun_is_up), 1, temp)
            blocked.append(temp)

        return pd.DataFrame(blocked).T.min(axis=1).values.tolist()

    def _mean_radiant_temperature(self) -> HourlyContinuousCollection:
        """Return the effective mean radiant temperature for the given typology.

        Returns:
            HourlyContinuousCollection: An calculated mean radiant temperature based on the shelter configuration for the given typology.
        """

        shaded_mrt = to_series(
            self.external_comfort_result.shaded_mean_radiant_temperature
        )
        unshaded_mrt = to_series(
            self.external_comfort_result.unshaded_mean_radiant_temperature
        )

        sun_exposure = self._annual_hourly_sun_exposure()
        effective_sky_visibility = self._sky_visibility()
        daytime = np.array(
            [
                True if i > 0 else False
                for i in self.external_comfort_result.external_comfort.epw.global_horizontal_radiation
            ]
        )
        mrts = []
        for hour in range(8760):
            if daytime[hour]:
                mrts.append(
                    np.interp(
                        sun_exposure[hour],
                        [0, 1],
                        [shaded_mrt[hour], unshaded_mrt[hour]],
                    )
                )
            else:
                mrts.append(
                    np.interp(
                        effective_sky_visibility,
                        [0, 1],
                        [shaded_mrt[hour], unshaded_mrt[hour]],
                    )
                )

        # Fill any gaps where sun-visible/sun-occluded values are missing, and apply an exponentially weighted moving average to account for transition betwen shaded/unshaded periods.
        mrt_series = pd.Series(
            mrts, index=shaded_mrt.index, name=shaded_mrt.name
        ).interpolate()

        def __smoother(
            series: pd.Series,
            difference_threshold: float = -10,
            transition_window: int = 4,
            ewm_span: float = 1.25,
        ) -> pd.Series:
            """Helper function that adds a decay rate to a time-series for values dropping significantly below the previous values.

            Args:
                series (pd.Series): The series to modify
                difference_threshold (float, optional): The difference between current/prtevious values which class as a "transition". Defaults to -10.
                transition_window (int, optional): The number of values after the "transition" within which an exponentially weighted mean should be applied. Defaults to 4.
                ewm_span (float, optional): The rate of decay. Defaults to 1.25.

            Returns:
                pd.Series: A modified series
            """
            # Find periods of major transition (where values drop signifigantly from loss of radiation mainly)
            transition_index = series.diff() < difference_threshold

            # Get boolean index for all periods within window from the transition indices
            ewm_mask = []
            n = 0
            for i in transition_index:
                if i:
                    n = 0
                if n < transition_window:
                    ewm_mask.append(True)
                else:
                    ewm_mask.append(False)
                n += 1

            # Run an EWM to get the smoothed values following changes to values
            ewm_smoothed: pd.Series = series.ewm(span=ewm_span).mean()

            # Choose from ewm or original values based on ewm mask
            new_series = ewm_smoothed.where(ewm_mask, series)
            return new_series

        mrt_series = __smoother(
            mrt_series, difference_threshold=-10, transition_window=4, ewm_span=1.25
        )

        return from_series(mrt_series)

    def _universal_thermal_climate_index(self) -> HourlyContinuousCollection:
        """Return the effective UTCI for the given typology.

        Returns:
            HourlyContinuousCollection: The calculated UTCI based on the shelter configuration for the given typology.
        """
        utci_collection: HourlyContinuousCollection = UTCI(
            air_temperature=self.dry_bulb_temperature,
            rel_humidity=self.relative_humidity,
            rad_temperature=self.mean_radiant_temperature,
            wind_speed=self.wind_speed,
        ).universal_thermal_climate_index
        utci_collection.header.metadata["description"] = self.typology.name
        return utci_collection

    def _standard_effective_temperature(self) -> HourlyContinuousCollection:
        """Return the standard effective temperature for the given typology.

        Returns:
            HourlyContinuousCollection: The calculated standard effective temperature based on the shelter configuration for the given typology.
        """

        return PMV(
            air_temperature=self.dry_bulb_temperature,
            rel_humidity=self.relative_humidity,
            rad_temperature=self.mean_radiant_temperature,
            air_speed=self.wind_speed,
            met_rate=1.1,
            clo_value=0.7,
            external_work=0,
        ).standard_effective_temperature

    def to_dataframe(
        self, include_external_comfort_results: bool = True
    ) -> pd.DataFrame:
        """Create a dataframe from the typology results.

        Args:
            include_external_comfort_results (bool, optional): Whether to include the external comfort results in the dataframe. Defaults to True.

        Returns:
            pd.DataFrame: A dataframe containing the typology results.
        """

        attributes = [
            "dry_bulb_temperature",
            "relative_humidity",
            "wind_speed",
            "mean_radiant_temperature",
            "universal_thermal_climate_index",
        ]
        series: List[pd.Series] = []
        for attribute in attributes:
            series.append(to_series(getattr(self, attribute)))
        df = pd.concat(
            series,
            axis=1,
            keys=[f"{self.__class__.__name__} - {i}" for i in attributes],
        )

        if include_external_comfort_results:
            df = pd.concat([df, self.external_comfort_result.to_dataframe()], axis=1)

        return df

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        attributes = [
            "typology",
            "external_comfort_result",
            "dry_bulb_temperature",
            "relative_humidity",
            "wind_speed",
            "mean_radiant_temperature",
            "universal_thermal_climate_index",
            # "standard_effective_temperature",
        ]
        return {attribute: getattr(self, attribute) for attribute in attributes}

    def to_json(self, file_path: str) -> Path:
        """Write the content of this object to a JSON file

        Returns:
            Path: The path to the newly created JSON file.
        """

        file_path: Path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)

        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, cls=Encoder, indent=4)

        return file_path

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.typology.name})"

    def plot_utci_day(self, month: int = 6, day: int = 21) -> Figure:
        """Plot a single day UTCI and composite components

        Args:
            month (int, optional): The month to plot. Defaults to 6.
            day (int, optional): The day to plot. Defaults to 21.

        Returns:
            Figure: A figure showing UTCI and component parts for the given day.
        """
        return plot_typology_day(
            utci=to_series(self.universal_thermal_climate_index),
            dbt=to_series(self.dry_bulb_temperature),
            mrt=to_series(self.mean_radiant_temperature),
            rh=to_series(self.relative_humidity),
            ws=to_series(self.wind_speed),
            month=month,
            day=day,
            title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
        )

    def plot_utci_heatmap(self) -> Figure:
        """Create a heatmap showing the annual hourly UTCI values associated with this Typology.

        Returns:
            Figure: A matplotlib Figure object.
        """

        fig = timeseries_heatmap(
            series=to_series(self.universal_thermal_climate_index),
            cmap=UTCI_COLORMAP,
            norm=UTCI_BOUNDARYNORM,
            title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
        )

        return fig

    def plot_utci_histogram(self) -> Figure:
        """Create a histogram showing the annual hourly UTCI values associated with this Typology."""

        fig = plot_utci_heatmap_histogram(
            self.universal_thermal_climate_index,
            title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
        )

        return fig

    def plot_dbt_heatmap(self, vlims: List[float] = None) -> Figure:
        """Create a heatmap showing the annual hourly DBT values associated with this Typology.

        Args:
            vlims (List[float], optional): A list of two values to set the lower and upper limits of the colorbar. Defaults to None.

        Returns:
            Figure: A matplotlib Figure object.
        """

        fig = timeseries_heatmap(
            series=to_series(self.dry_bulb_temperature),
            cmap=DBT_COLORMAP,
            title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
            vlims=vlims,
        )

        return fig

    def plot_rh_heatmap(self, vlims: List[float] = None) -> Figure:
        """Create a heatmap showing the annual hourly RH values associated with this Typology.

        Args:
            vlims (List[float], optional): A list of two values to set the lower and upper limits of the colorbar. Defaults to None.

        Returns:
            Figure: A matplotlib Figure object.
        """

        fig = timeseries_heatmap(
            series=to_series(self.relative_humidity),
            cmap=RH_COLORMAP,
            title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
            vlims=vlims,
        )

        return fig

    def plot_ws_heatmap(self, vlims: List[float] = None) -> Figure:
        """Create a heatmap showing the annual hourly WS values associated with this Typology.

        Args:
            vlims (List[float], optional): A list of two values to set the lower and upper limits of the colorbar. Defaults to None.

        Returns:
            Figure: A matplotlib Figure object.
        """

        fig = timeseries_heatmap(
            series=to_series(self.wind_speed),
            cmap=WS_COLORMAP,
            title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
            vlims=vlims,
        )

        return fig

    def plot_mrt_heatmap(self, vlims: List[float] = None) -> Figure:
        """Create a heatmap showing the annual hourly MRT values associated with this Typology.

        Args:
            vlims (List[float], optional): A list of two values to set the lower and upper limits of the colorbar. Defaults to None.

        Returns:
            Figure: A matplotlib Figure object.
        """

        fig = timeseries_heatmap(
            series=to_series(self.mean_radiant_temperature),
            cmap=MRT_COLORMAP,
            title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
            vlims=vlims,
        )

        return fig
