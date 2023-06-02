from __future__ import annotations

import calendar
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ladybug.epw import HourlyContinuousCollection
from ladybug_comfort.collection.utci import UTCI
from matplotlib.figure import Figure

from ..bhomutil.analytics import CONSOLE_LOGGER
from ..bhomutil.bhom_object import BHoMObject, bhom_dict_to_dict, pascalcase
from ..external_comfort.utci import categorise, utci_comfort_categories
from ..helpers import evaporative_cooling_effect
from ..ladybug_extension.analysis_period import (
    AnalysisPeriod,
    analysis_period_to_boolean,
)
from ..ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)
from ..ladybug_extension.epw import epw_to_dataframe
from ..ladybug_extension.location import location_to_string
from ..plot import (
    DBT_COLORMAP,
    MRT_COLORMAP,
    RH_COLORMAP,
    WS_COLORMAP,
    heatmap,
    utci_heatmap,
)
from ..plot.utci_day_comfort_metrics import utci_day_comfort_metrics
from ..plot.utci_distance_to_comfortable import utci_distance_to_comfortable
from ..plot.utci_heatmap_histogram import utci_heatmap_histogram
from .shelter import Shelter
from .simulate import SimulationResult
from .typology import Typology
from .utci import utci


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
        if isinstance(
            getattr(self, "universal_thermal_climate_index"), HourlyContinuousCollection
        ):
            pass
        elif all(
            [
                isinstance(
                    getattr(self, "dry_bulb_temperature"), HourlyContinuousCollection
                ),
                isinstance(
                    getattr(self, "relative_humidity"), HourlyContinuousCollection
                ),
                isinstance(getattr(self, "wind_speed"), HourlyContinuousCollection),
                isinstance(
                    getattr(self, "mean_radiant_temperature"),
                    HourlyContinuousCollection,
                ),
            ]
        ):
            CONSOLE_LOGGER.info(f"[{self.typology.name}] - Calculating UTCI")
            self.universal_thermal_climate_index = utci(
                air_temperature=self.dry_bulb_temperature,
                relative_humidity=self.relative_humidity,
                mean_radiant_temperature=self.mean_radiant_temperature,
                wind_speed=self.wind_speed,
            )
        else:
            self.universal_thermal_climate_index = (
                self.typology.universal_thermal_climate_index(self.simulation_result)
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
        self,
        include_epw: bool = False,
        include_simulation_results: bool = False,
        include_epw_additional: bool = False,
    ) -> pd.DataFrame:
        """Create a Pandas DataFrame from this object.

        Args:
            include_epw (bool, optional): Set to True to include the dataframe for the EPW file
                also.
            include_simulation_results (bool, optional): Set to True to include the dataframe for
                the simulation results also.
            include_epw_additional (bool, optional): Set to True to also include calculated
                values such as sun position along with EPW. Only includes if include_epw is
                True also.

        Returns:
            pd.DataFrame: A Pandas DataFrame with this objects properties.
        """
        dfs = []

        if include_epw:
            dfs.append(
                epw_to_dataframe(self.simulation_result.epw, include_epw_additional)
            )

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
            s = collection_to_series(getattr(self, var))
            s.rename(
                (
                    f"{self.simulation_result.identifier} - {self.typology.name}",
                    pascalcase(var),
                    s.name,
                ),
                inplace=True,
            )
            dfs.append(s)

        df = pd.concat(dfs, axis=1)

        return df

    def feasible_utci_limits(
        self, include_additional_moisture: bool = True, as_dataframe: bool = False
    ) -> Union[pd.DataFrame, List[HourlyContinuousCollection]]:
        """Calculate the absolute min/max collections of UTCI based on possible shade, wind and moisture adjustments to a pre-computed ExternalComfort condition.

        Args:
            include_additional_moisture (bool):
                Include the effect of evaporative cooling on the UTCI limits.
            as_dataframe (bool):
                Return the output as a dataframe with two columns, instread of two separate collections.

        Returns:
            List[HourlyContinuousCollection]: The lowest UTCI and highest UTCI temperatures for each hour of the year.
        """

        epw = self.simulation_result.epw

        dbt_evap, rh_evap = np.array(
            [
                evaporative_cooling_effect(
                    dry_bulb_temperature=_dbt,
                    relative_humidity=_rh,
                    evaporative_cooling_effectiveness=0.5,
                    atmospheric_pressure=_atm,
                )
                for _dbt, _rh, _atm in list(
                    zip(
                        *[
                            epw.dry_bulb_temperature,
                            epw.relative_humidity,
                            epw.atmospheric_station_pressure,
                        ]
                    )
                )
            ]
        ).T
        dbt_evap = epw.dry_bulb_temperature.get_aligned_collection(dbt_evap)
        rh_evap = epw.relative_humidity.get_aligned_collection(rh_evap)

        dbt_rh_options = (
            [[dbt_evap, rh_evap], [epw.dry_bulb_temperature, epw.relative_humidity]]
            if include_additional_moisture
            else [[epw.dry_bulb_temperature, epw.relative_humidity]]
        )

        utcis = []
        for _dbt, _rh in dbt_rh_options:
            for _ws in [
                self.wind_speed,
                self.wind_speed.get_aligned_collection(0),
                self.wind_speed * 1.1,
            ]:
                for _mrt in [
                    self.dry_bulb_temperature,
                    self.mean_radiant_temperature,
                ]:
                    utcis.append(
                        UTCI(
                            air_temperature=_dbt,
                            rad_temperature=_mrt,
                            rel_humidity=_rh,
                            wind_speed=_ws,
                        ).universal_thermal_climate_index,
                    )
        df = pd.concat([collection_to_series(i) for i in utcis], axis=1)
        min_utci = collection_from_series(
            df.min(axis=1).rename("Universal Thermal Climate Index (C)")
        )
        max_utci = collection_from_series(
            df.max(axis=1).rename("Universal Thermal Climate Index (C)")
        )

        if as_dataframe:
            return pd.concat(
                [
                    collection_to_series(min_utci),
                    collection_to_series(max_utci),
                ],
                axis=1,
                keys=["lowest", "highest"],
            )

        return min_utci, max_utci

    def feasible_comfort_category(
        self,
        include_additional_moisture: bool = True,
        analysis_periods: Tuple[AnalysisPeriod] = (AnalysisPeriod()),
        simplified: bool = False,
        comfort_limits: Tuple = (9, 26),
        density: bool = True,
    ) -> pd.DataFrame:
        """Calculate the feasible comfort categories for each hour of the year.

        Args:
            include_additional_moisture (bool):
                Include the effect of evaporative cooling on the UTCI limits.
            analysis_periods (Tuple[AnalysisPeriod]):
                A tuple of analysis periods to filter the results by.
            simplified (bool):
                Set to True to use the simplified comfort categories.
            comfort_limits (Tuple):
                A tuple of the lower and upper comfort limits.
            density (bool):
                Set to True to return the density of the comfort category.

        Returns:
            pd.DataFrame: A dataframe with the comfort categories for each hour of the year.
        """

        try:
            iter(analysis_periods)
            if not all(isinstance(ap, AnalysisPeriod) for ap in analysis_periods):
                raise TypeError(
                    "analysis_periods must be an iterable of AnalysisPeriods"
                )
        except TypeError as exc:
            raise TypeError(
                "analysis_periods must be an iterable of AnalysisPeriods"
            ) from exc

        for ap in analysis_periods:
            if (ap.st_month != 1) or (ap.end_month != 12):
                raise ValueError("Analysis periods must be for the whole year.")

        _df = self.feasible_utci_limits(
            as_dataframe=True, include_additional_moisture=include_additional_moisture
        )

        # filter by hours
        hours = analysis_period_to_boolean(analysis_periods)
        _df_filtered = _df.loc[hours]

        cats, _ = utci_comfort_categories(
            simplified=simplified,
            comfort_limits=comfort_limits,
        )

        # categorise
        _df_cat = categorise(
            _df_filtered, simplified=simplified, comfort_limits=comfort_limits
        )

        # join categories and get low/high lims
        temp = pd.concat(
            [
                _df_cat.groupby(_df_cat.index.month)
                .lowest.value_counts(normalize=density)
                .unstack()
                .reindex(cats, axis=1)
                .fillna(0),
                _df_cat.groupby(_df_cat.index.month)
                .highest.value_counts(normalize=density)
                .unstack()
                .reindex(cats, axis=1)
                .fillna(0),
            ],
            axis=1,
        )
        columns = pd.MultiIndex.from_product([cats, ["lowest", "highest"]])
        temp = pd.concat(
            [
                temp.groupby(temp.columns, axis=1).min(),
                temp.groupby(temp.columns, axis=1).max(),
            ],
            axis=1,
            keys=["lowest", "highest"],
        ).reorder_levels(order=[1, 0], axis=1)[columns]
        temp.index = [calendar.month_abbr[i] for i in temp.index]

        return temp

    def insitu_comfort_addmeasures_hotclimate(
        self,
        add_overhead_shelter: bool = True,
        add_additional_air_movement: bool = True,
        add_misting: bool = True,
        add_radiant_cooling: bool = True,
        evaporative_cooling_effectiveness: Union[float, Tuple[float]] = 0.7,
        wind_speed_multiplier: Union[float, Tuple[float]] = 1,
        increase_shelter_wind_porosity: bool = True,
        adjusted_shelter_wind_porosity: float = 0.75,
        radiant_temperature_adjustment: Union[float, Tuple[float]] = 0,
    ) -> ExternalComfort:
        """Apply varying levels of additional measures to the insitu comfort model, taking into account any existing measures that are in place already.

        Args:
            add_overhead_shelter (bool, optional):
                Add an overhead shelter to this object. Defaults to True.
            add_additional_air_movement (bool, optional):
                Add additional air movement. Vary the amount using the wind_speed_multiplier and adjusted_shelter_wind_porosity inputs. Defaults to True.
            add_misting (bool, optional):
                Add moisture to the air. Defaults to True.
            add_radiant_cooling (bool, optional):
                Include some adjustment to the MRT. Defaults to True.
            evaporative_cooling_effectiveness (float, optional):
                Set the effectivess of the evaporative cooling. Defaults to 0.7.
            wind_speed_multiplier (float, optional):
                Increase wind speed. Defaults to 1.
            increase_shelter_wind_porosity (bool, optional):
                Reduce opacity of shelters . Defaults to True.
            adjusted_shelter_wind_porosity (float, optional):
                Modity existing shelter porosity to help increase air movement from wind. Defaults to 0.75.
            radiant_temperature_adjustment (float, optional):
                The amount of radiant cooling to apply to the MRT. Defaults to -5.

        Returns:
            ExternalComfort:
                A modified object!
        """
        if not any(
            [
                add_overhead_shelter,
                add_additional_air_movement,
                add_misting,
                add_radiant_cooling,
            ]
        ):
            return self

        # check that inputs are right shape
        if isinstance(wind_speed_multiplier, (float, int)):
            wind_speed_multiplier = (
                np.ones_like(self.typology.wind_speed_multiplier)
                * wind_speed_multiplier
            )
        if len(wind_speed_multiplier) != len(self.typology.wind_speed_multiplier):
            raise ValueError(
                "wind_speed_multiplier must be a float or an iterable with the same length as the number times in the original EC object."
            )

        if isinstance(evaporative_cooling_effectiveness, (float, int)):
            evaporative_cooling_effectiveness = (
                np.ones_like(self.typology.evaporative_cooling_effect)
                * evaporative_cooling_effectiveness
            )
        if len(evaporative_cooling_effectiveness) != len(
            self.typology.evaporative_cooling_effect
        ):
            raise ValueError(
                "evaporative_cooling_effectiveness must be a float or an iterable with the same length as the number times in the original EC object."
            )

        if isinstance(radiant_temperature_adjustment, (float, int)):
            radiant_temperature_adjustment = (
                np.ones_like(self.typology.radiant_temperature_adjustment)
                * radiant_temperature_adjustment
            )
        if len(radiant_temperature_adjustment) != len(
            self.typology.radiant_temperature_adjustment
        ):
            raise ValueError(
                "radiant_temperature_adjustment must be a float or an iterable with the same length as the number times in the original EC object."
            )
        wind_speed_multiplier = np.array(wind_speed_multiplier)
        evaporative_cooling_effectiveness = np.array(evaporative_cooling_effectiveness)
        radiant_temperature_adjustment = np.array(radiant_temperature_adjustment)

        # create title to give the adjusted EC typology
        new_typology_name = f"{self.typology.name}"

        # OVERHEAD SHELTER
        if add_overhead_shelter:
            new_typology_name += " + overhead shelter"
            # TODO - check that overhead is not already sheltered and raise error if it is
            overhead_shelter_obj = Shelter(
                vertices=[[-3, -3, 3], [-3, 3, 3], [3, 3, 3], [3, -3, 3]],
                wind_porosity=0,
                radiation_porosity=0,
            )
            shelters = self.typology.shelters + [overhead_shelter_obj]
        else:
            shelters = self.typology.shelters

        # AIR MOVEMENT
        if add_additional_air_movement:
            if np.any(wind_speed_multiplier < self.typology.wind_speed_multiplier):
                raise ValueError(
                    'The original typology used has an elevated wind speed greater than that of the proposed "increase".'
                )
            new_typology_name += " + additional air movement"
            if increase_shelter_wind_porosity:
                if any(
                    shelter.wind_porosity > adjusted_shelter_wind_porosity
                    for shelter in shelters
                ):
                    raise ValueError(
                        'Shelters already on the original typology are more porous than the "shelter_wind_porosity_amount" value proposed.'
                    )
                if add_overhead_shelter:
                    shelters = [
                        Shelter(
                            vertices=shelter.vertices,
                            radiation_porosity=shelter.radiation_porosity,
                            wind_porosity=adjusted_shelter_wind_porosity,
                        )
                        for shelter in shelters[:-1]
                    ] + [shelters[-1]]
                else:
                    shelters = [
                        Shelter(
                            vertices=shelter.vertices,
                            radiation_porosity=shelter.radiation_porosity,
                            wind_porosity=adjusted_shelter_wind_porosity,
                        )
                        for shelter in shelters
                    ]

        # MISTING
        if add_misting:
            if np.any(
                evaporative_cooling_effectiveness
                < self.typology.evaporative_cooling_effect
            ):
                raise ValueError(
                    'The misting effect being applied is less effective than in the "baseline" it is being applied to.'
                )
            new_typology_name += f" + evaporative cooling (~{np.mean(evaporative_cooling_effectiveness):.0%} effective)"
        else:
            evaporative_cooling_effectiveness = self.typology.evaporative_cooling_effect

        # RADIANT COOLING
        if add_radiant_cooling:
            if np.any(radiant_temperature_adjustment > 0):
                raise ValueError("radiant_cooling_amount must be a negative value.")
            if np.any(
                radiant_temperature_adjustment
                > self.typology.radiant_temperature_adjustment
            ):
                raise ValueError(
                    'The radiant_temperature_adjustment being applied is less than in the original "baseline" it is being applied to.'
                )

            new_typology_name += (
                f" + radiant cooling ({np.mean(radiant_temperature_adjustment):0.0f}°C)"
            )
        else:
            radiant_temperature_adjustment = self.typology.evaporative_cooling_effect

        # create new typology
        new_typology = Typology(
            name=new_typology_name,
            shelters=shelters,
            wind_speed_multiplier=wind_speed_multiplier,
            evaporative_cooling_effect=evaporative_cooling_effectiveness,
            radiant_temperature_adjustment=radiant_temperature_adjustment,
        )

        return ExternalComfort(
            simulation_result=self.simulation_result,
            typology=new_typology,
        )

    @property
    def plot_title_string(self) -> str:
        """Return the description of this result suitable for use in plotting titles."""
        if self.typology.sky_exposure() == 1:
            return f"{location_to_string(self.simulation_result.epw.location)}\n{self.simulation_result.ground_material.to_lbt().display_name} ground, No shade\n{self.typology.name}"
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
            collection_to_series(self.universal_thermal_climate_index),
            collection_to_series(self.dry_bulb_temperature),
            collection_to_series(self.mean_radiant_temperature),
            collection_to_series(self.relative_humidity),
            collection_to_series(self.wind_speed),
            month,
            day,
            self.plot_title_string,
        )

    def plot_utci_heatmap(self) -> plt.Axes:
        """Create a heatmap showing the annual hourly UTCI values associated with this Typology.

        Returns:
            Figure: A matplotlib Figure object.
        """

        return utci_heatmap(
            utci_collection=self.universal_thermal_climate_index,
            ax=None,
            title=self.plot_title_string,
        )

    def plot_utci_heatmap_histogram(self) -> Figure:
        """Create a heatmap showing the annual hourly UTCI values associated with this Typology.

        Returns:
            Figure: A matplotlib Figure object.
        """

        return utci_heatmap_histogram(
            collection=self.universal_thermal_climate_index,
            title=self.plot_title_string,
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

    def plot_dbt_heatmap(self, **kwargs) -> plt.Axes:
        """Create a heatmap showing the annual hourly DBT values associated with this Typology.

        Args:
            **kwargs:
                Additional keyword arguments to pass to the heatmap function.

        Returns:
            Figure: A matplotlib Figure object.
        """

        return heatmap(
            series=collection_to_series(self.dry_bulb_temperature),
            cmap=DBT_COLORMAP,
            title=self.plot_title_string,
            **kwargs,
        )

    def plot_rh_heatmap(self, **kwargs) -> plt.Axes:
        """Create a heatmap showing the annual hourly RH values associated with this Typology.

        Args:
            **kwargs:
                Additional keyword arguments to pass to the heatmap function.

        Returns:
            Figure: A matplotlib Figure object.
        """

        return heatmap(
            series=collection_to_series(self.relative_humidity),
            cmap=RH_COLORMAP,
            title=self.plot_title_string,
            **kwargs,
        )

    def plot_ws_heatmap(self, **kwargs) -> plt.Axes:
        """Create a heatmap showing the annual hourly WS values associated with this Typology.

        Args:
            **kwargs:
                Additional keyword arguments to pass to the heatmap function.

        Returns:
            Figure: A matplotlib Figure object.
        """

        return heatmap(
            series=collection_to_series(self.wind_speed),
            cmap=WS_COLORMAP,
            title=self.plot_title_string,
            **kwargs,
        )

    def plot_mrt_heatmap(self, **kwargs) -> plt.Axes:
        """Create a heatmap showing the annual hourly MRT values associated with this Typology.

        Args:
            **kwargs:
                Additional keyword arguments to pass to the heatmap function.

        Returns:
            Figure: A matplotlib Figure object.
        """

        return heatmap(
            series=collection_to_series(self.mean_radiant_temperature),
            cmap=MRT_COLORMAP,
            title=self.plot_title_string,
            **kwargs,
        )
