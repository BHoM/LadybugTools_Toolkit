from __future__ import annotations

import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import logging
from typing import List

logging.basicConfig(level=logging.INFO)
import inspect

import numpy as np
import pandas as pd
from external_comfort.evaporatively_cooled_dbt_rh import evaporatively_cooled_dbt_rh
from external_comfort.openfield import Openfield
from external_comfort.shelter import overlaps
from external_comfort.shelter import Shelter
from ladybug.epw import HourlyContinuousCollection
from ladybug.sunpath import Sunpath
from ladybug_comfort.collection.utci import UTCI
from ladybug_extension.datacollection.from_series import from_series
from ladybug_extension.datacollection.to_series import to_series


class Typology:
    def __init__(
        self,
        openfield: Openfield,
        name: str = "",
        evaporative_cooling_effectiveness: float = 0,
        wind_speed_multiplier: float = 1,
        shelters: List[Shelter] = [],
        calculate: bool = False,
    ) -> Typology:
        """Class for defining a specific external comfort typology, and calculating the resultant thermal comfort values.

        Args:
            openfield (Openfield): An Openfield object.
            name (str, optional): A string for the name of the typology. Defaults to "Openfield".
            evaporative_cooling_effectiveness (float, optional): A float between 0 and 1 for the effectiveness of the contextual evaporative cooling modifying air temperature. Defaults to 0.
            shelters (List[Shelter], optional): A list ShelterNew objects defining the sheltered portions around the typology. Defaults to no shelters.
            calculate (bool, optional): A boolean for whether to calculate the comfort values for the typology. Defaults to False.
        """
        self.name = name
        self.openfield = openfield

        if not (
            (0 <= evaporative_cooling_effectiveness <= 1)
            or isinstance(evaporative_cooling_effectiveness, (int, float))
        ):
            raise ValueError(
                f"evaporative_cooling_effectiveness must be a number between 0 and 1"
            )
        else:
            self.evaporative_cooling_effectiveness = evaporative_cooling_effectiveness

        if not 0 <= wind_speed_multiplier:
            raise ValueError(f"wind_speed_multiplier must be a number greater than 0")
        else:
            self.wind_speed_multiplier = wind_speed_multiplier

        if overlaps(shelters):
            raise ValueError(f"shelters overlap")
        else:
            self.shelters = shelters

        self._effective_ws: HourlyContinuousCollection = None
        self._effective_dbt: HourlyContinuousCollection = None
        self._effective_rh: HourlyContinuousCollection = None
        self._effective_mrt: HourlyContinuousCollection = None
        self._effective_utci: HourlyContinuousCollection = None

        if calculate:
            self._calculate()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, openfield={self.openfield}, evaporative_cooling_effectiveness={self.evaporative_cooling_effectiveness:0.2%}, shelters={[str(i) for i in self.shelters]})"

    def _calculate(self) -> Typology:
        """Calculate the thermal comfort values for the typology, and in doing so populate the object with values."""
        self.universal_thermal_climate_index
        return self

    @property
    def dry_bulb_temperature(self) -> HourlyContinuousCollection:
        """Return the effective dry bulb temperature for the typology."""
        if not self._effective_dbt:
            self._effective_dbt = self.effective_dbt()
        return self._effective_dbt

    @property
    def relative_humidity(self) -> HourlyContinuousCollection:
        """Return the effective relative humidity for the typology."""
        if not self._effective_rh:
            self._effective_rh = self.effective_rh()
        return self._effective_rh

    @property
    def wind_speed(self) -> HourlyContinuousCollection:
        """Return the effective wind speed for the typology."""
        if not self._effective_ws:
            self._effective_ws = self.effective_ws()
        return self._effective_ws

    @property
    def mean_radiant_temperature(self) -> HourlyContinuousCollection:
        """Return the effective mean radiant temperature for the typology."""
        if not self._effective_mrt:
            self._effective_mrt = self.effective_mrt()
        return self._effective_mrt

    @property
    def universal_thermal_climate_index(self) -> HourlyContinuousCollection:
        """Return the effective UTCI for the typology."""
        if not self._effective_utci:
            self._effective_utci = self.effective_utci()
        return self._effective_utci

    def effective_sky_visibility(self) -> float:
        """Calculate the proportion of sky visible from a typology with any nuber of shelters."""
        unsheltered_proportion = 1
        sheltered_proportion = 0
        for shelter in self.shelters:
            occ = shelter.sky_occluded
            unsheltered_proportion -= occ
            sheltered_proportion += occ * shelter.porosity

        return sheltered_proportion + unsheltered_proportion

    def annual_hourly_sun_exposure(self) -> List[float]:
        """Return NaN if sun below horizon, and a value between 0-1 for sun-hidden to sun-exposed"""

        sunpath = Sunpath.from_location(self.openfield.epw.location)
        suns = [sunpath.calculate_sun_from_hoy(i) for i in range(8760)]
        sun_is_up = np.array([True if sun.altitude > 0 else False for sun in suns])

        nans = np.empty(len(self.openfield.epw.dry_bulb_temperature))
        nans[:] = np.NaN

        if len(self.shelters) == 0:
            return np.where(sun_is_up, 1, nans)

        blocked = []
        for shelter in self.shelters:
            shelter_blocking = shelter.sun_blocked(suns)
            temp = np.where(shelter_blocking, shelter.porosity, nans)
            temp = np.where(np.logical_and(np.isnan(temp), sun_is_up), 1, temp)
            blocked.append(temp)

        return pd.DataFrame(blocked).T.min(axis=1).values.tolist()

    def effective_ws(self) -> HourlyContinuousCollection:
        """Based on the shelters in-place, create a composity wind-speed collection affected by those shelters."""

        if len(self.shelters) == 0:
            self._ws = self.openfield.epw.wind_speed * self.wind_speed_multiplier
            return self._ws

        collections = []
        for shelter in self.shelters:
            collections.append(
                to_series(
                    shelter.effective_wind_speed(self.openfield.epw)
                    * self.wind_speed_multiplier
                )
            )
        return from_series(
            pd.concat(collections, axis=1).min(axis=1).rename("Wind Speed (m/s)")
        )

    def effective_dbt(self) -> HourlyContinuousCollection:
        """Based on the evaporative cooling configuration, calculate the effective dry bulb temperature for each hour of the year."""

        return evaporatively_cooled_dbt_rh(
            self.openfield.epw, self.evaporative_cooling_effectiveness
        )["dry_bulb_temperature"]

    def effective_rh(self) -> HourlyContinuousCollection:
        """Based on the evaporative cooling configuration, calculate the effective dry bulb temperature for each hour of the year."""

        return evaporatively_cooled_dbt_rh(
            self.openfield.epw, self.evaporative_cooling_effectiveness
        )["relative_humidity"]

    def effective_mrt(self) -> HourlyContinuousCollection:
        """Based on the shelters in-place, create a composite mean radiant temperature collection due to shading from those shelters and some pre-simulated shaded/unshaded mean-radiant temperatures."""

        shaded_mrt = to_series(self.openfield.shaded_mean_radiant_temperature)
        unshaded_mrt = to_series(self.openfield.unshaded_mean_radiant_temperature)

        sun_exposure = self.annual_hourly_sun_exposure()
        effective_sky_visibility = self.effective_sky_visibility()
        daytime = np.array(
            [
                True if i > 0 else False
                for i in self.openfield.epw.global_horizontal_radiation
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
            ewm_smoothed = series.ewm(span=ewm_span).mean()

            # Choose from ewm or original values based on ewm mask
            new_series = ewm_smoothed.where(ewm_mask, series)
            return new_series

        mrt_series = __smoother(
            mrt_series, difference_threshold=-10, transition_window=4, ewm_span=1.25
        )

        return from_series(mrt_series)

    def effective_utci(self) -> HourlyContinuousCollection:
        """Return the effective UTCI for the given typology."""

        logging.info(f"[{self.name}] Calculating {inspect.stack()[0][3]}")

        return UTCI(
            air_temperature=self.dry_bulb_temperature,
            rel_humidity=self.relative_humidity,
            rad_temperature=self.mean_radiant_temperature,
            wind_speed=self.wind_speed,
        ).universal_thermal_climate_index

    @property
    def description(self) -> str:
        """Return a description of the typology."""

        if self.name:
            name = f"[{self.name}] "
        else:
            name = ""

        shelter_descriptions = []
        for shelter in self.shelters:
            shelter_descriptions.append(shelter.description)
        shelter_descriptions = [s for s in shelter_descriptions if s is not None]

        wind_adj = ""
        if self.wind_speed_multiplier != 1:
            if self.wind_speed_multiplier < 1:
                wind_adj = (
                    f", and wind speed reduced by {1 - self.wind_speed_multiplier:0.0%}"
                )
            else:
                wind_adj = f", and wind speed increased by {self.wind_speed_multiplier - 1:0.0%}"
        if (len(shelter_descriptions) == 0) and (
            self.evaporative_cooling_effectiveness == 0
        ):
            return f"{name}Fully exposed" + wind_adj
        elif (len(shelter_descriptions) == 0) and (
            self.evaporative_cooling_effectiveness != 0
        ):
            return (
                f"{name}Fully exposed, with {self.evaporative_cooling_effectiveness:0.0%} effective evaporative cooling"
                + wind_adj
            )
        elif (len(shelter_descriptions) != 0) and (
            self.evaporative_cooling_effectiveness == 0
        ):
            return f"{name}{' and '.join(shelter_descriptions).capitalize()}" + wind_adj
        else:
            return (
                f"{name}{' and '.join(shelter_descriptions).capitalize()}, with {self.evaporative_cooling_effectiveness:0.0%} effective evaporative cooling"
                + wind_adj
            )
