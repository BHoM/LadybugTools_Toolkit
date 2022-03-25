from __future__ import annotations

import sys


sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from external_comfort.evaporatively_cooled_dbt_rh import evaporatively_cooled_dbt_rh
from external_comfort.shelter import Shelter, overlaps
from external_comfort.openfield import OpenfieldResult

from ladybug.epw import HourlyContinuousCollection
from ladybug.sunpath import Sunpath
from ladybug_comfort.collection.utci import UTCI
from ladybug_extension.datacollection.from_series import from_series
from ladybug_extension.datacollection.to_series import to_series


class Typology:
    def __init__(
        self,
        name: str = "",
        evaporative_cooling_effectiveness: float = 0,
        wind_speed_multiplier: float = 1,
        shelters: List[Shelter] = [],
    ) -> Typology:
        """Class for defining a specific external comfort typology, and calculating the resultant thermal comfort values.

        Args:
            name (str, optional): A string for the name of the typology. Defaults to "Openfield".
            evaporative_cooling_effectiveness (float, optional): A float between 0 and 1 for the effectiveness of the contextual evaporative cooling modifying air temperature. Defaults to 0.
            shelters (List[Shelter], optional): A list ShelterNew objects defining the sheltered portions around the typology. Defaults to no shelters.
        """
        self.name = name

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, evaporative_cooling_effectiveness={self.evaporative_cooling_effectiveness:0.2%}, shelters={[str(i) for i in self.shelters]})"

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


class TypologyResult:
    def __init__(
        self, typology: Typology, openfield_result: OpenfieldResult
    ) -> TypologyResult:
        self.typology = typology
        self.openfield_result = openfield_result

        if self.openfield_result.unshaded_mean_radiant_temperature is None:
            raise ValueError(f"The openfield_result object contains no results!")

        self.dry_bulb_temperature = self._dry_bulb_temperature()
        self.relative_humidity = self._relative_humidity()
        self.wind_speed = self._wind_speed()
        self.mean_radiant_temperature = self._mean_radiant_temperature()

        self.universal_thermal_climate_index = self._universal_thermal_climate_index()

    def _dry_bulb_temperature(self) -> HourlyContinuousCollection:
        """Return the effective dry bulb temperature for the given typology.

        Returns:
            HourlyContinuousCollection: An adjusted DBT based on effectiveness of evaporative cooling in the given typology.
        """
        dbt_rh = evaporatively_cooled_dbt_rh(
            self.openfield_result.openfield.epw,
            self.typology.evaporative_cooling_effectiveness,
        )
        return dbt_rh["dry_bulb_temperature"]

    def _relative_humidity(self) -> HourlyContinuousCollection:
        """Return the effective relative humidity for the given typology.

        Returns:
            HourlyContinuousCollection: An adjusted RH based on effectiveness of evaporative cooling in the given typology.
        """
        dbt_rh = evaporatively_cooled_dbt_rh(
            self.openfield_result.openfield.epw,
            self.typology.evaporative_cooling_effectiveness,
        )
        return dbt_rh["relative_humidity"]

    def _wind_speed(self) -> HourlyContinuousCollection:
        """Return the effective wind speed for the given typology.

        Returns:
            HourlyContinuousCollection: An adjusted wind speed based on the shelter configuration for the given typology.
        """

        if len(self.typology.shelters) == 0:
            self._ws = (
                self.openfield_result.openfield.epw.wind_speed
                * self.typology.wind_speed_multiplier
            )
            return self._ws

        collections = []
        for shelter in self.typology.shelters:
            collections.append(
                to_series(
                    shelter.effective_wind_speed(self.openfield_result.openfield.epw)
                    * self.typology.wind_speed_multiplier
                )
            )
        return from_series(
            pd.concat(collections, axis=1).min(axis=1).rename("Wind Speed (m/s)")
        )

        # DBT and RH values for the given typology
        dbt_rh = evaporatively_cooled_dbt_rh(
            openfield_result.openfield.epw, typology.evaporative_cooling_effectiveness
        )
        self.effective_dbt = dbt_rh["dry_bulb_temperature"]
        self.effective_rh = dbt_rh["relative_humidity"]

        # WS values for the given typology
        if len(typology.shelters) > 0:
            collections = []
            for shelter in self.typology.shelters:
                collections.append(
                    to_series(
                        shelter.effective_wind_speed(
                            self.openfield_result.openfield.epw
                        )
                        * self.typology.wind_speed_multiplier
                    )
                )
            self.effective_ws = from_series(
                pd.concat(collections, axis=1).min(axis=1).rename("Wind Speed (m/s)")
            )
        else:
            self.effective_ws = (
                self.openfield_result.openfield.epw.wind_speed
                * self.typology.wind_speed_multiplier
            )

        # MRT values for the given typology
        shaded_mrt = to_series(self.openfield_result.shaded_mean_radiant_temperature)
        unshaded_mrt = to_series(
            self.openfield_result.unshaded_mean_radiant_temperature
        )

        sun_exposure = self.annual_hourly_sun_exposure()
        effective_sky_visibility = self.effective_sky_visibility()
        daytime = np.array(
            [
                True if i > 0 else False
                for i in self.openfield_result.openfield.epw.global_horizontal_radiation
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

    def _mean_radiant_temperature(self) -> HourlyContinuousCollection:
        """Return the effective mean radiant temperature for the given typology.

        Returns:
            HourlyContinuousCollection: An calculated mean radiant temperature based on the shelter configuration for the given typology.
        """

        shaded_mrt = to_series(self.openfield_result.shaded_mean_radiant_temperature)
        unshaded_mrt = to_series(
            self.openfield_result.unshaded_mean_radiant_temperature
        )

        sun_exposure = self.__annual_hourly_sun_exposure()
        effective_sky_visibility = self.__sky_visibility()
        daytime = np.array(
            [
                True if i > 0 else False
                for i in self.openfield_result.openfield.epw.global_horizontal_radiation
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

    def __sky_visibility(self) -> float:
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

    def __annual_hourly_sun_exposure(self) -> List[float]:
        """Return NaN if sun below horizon, and a value between 0-1 for sun-hidden to sun-exposed.

        Returns:
            List[float]: A list of sun visibility values for each hour of the year.
        """

        sunpath = Sunpath.from_location(self.openfield_result.openfield.epw.location)
        suns = [sunpath.calculate_sun_from_hoy(i) for i in range(8760)]
        sun_is_up = np.array([True if sun.altitude > 0 else False for sun in suns])

        nans = np.empty(len(self.openfield_result.openfield.epw.dry_bulb_temperature))
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

    def _universal_thermal_climate_index(self) -> HourlyContinuousCollection:
        """Return the effective UTCI for the given typology.

        Returns:
            HourlyContinuousCollection: The calculated UTCI based on the shelter configuration for the given typology.
        """

        # logging.info(f"[{self.typology.name}] Calculating {inspect.stack()[0][3]}")

        return UTCI(
            air_temperature=self.dry_bulb_temperature,
            rel_humidity=self.relative_humidity,
            rad_temperature=self.mean_radiant_temperature,
            wind_speed=self.wind_speed,
        ).universal_thermal_climate_index


TYPOLOGIES: Dict[str, Typology] = {
    "Openfield": Typology(
        name="Openfield",
        evaporative_cooling_effectiveness=0,
        shelters=[],
        wind_speed_multiplier=1,
    ),
    "Enclosed": Typology(
        name="Enclosed",
        evaporative_cooling_effectiveness=0,
        shelters=[Shelter(altitude_range=[0, 90], azimuth_range=[0, 360], porosity=0)],
        wind_speed_multiplier=1,
    ),
    "FrittedEnclosure": Typology(
        name="Fritted enclosure",
        evaporative_cooling_effectiveness=0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[0, 360], porosity=0.5)
        ],
        wind_speed_multiplier=1,
    ),
    "SkyShelter": Typology(
        name="Sky-shelter",
        evaporative_cooling_effectiveness=0,
        shelters=[Shelter(altitude_range=[45, 90], azimuth_range=[0, 360], porosity=0)],
        wind_speed_multiplier=1,
    ),
    "FrittedSkyShelter": Typology(
        name="Fritted sky-shelter",
        evaporative_cooling_effectiveness=0,
        shelters=[
            Shelter(altitude_range=[45, 90], azimuth_range=[0, 360], porosity=0.5),
        ],
        wind_speed_multiplier=1,
    ),
    "NearWater": Typology(
        name="Near water",
        evaporative_cooling_effectiveness=0.15,
        shelters=[
            Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
        ],
        wind_speed_multiplier=1.2,
    ),
    "Misting": Typology(
        name="Misting",
        evaporative_cooling_effectiveness=0.3,
        shelters=[
            Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
        ],
        wind_speed_multiplier=0.5,
    ),
    "PDEC": Typology(
        name="PDEC",
        evaporative_cooling_effectiveness=0.7,
        shelters=[
            Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
        ],
        wind_speed_multiplier=0.5,
    ),
    "NorthShelter": Typology(
        name="North shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[337.5, 22.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "NortheastShelter": Typology(
        name="Northeast shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[22.5, 67.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "EastShelter": Typology(
        name="East shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[67.5, 112.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "SoutheastShelter": Typology(
        name="Southeast shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[112.5, 157.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "SouthShelter": Typology(
        name="South shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[157.5, 202.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "SouthwestShelter": Typology(
        name="Southwest shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[202.5, 247.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "WestShelter": Typology(
        name="West shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[247.5, 292.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "NorthwestShelter": Typology(
        name="Northwest shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[292.5, 337.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "NorthShelterWithCanopy": Typology(
        name="North shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[337.5, 22.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "NortheastShelterWithCanopy": Typology(
        name="Northeast shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[22.5, 67.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "EastShelterWithCanopy": Typology(
        name="East shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[67.5, 112.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "SoutheastShelterWithCanopy": Typology(
        name="Southeast shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[112.5, 157.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "SouthShelterWithCanopy": Typology(
        name="South shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[157.5, 202.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "SouthwestShelterWithCanopy": Typology(
        name="Southwest shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[202.5, 247.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "WestShelterWithCanopy": Typology(
        name="West shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[247.5, 292.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "NorthwestShelterWithCanopy": Typology(
        name="Northwest shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[292.5, 337.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "EastWestShelter": Typology(
        name="East-west shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[67.5, 112.5], porosity=0),
            Shelter(altitude_range=[0, 70], azimuth_range=[247.5, 292.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
    "EastWestShelterWithCanopy": Typology(
        name="East-west shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[67.5, 112.5], porosity=0),
            Shelter(altitude_range=[0, 90], azimuth_range=[247.5, 292.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    ),
}
