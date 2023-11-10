"""Base class for typology objects."""
# pylint: disable=E0401
import json
from dataclasses import dataclass
from pathlib import Path

# pylint: enable=E0401

import numpy as np
import pandas as pd
from ladybug.epw import EPW, HourlyContinuousCollection

from ..bhom import CONSOLE_LOGGER, decorator_factory
from ..helpers import (
    convert_keys_to_snake_case,
    decay_rate_smoother,
    evaporative_cooling_effect,
)
from ..ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)
from ._shelterbase import (
    Shelter,
    annual_sky_exposure,
    annual_sun_exposure,
    annual_wind_speed,
)
from .simulate import SimulationResult


@dataclass(init=True, repr=True, eq=True)
class Typology:
    """_"""

    identifier: str
    shelters: tuple[Shelter] = ()
    evaporative_cooling_effect: tuple[float] = (0,) * 8760
    target_wind_speed: tuple[float] = (None,) * 8760
    radiant_temperature_adjustment: tuple[float] = (0,) * 8760

    def __post_init__(self):
        """_"""

        # validation
        if len(self.shelters) > 0:
            if any(not isinstance(shelter, Shelter) for shelter in self.shelters):
                raise ValueError("All shelters must be of type 'Shelter'.")

        if len(self.evaporative_cooling_effect) != 8760:
            raise ValueError("evaporative_cooling_effect must be 8760 items long.")
        if any(
            not isinstance(i, (float, int)) for i in self.evaporative_cooling_effect
        ):
            raise ValueError("evaporative_cooling_effect must be a list of floats.")
        if any(i < 0 for i in self.evaporative_cooling_effect):
            raise ValueError("evaporative_cooling_effect must be >= 0.")
        if any(i > 1 for i in self.evaporative_cooling_effect):
            raise ValueError("evaporative_cooling_effect must be <= 1.")

        if len(self.target_wind_speed) != 8760:
            raise ValueError("target_wind_speed must be 8760 items long.")
        for tws in self.target_wind_speed:
            if not isinstance(tws, (float, int, type(None))):
                raise ValueError(
                    "target_wind_speed must be a list of numbers, which can include None values."
                )
            if tws is None:
                continue
            if tws < 0:
                raise ValueError(
                    "target_wind_speed must be >= 0, or None (representing no change to in-situ wind speed)."
                )

        if len(self.radiant_temperature_adjustment) != 8760:
            raise ValueError("radiant_temperature_adjustment must be 8760 items long.")
        if any(
            not isinstance(i, (float, int)) for i in self.radiant_temperature_adjustment
        ):
            raise ValueError("radiant_temperature_adjustment must be a list of floats.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.identifier})"

    def to_dict(self) -> str:
        """Convert this object to a dictionary."""
        shelter_dicts = []
        for shelter in self.shelters:
            shelter_dicts.append(shelter.to_dict())

        d = {
            "_t": "BH.oM.LadybugTools.Typology",
            "Identifier": self.identifier,
            "Shelters": shelter_dicts,
            "EvaporativeCoolingEffect": self.evaporative_cooling_effect,
            "TargetWindSpeed": self.target_wind_speed,
            "RadiantTemperatureAdjustment": self.radiant_temperature_adjustment,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Shelter":
        """Create this object from a dictionary."""
        d = convert_keys_to_snake_case(d)

        new_shelters = []
        for shelter in d["shelters"]:
            if isinstance(shelter, dict):
                new_shelters.append(Shelter.from_dict(shelter))
        d["shelters"] = new_shelters

        return cls(
            identifier=d["identifier"],
            shelters=d["shelters"],
            evaporative_cooling_effect=d["evaporative_cooling_effect"],
            target_wind_speed=d["target_wind_speed"],
            radiant_temperature_adjustment=d["radiant_temperature_adjustment"],
        )

    def to_json(self) -> str:
        """Convert this object to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_string: str) -> "Shelter":
        """Create this object from a JSON string."""
        return cls.from_dict(json.loads(json_string))

    def to_file(self, path: Path) -> Path:
        """Convert this object to a JSON file."""
        if Path(path).suffix != ".json":
            raise ValueError("path must be a JSON file.")

        with open(Path(path), "w") as fp:
            fp.write(self.to_json())

        return Path(path)

    @classmethod
    def from_file(cls, path: Path) -> "Typology":
        """Create this object from a JSON file."""
        with open(Path(path), "r") as fp:
            return cls.from_json(fp.read())

    @property
    def average_evaporative_cooling_effect(self) -> float:
        """_"""
        return np.mean(self.evaporative_cooling_effect)

    @property
    def average_radiant_temperature_adjustment(self) -> float:
        """_"""
        return np.mean(self.radiant_temperature_adjustment)

    @property
    def average_target_wind_speed(self) -> float:
        """_"""
        return np.mean([i for i in self.target_wind_speed if i is not None])

    @decorator_factory()
    def sky_exposure(self) -> list[float]:
        """Direct access to "sky_exposure" method for this typology object."""
        return annual_sky_exposure(self.shelters, include_radiation_porosity=True)

    @decorator_factory()
    def sun_exposure(self, epw: EPW) -> list[float]:
        """Direct access to "sun_exposure" method for this typology object."""
        return annual_sun_exposure(self.shelters, epw, include_radiation_porosity=True)

    @decorator_factory()
    def wind_speed(self, epw: EPW) -> list[float]:
        """Direct access to "wind_speed" method for this typology object."""
        shelter_wind_speed = annual_wind_speed(self.shelters, epw)

        ws = []
        for sh_ws, tgt_ws in list(zip(shelter_wind_speed, self.target_wind_speed)):
            if tgt_ws is None:
                ws.append(sh_ws)
            else:
                ws.append(tgt_ws)
        return epw.wind_speed.get_aligned_collection(ws)

    @decorator_factory()
    def dry_bulb_temperature(self, epw: EPW) -> HourlyContinuousCollection:
        """Get the effective DBT for the given EPW file for this Typology.

        Args:
            epw (EPW): A ladybug EPW object.

        Returns:
            HourlyContinuousCollection: The effective DBT following application of any evaporative
                cooling effects.
        """
        # if there is no evaporative cooling effect, return the original DBT
        if sum(self.evaporative_cooling_effect) == 0:
            return epw.dry_bulb_temperature

        # if there is evaporative cooling effect, return the adjusted DBT
        dbt_evap = []
        for dbt, rh, ece, atm in list(
            zip(
                *[
                    epw.dry_bulb_temperature,
                    epw.relative_humidity,
                    self.evaporative_cooling_effect,
                    epw.atmospheric_station_pressure,
                ]
            )
        ):
            _dbt, _ = evaporative_cooling_effect(
                dry_bulb_temperature=dbt,
                relative_humidity=rh,
                evaporative_cooling_effectiveness=ece,
                atmospheric_pressure=atm,
            )
            dbt_evap.append(_dbt)

        return epw.dry_bulb_temperature.get_aligned_collection(dbt_evap)

    @decorator_factory()
    def relative_humidity(self, epw: EPW) -> HourlyContinuousCollection:
        """Get the effective RH for the given EPW file for this Typology.

        Args:
            epw (EPW): A ladybug EPW object.

        Returns:
            HourlyContinuousCollection: The effective RH following application of any evaporative
                cooling effects.
        """

        # if there is no evaporative cooling effect, return the original RH
        if sum(self.evaporative_cooling_effect) == 0:
            return epw.relative_humidity

        # if there is evaporative cooling effect, return the adjusted RH
        rh_evap = []
        for dbt, rh, ece, atm in list(
            zip(
                *[
                    epw.dry_bulb_temperature,
                    epw.relative_humidity,
                    self.evaporative_cooling_effect,
                    epw.atmospheric_station_pressure,
                ]
            )
        ):
            _, _rh = evaporative_cooling_effect(
                dry_bulb_temperature=dbt,
                relative_humidity=rh,
                evaporative_cooling_effectiveness=ece,
                atmospheric_pressure=atm,
            )
            rh_evap.append(_rh)

        return epw.relative_humidity.get_aligned_collection(rh_evap)

    @decorator_factory()
    def mean_radiant_temperature(
        self,
        simulation_result: SimulationResult,
    ) -> HourlyContinuousCollection:
        """Return the effective mean radiant temperature for the given typology following a
            simulation of the collections necessary to calculate this.

        Args:
            simulation_result (SimulationResult): An object containing all collections describing
                shaded and unshaded mean-radiant-temperature.

        Returns:
            HourlyContinuousCollection: An calculated mean radiant temperature based on the shelter
                configuration for the given typology.
        """

        shaded_mrt = collection_to_series(
            simulation_result.shaded_mean_radiant_temperature
        )
        unshaded_mrt = collection_to_series(
            simulation_result.unshaded_mean_radiant_temperature
        )

        _sun_exposure = self.sun_exposure(simulation_result.epw)
        _sky_exposure = self.sky_exposure()
        mrts = []
        for sun_exp, sky_exp, shaded, unshaded in list(
            zip(*[_sun_exposure, _sky_exposure, shaded_mrt.values, unshaded_mrt.values])
        ):
            if sun_exp:
                mrts.append(np.interp(sun_exp, [0, 1], [shaded, unshaded]))
            else:
                mrts.append(np.interp(sky_exp, [0, 1], [shaded, unshaded]))

        # Fill any gaps where sun-visible/sun-occluded values are missing
        mrt_series = pd.Series(
            mrts, index=shaded_mrt.index, name=shaded_mrt.name
        ).interpolate()

        # apply an exponentially weighted moving average to account for
        # transition between shaded/unshaded periods on surrounding surface
        # temperatures
        mrt_series = decay_rate_smoother(
            mrt_series, difference_threshold=-10, transition_window=4, ewm_span=1.25
        )

        # fill missing data at beginning with interpolation from end of year
        mrt_series.iloc[0] = mrt_series.iloc[24]
        mrt_series.interpolate(inplace=True)

        # apply radiant temperature adjustment if given
        mrt_series += self.radiant_temperature_adjustment
        return collection_from_series(mrt_series)
