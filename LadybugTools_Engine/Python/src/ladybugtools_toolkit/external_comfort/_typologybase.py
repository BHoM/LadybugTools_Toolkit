"""Base class for typology objects."""
import numpy as np
import pandas as pd
from ladybug.epw import EPW, HourlyContinuousCollection
from pydantic import BaseModel, Field, validator  # pylint: disable=E0611

from ..bhom import decorator_factory
from ..helpers import decay_rate_smoother, evaporative_cooling_effect
from ..ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)
from ._shelterbase import (
    Point3D,
    Shelter,
    annual_sky_exposure,
    annual_sun_exposure,
    annual_wind_speed,
)
from .simulate import SimulationResult


class Typology(BaseModel):
    """_"""

    name: str = Field(alias="Name")
    shelters: list[Shelter] = Field(alias="Shelters", default_factory=list)
    evaporative_cooling_effect: list[float] = Field(
        alias="EvaporativeCoolingEffect",
        min_items=1,
        max_items=8760,
        default=(0,) * 8760,
        ge=0,
        le=1,
    )
    target_wind_speed: list[float] = Field(
        alias="TargetWindSpeed",
        min_items=8760,
        max_items=8760,
        default=(np.nan,) * 8760,
    )
    radiant_temperature_adjustment: list[float] = Field(
        alias="RadiantTemperatureAdjustment",
        min_items=8760,
        max_items=8760,
        default=(0,) * 8760,
    )

    class Config:
        """_"""

        arbitrary_types_allowed = True
        allow_population_by_field_name = True
        json_encoders = {
            Point3D: lambda v: v.to_dict(),
        }

    @validator(
        "target_wind_speed",
        pre=True,
        each_item=True,
    )
    @classmethod
    def validate_target_wind_speed(cls, value) -> list[float]:  # pylint: disable=E0213
        """_"""
        if value < 0 and not np.isnan(value):
            raise ValueError("value must be >= 0.")
        return value

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
            if np.isnan(tgt_ws):
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
