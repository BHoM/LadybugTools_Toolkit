import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import FunctionType
from typing import Any, Dict, List, Tuple, Union

import honeybee.dictutil as hb_dict_util
import honeybee_energy.dictutil as energy_dict_util
import honeybee_radiance.dictutil as radiance_dict_util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from honeybee.model import Face, Model, Shade
from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybug.viewsphere import ViewSphere
from ladybug_comfort.collection.pmv import PMV
from ladybug_comfort.collection.utci import UTCI
from ladybug_geometry.geometry3d import (
    Face3D,
    LineSegment3D,
    Plane,
    Point3D,
    Ray3D,
    Vector3D,
)
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d

from ..bhom import decorator_factory, keys_to_pascalcase, keys_to_snakecase
from ..helpers import decay_rate_smoother, evaporative_cooling_effect
from ..ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)
from ..ladybug_extension.epw import sun_position_list

# from ..plot import utci_heatmap_histogram
from .shelter import (
    Shelter,
    annual_sky_exposure,
    annual_sun_exposure,
    annual_wind_speed,
)
from .simulate import SimulationResult


@dataclass(init=True, repr=True, eq=True)
class Typology:
    """An external comfort typology, describing the context in which thermal comfort will be calculated.

    Args:
        name (str):
            The name of the external comfort typology.
        shelters (list[Shelter], optional):
            A list of shelters modifying exposure to the elements.
            Defaults to None.
        evaporative_cooling_effect (float | list[float], optional):
            An amount of evaporative cooling to add to results calculated by
            this typology. Defaults to None, which does not introduce any
            change. Can be used to account for local evaporative cooling
            periodically operating. If a list of 8760 values
            is given, where 0 is found in that list the original DBT and RH
            will be used. If a single value is given, then this value will be
            used across all time steps.
        target_wind_speed (float | list[float], optional):
            A value to be used to override wind speed calulated by the Shelter
            objects. Defaults to 0, which does not introduce any change.
            Can be used to account for local fans periodically operating, or
            approximating the effects of acceleration. If a list of 8760 values
            is given, where 0 is found in that list the original wind speed
            will be used. If values given are lower than wind speed, then wind
            speed will be used.
        radiant_temperature_adjustment (float | list[float], optional):
            A change in MRT to be applied to the resultant MRT. Defaults to
            0, which does not introduce any change.

    Returns:
        Typology: An external comfort typology.
    """

    name: str = field(init=True, compare=True, repr=True)
    shelters: list[Shelter] = field(
        init=True, compare=True, repr=False, default_factory=list
    )
    evaporative_cooling_effect: Union[float, List[float]] = field(
        init=True, compare=True, repr=False, default=0
    )
    target_wind_speed: Union[float, List[float]] = field(
        init=True, compare=True, repr=False, default=1
    )
    radiant_temperature_adjustment: Union[float, List[float]] = field(
        init=True, compare=True, repr=False, default=0
    )

    _t: str = field(
        init=False, compare=True, repr=False, default="BH.oM.LadybugTools.Typology"
    )

    def __post_init__(self):
        # null handling and validation
        null_values = {
            "evaporative_cooling_effect": 0,
            "target_wind_speed": 0,
            "radiant_temperature_adjustment": 0,
        }
        for k, v in null_values.items():
            if getattr(self, k) is None:
                setattr(self, k, np.full(8760, v))
            elif isinstance(getattr(self, k), (float, int)):
                setattr(self, k, np.full(8760, getattr(self, k)))
            if len(getattr(self, k)) != 8760:
                raise ValueError(f"{k} must be a float or a list/tuple of 8760 values")
            setattr(self, k, np.atleast_1d(getattr(self, k)))

            if k == "evaporative_cooling_effect":
                if sum(getattr(self, k) > 1) + sum(getattr(self, k) < 0) != 0:
                    raise ValueError(f"{k} values must be between 0 and 1")

    #     @classmethod
    #     def from_dict(cls, dictionary: Dict[str, Any]) -> Typology:
    #         """Create this object from a dictionary."""

    #         # handle shelter object conversions
    #         for n, shelter in enumerate(dictionary["Shelters"]):
    #             if not isinstance(shelter, Shelter):
    #                 dictionary["Shelters"][n] = Shelter.from_dict(shelter)

    #         return cls(
    #             Name=dictionary["Name"],
    #             Shelters=dictionary["Shelters"],
    #             EvaporativeCoolingEffect=dictionary["EvaporativeCoolingEffect"],
    #             WindSpeedMultiplier=dictionary["WindSpeedMultiplier"],
    #             RadiantTemperatureAdjustment=dictionary["RadiantTemperatureAdjustment"],
    #         )

    #     @classmethod
    #     def from_json(cls, json_string: str) -> Typology:
    #         """Create this object from a JSON string."""
    #         return cls.from_dict(json.loads(json_string))

    #     def to_dict(self) -> Dict[str, Any]:
    #         """Return this object as it's dictionary equivalent."""
    #         dictionary = {}
    #         for k, v in self.__dict__.items():
    #             if isinstance(getattr(self, k), FunctionType):
    #                 continue
    #             dictionary[k] = v
    #         dictionary["_t"] = self._t
    #         return dictionary

    #     def to_json(self) -> str:
    #         """Return this object as it's JSON string equivalent."""
    #         return json.dumps(self.to_dict(), cls=BHoMEncoder)

    def sky_exposure(self) -> list[float]:
        """Direct access to "sky_exposure" method for this typology object."""
        return annual_sky_exposure(self.shelters, include_radiation_porosity=True)

    def sun_exposure(self, epw: EPW) -> list[float]:
        """Direct access to "sun_exposure" method for this typology object."""
        return annual_sun_exposure(self.shelters, epw, include_radiation_porosity=True)

    def wind_speed(self, epw: EPW) -> list[float]:
        """Direct access to "wind_speed" method for this typology object."""
        shelter_wind_speed = annual_wind_speed(self.shelters, epw)
        return epw.wind_speed.get_aligned_collection(
            np.stack([shelter_wind_speed, self.target_wind_speed], axis=1).max(axis=1)
        )

    @decorator_factory(disable=False)
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

    @decorator_factory(disable=False)
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

    @decorator_factory(disable=False)
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

    def universal_thermal_climate_index(
        self, simulation_result: SimulationResult, return_comfort_obj: bool = False
    ) -> UTCI | HourlyContinuousCollection:
        """Return the universal thermal climate index for the given typology
            following a simulation of the collections necessary to calculate
            this.

        Args:
            simulation_result (SimulationResult):
                An object containing all collections describing shaded and
                unshaded mean-radiant-temperature.
            return_comfort_obj (bool, optional):
                Set to True to return the UTCI comfort object instead of the
                data collection.

        Returns:
            Union[UTCI, HourlyContinuousCollection]:
                The resultant UTCI for this Typology, in the given
                SimulationResult location.
        """
        epw = simulation_result.epw
        utci = UTCI(
            air_temperature=self.dry_bulb_temperature(epw),
            rel_humidity=self.relative_humidity(epw),
            rad_temperature=self.mean_radiant_temperature(simulation_result),
            wind_speed=self.wind_speed(epw),
        )

        # TODO - add logging again
        # CONSOLE_LOGGER.info(
        #     f"[{simulation_result.Identifier} - {self.Name}] - Calculating universal thermal climate index"
        # )

        if return_comfort_obj:
            return utci

        return utci.universal_thermal_climate_index

    @decorator_factory(disable=False)
    def standard_effective_temperature(
        self,
        simulation_result: SimulationResult,
        met_rate: float = 1.1,
        clo_value: float = 0.7,
        return_comfort_obj: bool = False,
    ) -> Union[PMV, HourlyContinuousCollection]:
        """Return the predicted mean vote for the given typology
            following a simulation of the collections necessary to calculate
            this.

        Args:
            simulation_result (SimulationResult):
                An object containing all collections describing shaded and
                unshaded mean-radiant-temperature.
            met_rate (float, optional):
                Metabolic rate value.
            clo_value (float, optional):
                Clothing value.
            return_comfort_obj (bool, optional):
                Set to True to return the PMV comfort object instead of the
                data collection.

        Returns:
            Union[PMV, HourlyContinuousCollection]:
                The resultant PMV for this Typology, in the given
                SimulationResult location.
        """
        epw = simulation_result.epw
        pmv = PMV(
            air_temperature=self.dry_bulb_temperature(epw),
            rel_humidity=self.relative_humidity(epw),
            rad_temperature=self.mean_radiant_temperature(simulation_result),
            air_speed=self.wind_speed(epw),
            met_rate=met_rate,
            clo_value=clo_value,
        )

        if return_comfort_obj:
            return pmv

        return pmv.standard_effective_temperature


#     def plot_utci_hist(self, res: SimulationResult) -> None:
#         """Convenience method to plot UTCI histogram directly from a typology."""
#         warnings.warn(
#             "While it is possible to call from a Typology object, the recommended method of calling the UTCI histogram is from an ExternalComfort object."
#         )
#         return utci_heatmap_histogram(
#             self.universal_thermal_climate_index(res), title=self.Name
#         )


# def combine_typologies(
#     typologies: Tuple[Typology],
#     evaporative_cooling_effect_weights: Tuple[float] = None,
#     wind_speed_multiplier_weights: Tuple[float] = None,
#     radiant_temperature_adjustment_weights: Tuple[float] = None,
# ) -> Typology:
#     """Combine multiple typologies into a single typology.

#     Args:
#         typologies (Tuple[Typology]):
#             A tuple of typologies to combine.
#         evaporative_cooling_effect_weights (Tuple[float], optional):
#             A tuple of weights to apply to the evaporative cooling effect
#             of each typology. Defaults to None.
#         wind_speed_multiplier_weights (Tuple[float], optional):
#             A tuple of weights to apply to the wind speed multiplier
#             of each typology. Defaults to None.
#         radiant_temperature_adjustment_weights (Tuple[float], optional):
#             A tuple of weights to apply to the radiant temperature adjustment
#             of each typology. Defaults to None.

#     Raises:
#         ValueError: If the weights do not sum to 1.

#     Returns:
#         Typology: A combined typology.
#     """

#     if evaporative_cooling_effect_weights is None:
#         evaporative_cooling_effect_weights = np.ones_like(typologies) * (
#             1 / len(typologies)
#         )
#     else:
#         if sum(evaporative_cooling_effect_weights) != 1:
#             raise ValueError("evaporative_cooling_effect_weights must sum to 1.")

#     if wind_speed_multiplier_weights is None:
#         wind_speed_multiplier_weights = np.ones_like(typologies) * (1 / len(typologies))
#     else:
#         if sum(wind_speed_multiplier_weights) != 1:
#             raise ValueError("wind_speed_multiplier_weights must sum to 1.")

#     if radiant_temperature_adjustment_weights is None:
#         radiant_temperature_adjustment_weights = np.ones_like(typologies) * (
#             1 / len(typologies)
#         )
#     else:
#         if sum(radiant_temperature_adjustment_weights) != 1:
#             raise ValueError("radiant_temperature_adjustment_weights must sum to 1.")

#     all_shelters = []
#     for typ in typologies:
#         all_shelters.extend(typ.Shelters)

#     ec_effect = np.average(
#         [i.EvaporativeCoolingEffect for i in typologies],
#         weights=evaporative_cooling_effect_weights,
#         axis=0,
#     )
#     ws_multiplier = np.average(
#         [i.WindSpeedMultiplier for i in typologies],
#         weights=wind_speed_multiplier_weights,
#         axis=0,
#     )
#     tr_adjustment = np.average(
#         [i.RadiantTemperatureAdjustment for i in typologies],
#         weights=radiant_temperature_adjustment_weights,
#         axis=0,
#     )

#     return Typology(
#         Name=" + ".join([i.Name for i in typologies]),
#         Shelters=all_shelters,
#         EvaporativeCoolingEffect=ec_effect,
#         WindSpeedMultiplier=ws_multiplier,
#         RadiantTemperatureAdjustment=tr_adjustment,
#     )


# class Typologies(Enum):
#     """A list of pre-defined Typology objects."""

#     OPENFIELD = Typology(
#         Name="Openfield",
#     )
#     ENCLOSED = Typology(
#         Name="Enclosed",
#         Shelters=[
#             Shelters.NORTH.value,
#             Shelters.EAST.value,
#             Shelters.SOUTH.value,
#             Shelters.WEST.value,
#             Shelters.OVERHEAD_LARGE.value,
#         ],
#     )
#     POROUS_ENCLOSURE = Typology(
#         Name="Porous enclosure",
#         Shelters=[
#             Shelters.NORTH.value.set_porosity(0.5),
#             Shelters.EAST.value.set_porosity(0.5),
#             Shelters.SOUTH.value.set_porosity(0.5),
#             Shelters.WEST.value.set_porosity(0.5),
#             Shelters.OVERHEAD_LARGE.value.set_porosity(0.5),
#         ],
#     )
#     SKY_SHELTER = Typology(
#         Name="Sky-shelter",
#         Shelters=[
#             Shelters.OVERHEAD_LARGE.value,
#         ],
#     )
#     FRITTED_SKY_SHELTER = Typology(
#         Name="Fritted sky-shelter",
#         Shelters=[
#             Shelters.OVERHEAD_LARGE.value.set_porosity(0.5),
#         ],
#     )
#     NEAR_WATER = Typology(
#         Name="Near water",
#         EvaporativeCoolingEffect=0.15,
#     )
#     MISTING = Typology(
#         Name="Misting",
#         EvaporativeCoolingEffect=0.3,
#     )
#     PDEC = Typology(
#         Name="PDEC",
#         EvaporativeCoolingEffect=0.7,
#     )
#     NORTH_SHELTER = Typology(
#         Name="North shelter",
#         Shelters=[
#             Shelters.NORTH.value,
#         ],
#     )
#     NORTHEAST_SHELTER = Typology(
#         Name="Northeast shelter", Shelters=[Shelters.NORTHEAST.value]
#     )
#     EAST_SHELTER = Typology(Name="East shelter", Shelters=[Shelters.EAST.value])
#     SOUTHEAST_SHELTER = Typology(
#         Name="Southeast shelter", Shelters=[Shelters.SOUTHEAST.value]
#     )
#     SOUTH_SHELTER = Typology(
#         Name="South shelter",
#         Shelters=[
#             Shelters.SOUTH.value,
#         ],
#     )
#     SOUTHWEST_SHELTER = Typology(
#         Name="Southwest shelter", Shelters=[Shelters.SOUTHWEST.value]
#     )
#     WEST_SHELTER = Typology(Name="West shelter", Shelters=[Shelters.WEST.value])
#     NORTHWEST_SHELTER = Typology(
#         Name="Northwest shelter", Shelters=[Shelters.NORTHWEST.value]
#     )
#     NORTH_SHELTER_WITH_CANOPY = Typology(
#         Name="North shelter with canopy",
#         Shelters=[
#             Shelters.NORTH.value,
#             Shelters.CANOPY_N_E_S_W.value,
#         ],
#     )
#     NORTHEAST_SHELTER_WITH_CANOPY = Typology(
#         Name="Northeast shelter with canopy",
#         Shelters=[
#             Shelters.NORTHEAST.value,
#             Shelters.CANOPY_NE_SE_SW_NW.value,
#         ],
#     )
#     EAST_SHELTER_WITH_CANOPY = Typology(
#         Name="East shelter with canopy",
#         Shelters=[
#             Shelters.EAST.value,
#             Shelters.CANOPY_N_E_S_W.value,
#         ],
#     )
#     SOUTHEAST_SHELTER_WITH_CANOPY = Typology(
#         Name="Southeast shelter with canopy",
#         Shelters=[
#             Shelters.SOUTHEAST.value,
#             Shelters.CANOPY_NE_SE_SW_NW.value,
#         ],
#     )
#     SOUTH_SHELTER_WITH_CANOPY = Typology(
#         Name="South shelter with canopy",
#         Shelters=[
#             Shelters.SOUTH.value,
#             Shelters.CANOPY_N_E_S_W.value,
#         ],
#     )
#     SOUTHWEST_SHELTER_WITH_CANOPY = Typology(
#         Name="Southwest shelter with canopy",
#         Shelters=[
#             Shelters.SOUTHWEST.value,
#             Shelters.CANOPY_NE_SE_SW_NW.value,
#         ],
#     )
#     WEST_SHELTER_WITH_CANOPY = Typology(
#         Name="West shelter with canopy",
#         Shelters=[
#             Shelters.WEST.value,
#             Shelters.CANOPY_N_E_S_W.value,
#         ],
#     )
#     NORTHWEST_SHELTER_WITH_CANOPY = Typology(
#         Name="Northwest shelter with canopy",
#         Shelters=[
#             Shelters.NORTHWEST.value,
#             Shelters.CANOPY_NE_SE_SW_NW.value,
#         ],
#     )

#     NORTHSOUTH_LINEAR_SHELTER = Typology(
#         Name="North-south linear overhead shelter",
#         Shelters=[
#             Shelters.NORTH_SOUTH_LINEAR.value,
#         ],
#     )
#     NORTHEAST_SOUTHWEST_LINEAR_SHELTER = Typology(
#         Name="Northeast-southwest linear overhead shelter",
#         Shelters=[
#             Shelters.NORTHEAST_SOUTHWEST_LINEAR.value,
#         ],
#     )
#     EAST_WEST_LINEAR_SHELTER = Typology(
#         Name="East-west linear overhead shelter",
#         Shelters=[
#             Shelters.EAST_WEST_LINEAR.value,
#         ],
#     )
#     NORTHWEST_SOUTHEAST_LINEAR_SHELTER = Typology(
#         Name="Northwest-southeast linear overhead shelter",
#         Shelters=[
#             Shelters.NORTHWEST_SOUTHEAST_LINEAR.value,
#         ],
#     )

#     def to_json(self) -> str:
#         """Convert the current typology into its BHoM JSON string format."""
#         return self.value.to_json()
