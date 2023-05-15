from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybug_comfort.collection.pmv import PMV
from ladybug_comfort.collection.utci import UTCI

from ..bhomutil.analytics import CONSOLE_LOGGER
from ..bhomutil.bhom_object import BHoMObject, bhom_dict_to_dict
from ..helpers import decay_rate_smoother
from ..ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)
from ..plot.utci_heatmap_histogram import utci_heatmap_histogram
from .moisture import evaporative_cooling_effect, evaporative_cooling_effect_collection
from .shelter import (
    Shelter,
    Shelters,
    annual_effective_wind_speed,
    annual_sun_exposure,
    sky_exposure,
)
from .simulate import SimulationResult


@dataclass(init=True, repr=True, eq=True)
class Typology(BHoMObject):
    """An external comfort typology, describing the context in which thermal comfort will be calculated.

    Args:
        name (str):
            The name of the external comfort typology.
        shelters (List[Shelter], optional):
            A list of shelters modifying exposure to the elements.
            Defaults to None.
        evaporative_cooling_effect (Union[float, List[float]), optional):
            An amount of evaporative cooling to add to results calculated by
            this typology. Defaults to 0. Can also be a list of 8760 values.
        wind_speed_multiplier (float, optional):
            A factor to multiply wind speed by. Defaults to 1. Can be used to
            account for wind speed reduction due to sheltering not accounted
            for by shelter objects, or to approximate effects of acceleration.
        radiant_temperature_adjustment (Union[float, List[float]], optional):
            A change in MRT to be applied. Defaults to 0. Can also be a
            list of 8760 values. A positive value will increase the
            MRT and a negative value will decrease it.

    Returns:
        Typology: An external comfort typology.
    """

    name: str = field(init=True, compare=True, repr=True)
    shelters: List[Shelter] = field(
        init=True, compare=True, repr=False, default_factory=list
    )
    evaporative_cooling_effect: Union[float, List[float]] = field(
        init=True, compare=True, repr=False, default=0
    )
    wind_speed_multiplier: Union[float, List[float]] = field(
        init=True, compare=True, repr=False, default=1
    )
    radiant_temperature_adjustment: Union[float, List[float]] = field(
        init=True, compare=True, repr=False, default=0
    )

    _t: str = field(
        init=False, compare=True, repr=False, default="BH.oM.LadybugTools.Typology"
    )

    def __post_init__(self):
        if isinstance(self.wind_speed_multiplier, (float, int)):
            self.wind_speed_multiplier = np.ones(8760) * self.wind_speed_multiplier
        else:
            self.wind_speed_multiplier = np.array(self.wind_speed_multiplier)
        if min(self.wind_speed_multiplier) < 0:
            raise ValueError("The wind_speed_adjustment factor cannot be less than 0.")
        if len(self.wind_speed_multiplier) != 8760:
            raise ValueError(
                "Wind speed multiplier can only currently be either a single value applied across the entire year, or a list of 8760 values."
            )

        if isinstance(self.evaporative_cooling_effect, (float, int)):
            self.evaporative_cooling_effect = (
                np.ones(8760) * self.evaporative_cooling_effect
            )
        else:
            self.evaporative_cooling_effect = np.array(self.evaporative_cooling_effect)
        if (
            min(self.evaporative_cooling_effect) < 0
            or max(self.evaporative_cooling_effect) > 1
        ):
            raise ValueError("Evaporative cooling effect must be between 0 and 1.")
        if len(self.evaporative_cooling_effect) != 8760:
            raise ValueError(
                "Evaporative cooling effect can only currently be either a single value applied across the entire year, or a list of 8760 values."
            )

        if isinstance(self.radiant_temperature_adjustment, (float, int)):
            self.radiant_temperature_adjustment = (
                np.ones(8760) * self.radiant_temperature_adjustment
            )
        else:
            self.radiant_temperature_adjustment = np.array(
                self.radiant_temperature_adjustment
            )
        if len(self.radiant_temperature_adjustment) != 8760:
            raise ValueError(
                "Radiant temperature adjustment can only currently be either a single value applied across the entire year, or a list of 8760 values."
            )

        # wrap methods within this class
        super().__post_init__()

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> Typology:
        """Create this object from a dictionary."""

        sanitised_dict = bhom_dict_to_dict(dictionary)
        sanitised_dict.pop("_t", None)

        # handle shelter object conversions
        shelters = []
        for obj in sanitised_dict["shelters"]:
            if isinstance(obj, BHoMObject):
                shelters.append(obj)
            elif isinstance(obj, dict):
                shelters.append(Shelter.from_dict(obj))
            elif isinstance(obj, str):
                shelters.append(Shelter.from_json(obj))
            else:
                raise ValueError(
                    "Objects in the in input dictionary are not of a recognised type."
                )
        sanitised_dict["shelters"] = shelters

        return cls(
            name=sanitised_dict["name"],
            shelters=sanitised_dict["shelters"],
            evaporative_cooling_effect=sanitised_dict["evaporative_cooling_effect"],
            wind_speed_multiplier=sanitised_dict["wind_speed_multiplier"],
            radiant_temperature_adjustment=sanitised_dict[
                "radiant_temperature_adjustment"
            ],
        )

    @classmethod
    def from_json(cls, json_string: str) -> Typology:
        """Create this object from a JSON string."""

        dictionary = json.loads(json_string)

        return cls.from_dict(dictionary)

    def sky_exposure(self) -> float:
        """Direct access to "sky_exposure" method for this typology object."""
        return sky_exposure(self.shelters, include_radiation_porosity=True)

    def sun_exposure(self, epw: EPW) -> List[float]:
        """Direct access to "sun_exposure" method for this typology object."""
        return annual_sun_exposure(self.shelters, epw, include_radiation_porosity=True)

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

    def wind_speed(self, epw: EPW) -> HourlyContinuousCollection:
        """Calculate wind speed when subjected to a set of shelters.

        Args:
            epw (EPW): The input EPW.

        Returns:
            HourlyContinuousCollection: The resultant wind-speed.
        """

        ws = epw.wind_speed

        if len(self.shelters) == 0:
            return ws.get_aligned_collection(
                np.array(ws.values) * self.wind_speed_multiplier
            )

        # adjust to 0 if multiplier is 0
        if sum(self.wind_speed_multiplier) == 0:
            return ws.get_aligned_collection(0)

        # adjust ws based on shelter configuration
        wind_speed_pre_multiplier = epw.wind_speed.get_aligned_collection(
            annual_effective_wind_speed(self.shelters, epw)
        )

        # adjust ws based on multiplier (collection)
        return ws.get_aligned_collection(
            np.array(wind_speed_pre_multiplier.values) * self.wind_speed_multiplier
        )

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

        daytime = np.array(
            [i > 0 for i in simulation_result.epw.global_horizontal_radiation]
        )
        _sun_exposure = self.sun_exposure(simulation_result.epw)
        _sky_exposure = self.sky_exposure()
        mrts = []
        for sun_exp, sun_up, shaded, unshaded in list(
            zip(*[_sun_exposure, daytime, shaded_mrt.values, unshaded_mrt.values])
        ):
            if sun_up:
                mrts.append(np.interp(sun_exp, [0, 1], [shaded, unshaded]))
            else:
                mrts.append(np.interp(_sky_exposure, [0, 1], [shaded, unshaded]))

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

        # apply radiant temperature adjustment if given
        return collection_from_series(mrt_series + self.radiant_temperature_adjustment)

    def universal_thermal_climate_index(
        self, simulation_result: SimulationResult, return_comfort_obj: bool = False
    ) -> Union[UTCI, HourlyContinuousCollection]:
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

        CONSOLE_LOGGER.info(
            f"[{simulation_result.identifier} - {self.name}] - Calculating universal thermal climate index"
        )

        if return_comfort_obj:
            return utci

        return utci.universal_thermal_climate_index

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

    def plot_utci_hist(self, res: SimulationResult) -> None:
        """Convenience method to plot UTCI histogram directly from a typology."""
        warnings.warn(
            "While it is possible to call from a Typology object, the recommended method of calling the UTCI histogram is from an ExternalComfort object."
        )
        return utci_heatmap_histogram(
            self.universal_thermal_climate_index(res), self.name
        )


def combine_typologies(
    typologies: Tuple[Typology],
    evaporative_cooling_effect_weights: Tuple[float] = None,
    wind_speed_multiplier_weights: Tuple[float] = None,
    radiant_temperature_adjustment_weights: Tuple[float] = None,
) -> Typology:
    """Combine multiple typologies into a single typology.

    Args:
        typologies (Tuple[Typology]):
            A tuple of typologies to combine.
        evaporative_cooling_effect_weights (Tuple[float], optional):
            A tuple of weights to apply to the evaporative cooling effect
            of each typology. Defaults to None.
        wind_speed_multiplier_weights (Tuple[float], optional):
            A tuple of weights to apply to the wind speed multiplier
            of each typology. Defaults to None.
        radiant_temperature_adjustment_weights (Tuple[float], optional):
            A tuple of weights to apply to the radiant temperature adjustment
            of each typology. Defaults to None.

    Raises:
        ValueError: If the weights do not sum to 1.

    Returns:
        Typology: A combined typology.
    """

    if evaporative_cooling_effect_weights is None:
        evaporative_cooling_effect_weights = np.ones_like(typologies) * (
            1 / len(typologies)
        )
    else:
        if sum(evaporative_cooling_effect_weights) != 1:
            raise ValueError("evaporative_cooling_effect_weights must sum to 1.")

    if wind_speed_multiplier_weights is None:
        wind_speed_multiplier_weights = np.ones_like(typologies) * (1 / len(typologies))
    else:
        if sum(wind_speed_multiplier_weights) != 1:
            raise ValueError("wind_speed_multiplier_weights must sum to 1.")

    if radiant_temperature_adjustment_weights is None:
        radiant_temperature_adjustment_weights = np.ones_like(typologies) * (
            1 / len(typologies)
        )
    else:
        if sum(radiant_temperature_adjustment_weights) != 1:
            raise ValueError("radiant_temperature_adjustment_weights must sum to 1.")

    all_shelters = []
    for typ in typologies:
        all_shelters.extend(typ.shelters)

    ec_effect = np.average(
        [i.evaporative_cooling_effect for i in typologies],
        weights=evaporative_cooling_effect_weights,
        axis=0,
    )
    ws_multiplier = np.average(
        [i.wind_speed_multiplier for i in typologies],
        weights=wind_speed_multiplier_weights,
        axis=0,
    )
    tr_adjustment = np.average(
        [i.radiant_temperature_adjustment for i in typologies],
        weights=radiant_temperature_adjustment_weights,
        axis=0,
    )

    return Typology(
        name=" + ".join([i.name for i in typologies]),
        shelters=all_shelters,
        evaporative_cooling_effect=ec_effect,
        wind_speed_multiplier=ws_multiplier,
        radiant_temperature_adjustment=tr_adjustment,
    )


class Typologies(Enum):
    """A list of pre-defined Typology objects."""

    OPENFIELD = Typology(
        name="Openfield",
    )
    ENCLOSED = Typology(
        name="Enclosed",
        shelters=[
            Shelters.NORTH.value,
            Shelters.EAST.value,
            Shelters.SOUTH.value,
            Shelters.WEST.value,
            Shelters.OVERHEAD_LARGE.value,
        ],
    )
    POROUS_ENCLOSURE = Typology(
        name="Porous enclosure",
        shelters=[
            Shelters.NORTH.value.set_porosity(0.5),
            Shelters.EAST.value.set_porosity(0.5),
            Shelters.SOUTH.value.set_porosity(0.5),
            Shelters.WEST.value.set_porosity(0.5),
            Shelters.OVERHEAD_LARGE.value.set_porosity(0.5),
        ],
    )
    SKY_SHELTER = Typology(
        name="Sky-shelter",
        shelters=[
            Shelters.OVERHEAD_LARGE.value,
        ],
    )
    FRITTED_SKY_SHELTER = Typology(
        name="Fritted sky-shelter",
        shelters=[
            Shelters.OVERHEAD_LARGE.value.set_porosity(0.5),
        ],
    )
    NEAR_WATER = Typology(
        name="Near water",
        evaporative_cooling_effect=0.15,
    )
    MISTING = Typology(
        name="Misting",
        evaporative_cooling_effect=0.3,
    )
    PDEC = Typology(
        name="PDEC",
        evaporative_cooling_effect=0.7,
    )
    NORTH_SHELTER = Typology(
        name="North shelter",
        shelters=[
            Shelters.NORTH.value,
        ],
    )
    NORTHEAST_SHELTER = Typology(
        name="Northeast shelter", shelters=[Shelters.NORTHEAST.value]
    )
    EAST_SHELTER = Typology(name="East shelter", shelters=[Shelters.EAST.value])
    SOUTHEAST_SHELTER = Typology(
        name="Southeast shelter", shelters=[Shelters.SOUTHEAST.value]
    )
    SOUTH_SHELTER = Typology(
        name="South shelter",
        shelters=[
            Shelters.SOUTH.value,
        ],
    )
    SOUTHWEST_SHELTER = Typology(
        name="Southwest shelter", shelters=[Shelters.SOUTHWEST.value]
    )
    WEST_SHELTER = Typology(name="West shelter", shelters=[Shelters.WEST.value])
    NORTHWEST_SHELTER = Typology(
        name="Northwest shelter", shelters=[Shelters.NORTHWEST.value]
    )
    NORTH_SHELTER_WITH_CANOPY = Typology(
        name="North shelter with canopy",
        shelters=[
            Shelters.NORTH.value,
            Shelters.CANOPY_N_E_S_W.value,
        ],
    )
    NORTHEAST_SHELTER_WITH_CANOPY = Typology(
        name="Northeast shelter with canopy",
        shelters=[
            Shelters.NORTHEAST.value,
            Shelters.CANOPY_NE_SE_SW_NW.value,
        ],
    )
    EAST_SHELTER_WITH_CANOPY = Typology(
        name="East shelter with canopy",
        shelters=[
            Shelters.EAST.value,
            Shelters.CANOPY_N_E_S_W.value,
        ],
    )
    SOUTHEAST_SHELTER_WITH_CANOPY = Typology(
        name="Southeast shelter with canopy",
        shelters=[
            Shelters.SOUTHEAST.value,
            Shelters.CANOPY_NE_SE_SW_NW.value,
        ],
    )
    SOUTH_SHELTER_WITH_CANOPY = Typology(
        name="South shelter with canopy",
        shelters=[
            Shelters.SOUTH.value,
            Shelters.CANOPY_N_E_S_W.value,
        ],
    )
    SOUTHWEST_SHELTER_WITH_CANOPY = Typology(
        name="Southwest shelter with canopy",
        shelters=[
            Shelters.SOUTHWEST.value,
            Shelters.CANOPY_NE_SE_SW_NW.value,
        ],
    )
    WEST_SHELTER_WITH_CANOPY = Typology(
        name="West shelter with canopy",
        shelters=[
            Shelters.WEST.value,
            Shelters.CANOPY_N_E_S_W.value,
        ],
    )
    NORTHWEST_SHELTER_WITH_CANOPY = Typology(
        name="Northwest shelter with canopy",
        shelters=[
            Shelters.NORTHWEST.value,
            Shelters.CANOPY_NE_SE_SW_NW.value,
        ],
    )

    NORTHSOUTH_LINEAR_SHELTER = Typology(
        name="North-south linear overhead shelter",
        shelters=[
            Shelters.NORTH_SOUTH_LINEAR.value,
        ],
    )
    NORTHEAST_SOUTHWEST_LINEAR_SHELTER = Typology(
        name="Northeast-southwest linear overhead shelter",
        shelters=[
            Shelters.NORTHEAST_SOUTHWEST_LINEAR.value,
        ],
    )
    EAST_WEST_LINEAR_SHELTER = Typology(
        name="East-west linear overhead shelter",
        shelters=[
            Shelters.EAST_WEST_LINEAR.value,
        ],
    )
    NORTHWEST_SOUTHEAST_LINEAR_SHELTER = Typology(
        name="Northwest-southeast linear overhead shelter",
        shelters=[
            Shelters.NORTHWEST_SOUTHEAST_LINEAR.value,
        ],
    )

    def to_json(self) -> str:
        """Convert the current typology into its BHoM JSON string format."""
        return self.value.to_json()
