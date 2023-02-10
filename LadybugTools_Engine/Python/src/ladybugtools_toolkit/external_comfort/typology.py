from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybug_comfort.collection.pmv import PMV
from ladybug_comfort.collection.utci import UTCI

from ..bhomutil.analytics import CONSOLE_LOGGER
from ..bhomutil.bhom_object import BHoMObject, bhom_dict_to_dict
from ..helpers import decay_rate_smoother
from ..ladybug_extension.datacollection import from_series, to_series
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
    """An external comfort typology, described by shelters and a proportion
        of evaporative cooling.

    Args:
        name (str):
            The name of the external comfort typology.
        shelters (List[Shelter], optional):
            A list of shelters modifying exposure to the elements.
            Defaults to None.
        evaporative_cooling_effectiveness (Union[float, List[float]), optional):
            An amount of evaporative cooling to add to results calculated by
            this typology. Defaults to 0. Can also be a list of 8760 values.
        wind_speed_adjustment (float, optional):
            A factor to multiply wind speed by. Defaults to 1.

    Returns:
        Typology: An external comfort typology.
    """

    name: str = field(init=True, compare=True, repr=True)
    shelters: List[Shelter] = field(
        init=True, compare=True, repr=False, default_factory=list
    )
    evaporative_cooling_effectiveness: Union[float, List[float]] = field(
        init=True, compare=True, repr=False, default=0
    )
    wind_speed_adjustment: float = field(init=True, compare=True, repr=False, default=1)

    _t: str = field(
        init=False, compare=True, repr=False, default="BH.oM.LadybugTools.Typology"
    )

    def __post_init__(self):

        if self.wind_speed_adjustment < 0:
            raise ValueError("The wind_speed_adjustment factor cannot be less than 0.")
        if isinstance(self.evaporative_cooling_effectiveness, (float, int)):
            if (
                self.evaporative_cooling_effectiveness < 0
                or self.evaporative_cooling_effectiveness > 1
            ):
                raise ValueError("Evaporative cooling effect must be between 0 and 1.")
        else:
            if len(self.evaporative_cooling_effectiveness) != 8760:
                raise ValueError(
                    "Evaporative cooling effect can only currently be either a single value applied across the entire year, or a list of 8760 values."
                )
            if (
                min(self.evaporative_cooling_effectiveness) < 0
                or max(self.evaporative_cooling_effectiveness) > 1
            ):
                raise ValueError("Evaporative cooling effect must be between 0 and 1.")

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
            evaporative_cooling_effectiveness=sanitised_dict[
                "evaporative_cooling_effectiveness"
            ],
            wind_speed_adjustment=sanitised_dict["wind_speed_adjustment"],
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
            typology (Typology): A Typology object.
            epw (EPW): A ladybug EPW object.

        Returns:
            HourlyContinuousCollection: The effective DBT following application of any evaporative
                cooling effects.
        """
        if isinstance(self.evaporative_cooling_effectiveness, (float, int)):
            return evaporative_cooling_effect_collection(
                epw, self.evaporative_cooling_effectiveness
            )[0]
        dbt_evap, _ = np.array(
            [
                evaporative_cooling_effect(dbt, rh, evap_clg, ap)
                for dbt, rh, evap_clg, ap in list(
                    zip(
                        *[
                            epw.dry_bulb_temperature,
                            epw.relative_humidity,
                            self.evaporative_cooling_effectiveness,
                            epw.atmospheric_station_pressure,
                        ]
                    )
                )
            ]
        ).T
        return epw.dry_bulb_temperature.get_aligned_collection(dbt_evap)

    def relative_humidity(self, epw: EPW) -> HourlyContinuousCollection:
        """Get the effective RH for the given EPW file for this Typology.

        Args:
            typology (Typology): A Typology object.
            epw (EPW): A ladybug EPW object.

        Returns:
            HourlyContinuousCollection: The effective RH following application of any evaporative
                cooling effects.
        """
        if isinstance(self.evaporative_cooling_effectiveness, (float, int)):
            return evaporative_cooling_effect_collection(
                epw, self.evaporative_cooling_effectiveness
            )[1]
        _, rh_evap = np.array(
            [
                evaporative_cooling_effect(dbt, rh, evap_clg, ap)
                for dbt, rh, evap_clg, ap in list(
                    zip(
                        *[
                            epw.dry_bulb_temperature,
                            epw.relative_humidity,
                            self.evaporative_cooling_effectiveness,
                            epw.atmospheric_station_pressure,
                        ]
                    )
                )
            ]
        ).T
        return epw.relative_humidity.get_aligned_collection(rh_evap)

    def wind_speed(self, epw: EPW) -> HourlyContinuousCollection:
        """Calculate wind speed when subjected to a set of shelters.

        Args:
            typology (Typology): A Typology object.
            epw (EPW): The input EPW.

        Returns:
            HourlyContinuousCollection: The resultant wind-speed.
        """

        if len(self.shelters) == 0:
            return epw.wind_speed * self.wind_speed_adjustment

        if self.wind_speed_adjustment == 0:
            epw.wind_speed.get_aligned_collection(0)

        return (epw.wind_speed * self.wind_speed_adjustment).get_aligned_collection(
            annual_effective_wind_speed(self.shelters, epw)
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

        shaded_mrt = to_series(simulation_result.shaded_mean_radiant_temperature)
        unshaded_mrt = to_series(simulation_result.unshaded_mean_radiant_temperature)

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

        # apply an exponentially weighted moving average to account for transition between shaded/unshaded periods on surrounding surface temperatures
        mrt_series = decay_rate_smoother(
            mrt_series, difference_threshold=-10, transition_window=4, ewm_span=1.25
        )

        return from_series(mrt_series)

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
            "While it is possible to call from a Typology object, the recommended method of calling the UTCI historgam is from an ExternalComfort object."
        )
        return utci_heatmap_histogram(
            self.universal_thermal_climate_index(res), self.name
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
        evaporative_cooling_effectiveness=0.15,
    )
    MISTING = Typology(
        name="Misting",
        evaporative_cooling_effectiveness=0.3,
    )
    PDEC = Typology(
        name="PDEC",
        evaporative_cooling_effectiveness=0.7,
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
