from __future__ import annotations

import json
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
from .moisture import evaporative_cooling_effect_collection
from .shelter import Shelter
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
        evaporative_cooling_effectiveness (float, optional):
            An amount of evaporative cooling to add to results calculated by
            this typology. Defaults to 0.
        wind_speed_adjustment (float, optional):
            A factor to multiply wind speed by. Defaults to 1.

    Returns:
        Typology: An external comfort typology.
    """

    name: str = field(init=True, compare=True, repr=True)
    shelters: List[Shelter] = field(
        init=True, compare=True, repr=True, default_factory=list
    )
    evaporative_cooling_effectiveness: float = field(
        init=True, compare=True, repr=True, default=0
    )
    wind_speed_adjustment: float = field(init=True, compare=True, repr=True, default=1)

    _t: str = field(
        init=False, compare=True, repr=False, default="BH.oM.LadybugTools.Typology"
    )

    def __post_init__(self):
        if self.wind_speed_adjustment < 0:
            raise ValueError("The wind_speed_adjustment factor cannot be less than 0.")

        if self.shelters:
            try:
                if Shelter.any_shelters_overlap.__wrapped__(self.shelters):
                    raise ValueError("Shelters overlap")
            except AttributeError:
                if Shelter.any_shelters_overlap(self.shelters):
                    raise ValueError(  # pylint: disable=raise-missing-from
                        "Shelters overlap"
                    )

        if (
            self.evaporative_cooling_effectiveness < 0
            or self.evaporative_cooling_effectiveness > 1
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
        return Shelter.sky_exposure(self.shelters)

    def sun_exposure(self, epw: EPW) -> List[float]:
        """Direct access to "sun_exposure" method for this typology object."""
        return Shelter.sun_exposure(self.shelters, epw)

    def dry_bulb_temperature(self, epw: EPW) -> HourlyContinuousCollection:
        """Get the effective DBT for the given EPW file for this Typology.

        Args:
            typology (Typology): A Typology object.
            epw (EPW): A ladybug EPW object.

        Returns:
            HourlyContinuousCollection: The effective DBT following application of any evaporative
                cooling effects.
        """
        # TODO - add ability to specify evap clg magnitude via schedule
        return evaporative_cooling_effect_collection(
            epw, self.evaporative_cooling_effectiveness
        )[0]

    def relative_humidity(self, epw: EPW) -> HourlyContinuousCollection:
        """Get the effective RH for the given EPW file for this Typology.

        Args:
            typology (Typology): A Typology object.
            epw (EPW): A ladybug EPW object.

        Returns:
            HourlyContinuousCollection: The effective RH following application of any evaporative
                cooling effects.
        """
        return evaporative_cooling_effect_collection(
            epw, self.evaporative_cooling_effectiveness
        )[1]

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

        collections = []
        for shelter in self.shelters:
            collections.append(to_series(shelter.effective_wind_speed(epw)))
        return from_series(
            pd.concat(collections, axis=1).min(axis=1).rename("Wind Speed (m/s)")
            * self.wind_speed_adjustment
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
        mrts = []
        for hour in range(8760):
            if daytime[hour]:
                mrts.append(
                    np.interp(
                        _sun_exposure[hour],
                        [0, 1],
                        [shaded_mrt[hour], unshaded_mrt[hour]],
                    )
                )
            else:
                mrts.append(
                    np.interp(
                        self.sky_exposure(),
                        [0, 1],
                        [shaded_mrt[hour], unshaded_mrt[hour]],
                    )
                )

        # Fill any gaps where sun-visible/sun-occluded values are missing, and apply an
        # exponentially weighted moving average to account for transition between shaded/unshaded
        # periods.
        mrt_series = pd.Series(
            mrts, index=shaded_mrt.index, name=shaded_mrt.name
        ).interpolate()

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


class Typologies(Enum):
    """A list of pre-defined Typology objects."""

    OPENFIELD = Typology(
        name="Openfield",
        evaporative_cooling_effectiveness=0,
        shelters=[],
        wind_speed_adjustment=1,
    )
    ENCLOSED = Typology(
        name="Enclosed",
        evaporative_cooling_effectiveness=0,
        shelters=[
            Shelter(
                altitude_range=[0, 90],
                azimuth_range=[0, 360],
                wind_porosity=0,
                radiation_porosity=0,
            )
        ],
        wind_speed_adjustment=1,
    )
    POROUS_ENCLOSURE = Typology(
        name="Porous enclosure",
        evaporative_cooling_effectiveness=0,
        shelters=[
            Shelter(
                altitude_range=[0, 90],
                azimuth_range=[0, 360],
                wind_porosity=0,
                radiation_porosity=0.5,
            )
        ],
        wind_speed_adjustment=1,
    )
    SKY_SHELTER = Typology(
        name="Sky-shelter",
        evaporative_cooling_effectiveness=0,
        shelters=[
            Shelter(
                altitude_range=[45, 90],
                azimuth_range=[0, 360],
                wind_porosity=0,
                radiation_porosity=0,
            )
        ],
        wind_speed_adjustment=1,
    )
    FRITTED_SKY_SHELTER = Typology(
        name="Fritted sky-shelter",
        evaporative_cooling_effectiveness=0,
        shelters=[
            Shelter(
                altitude_range=[45, 90],
                azimuth_range=[0, 360],
                wind_porosity=0.5,
                radiation_porosity=0.5,
            ),
        ],
        wind_speed_adjustment=1,
    )
    NEAR_WATER = Typology(
        name="Near water",
        evaporative_cooling_effectiveness=0.15,
        shelters=[
            Shelter(
                altitude_range=[0, 0],
                azimuth_range=[0, 0],
                radiation_porosity=1,
                wind_porosity=1,
            ),
        ],
        wind_speed_adjustment=1,
    )
    MISTING = Typology(
        name="Misting",
        evaporative_cooling_effectiveness=0.3,
        shelters=[
            Shelter(
                altitude_range=[0, 0],
                azimuth_range=[0, 0],
                radiation_porosity=1,
                wind_porosity=1,
            ),
        ],
        wind_speed_adjustment=1,
    )
    PDEC = Typology(
        name="PDEC",
        evaporative_cooling_effectiveness=0.7,
        shelters=[
            Shelter(
                altitude_range=[0, 0],
                azimuth_range=[0, 0],
                radiation_porosity=1,
                wind_porosity=1,
            ),
        ],
        wind_speed_adjustment=1,
    )
    NORTH_SHELTER = Typology(
        name="North shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 70],
                azimuth_range=[337.5, 22.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    NORTHEAST_SHELTER = Typology(
        name="Northeast shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 70],
                azimuth_range=[22.5, 67.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    EAST_SHELTER = Typology(
        name="East shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 70],
                azimuth_range=[67.5, 112.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    SOUTHEAST_SHELTER = Typology(
        name="Southeast shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 70],
                azimuth_range=[112.5, 157.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    SOUTH_SHELTER = Typology(
        name="South shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 70],
                azimuth_range=[157.5, 202.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    SOUTHWEST_SHELTER = Typology(
        name="Southwest shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 70],
                azimuth_range=[202.5, 247.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    WEST_SHELTER = Typology(
        name="West shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 70],
                azimuth_range=[247.5, 292.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    NORTHWEST_SHELTER = Typology(
        name="Northwest shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 70],
                azimuth_range=[292.5, 337.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    NORTH_SHELTER_WITH_CANOPY = Typology(
        name="North shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 90],
                azimuth_range=[337.5, 22.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    NORTHEAST_SHELTER_WITH_CANOPY = Typology(
        name="Northeast shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 90],
                azimuth_range=[22.5, 67.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    EAST_SHELTER_WITH_CANOPY = Typology(
        name="East shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 90],
                azimuth_range=[67.5, 112.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    SOUTHEAST_SHELTER_WITH_CANOPY = Typology(
        name="Southeast shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 90],
                azimuth_range=[112.5, 157.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    SOUTH_SHELTER_WITH_CANOPY = Typology(
        name="South shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 90],
                azimuth_range=[157.5, 202.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    SOUTHWEST_SHELTER_WITH_CANOPY = Typology(
        name="Southwest shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 90],
                azimuth_range=[202.5, 247.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    WEST_SHELTER_WITH_CANOPY = Typology(
        name="West shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 90],
                azimuth_range=[247.5, 292.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    NORTHWEST_SHELTER_WITH_CANOPY = Typology(
        name="Northwest shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 90],
                azimuth_range=[292.5, 337.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    EAST_WEST_SHELTER = Typology(
        name="East-west shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 70],
                azimuth_range=[67.5, 112.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
            Shelter(
                altitude_range=[0, 70],
                azimuth_range=[247.5, 292.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )
    EAST_WEST_SHELTER_WITH_CANOPY = Typology(
        name="East-west shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(
                altitude_range=[0, 90],
                azimuth_range=[67.5, 112.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
            Shelter(
                altitude_range=[0, 90],
                azimuth_range=[247.5, 292.5],
                radiation_porosity=0,
                wind_porosity=0,
            ),
        ],
        wind_speed_adjustment=1,
    )

    def to_json(self) -> str:
        """Convert the current typology into its BHoM JSON string format."""
        return self.value.to_json()
