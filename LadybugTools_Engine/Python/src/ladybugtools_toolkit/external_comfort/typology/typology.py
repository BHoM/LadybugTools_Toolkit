from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.moisture.evaporative_cooling_effect_collection import (
    evaporative_cooling_effect_collection,
)
from ladybugtools_toolkit.external_comfort.shelter.any_shelters_overlap import (
    any_shelters_overlap,
)
from ladybugtools_toolkit.external_comfort.shelter.shelter import Shelter
from ladybugtools_toolkit.external_comfort.shelter.sky_exposure import sky_exposure
from ladybugtools_toolkit.external_comfort.shelter.sun_exposure import sun_exposure
from ladybugtools_toolkit.external_comfort.simulate.simulation_result import (
    SimulationResult,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.from_series import (
    from_series,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.ladybug_extension.helpers.decay_rate_smoother import (
    decay_rate_smoother,
)


class Typology:
    """An external comfort typology, described by shelters and a proportion of evaporative
        cooling.

    Args:
        name (str): The name of the external comfort typology.
        shelters (List[Shelter], optional): A list of shelters modifying exposure to the
            elements. Defaults to None.
        evaporative_cooling_effect (float, optional): An amount of evaporative cooling to add to
            results calculated by this typology. Defaults to 0.

    Returns:
        Typology: An external comfort typology.
    """

    def __init__(
        self,
        name: str,
        shelters: List[Shelter] = None,
        evaporative_cooling_effectiveness: float = 0,
    ) -> Typology:

        self.name = name
        self.shelters = shelters
        self.evaporative_cooling_effect = evaporative_cooling_effectiveness

        if any_shelters_overlap(shelters):
            raise ValueError("Shelters overlap")

        if (
            evaporative_cooling_effectiveness < 0
            or evaporative_cooling_effectiveness > 1
        ):
            raise ValueError("Evaporative cooling effect must be between 0 and 1.")

    def __repr__(self):
        return f"{self.__class__.__name__} - {self.name}"

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        d = {
            "name": self.name,
            "shelters": self.shelters,
            "evaporative_cooling_effect": self.evaporative_cooling_effect,
        }
        return d

    def sky_exposure(self) -> float:
        """Direct access to "sky_exposure" method for this typology object."""
        return sky_exposure(self.shelters)

    def sun_exposure(self, epw: EPW) -> List[float]:
        """Direct access to "sun_exposure" method for this typology object."""
        return sun_exposure(self.shelters, epw)

    def dry_bulb_temperature(self, epw: EPW) -> HourlyContinuousCollection:
        """Get the effective DBT for the given EPW file for this Typology.

        Args:
            typology (Typology): A Typology object.
            epw (EPW): A ladybug EPW object.

        Returns:
            HourlyContinuousCollection: The effective DBT following application of any evaporative
                cooling effects.
        """
        return evaporative_cooling_effect_collection(
            epw, self.evaporative_cooling_effect
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
            epw, self.evaporative_cooling_effect
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
            return epw.wind_speed

        collections = []
        for shelter in self.shelters:
            collections.append(to_series(shelter.effective_wind_speed(epw)))
        return from_series(
            pd.concat(collections, axis=1).min(axis=1).rename("Wind Speed (m/s)")
        )

    def mean_radiant_temperature(
        self,
        simulation_result: SimulationResult,
    ) -> HourlyContinuousCollection:
        """Return the effective mean radiant temperature for the given typology following a
            simulation of the collections necessary to calculate this.

        Args:
            simulation_result (SimulationResult): A dictionary containing all collections describing
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
        # exponentially weighted moving average to account for transition betwen shaded/unshaded
        # periods.
        mrt_series = pd.Series(
            mrts, index=shaded_mrt.index, name=shaded_mrt.name
        ).interpolate()

        mrt_series = decay_rate_smoother(
            mrt_series, difference_threshold=-10, transition_window=4, ewm_span=1.25
        )

        return from_series(mrt_series)
