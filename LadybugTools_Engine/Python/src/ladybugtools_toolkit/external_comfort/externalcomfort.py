"""Derived methods for the ExternalComfort class."""

# pylint: disable=E0401
import copy
import warnings

# pylint: enable=E0401

import numpy as np

from ..bhom import decorator_factory
from ._externalcomfortbase import ExternalComfort
from ._shelterbase import Shelter
from ._typologybase import Typology


@decorator_factory()
def modify_external_comfort(
    external_comfort: ExternalComfort,
    additional_shelters: tuple[Shelter] = (),
    target_wind_speed: tuple[float] = (np.nan * np.empty(8760)).tolist(),
    evaporative_cooling_effect: tuple[float] = (np.nan * np.empty(8760)).tolist(),
    radiant_temperature_adjustment: tuple[float] = (np.nan * np.empty(8760)).tolist(),
    existing_shelters_wind_porosity: tuple[float] = (np.nan * np.empty(8760)).tolist(),
    existing_shelters_radiation_porosity: tuple[float] = (
        np.nan * np.empty(8760)
    ).tolist(),
) -> ExternalComfort:
    """Apply varying levels of additional measures to the insitu comfort model,
    taking into account any existing measures that are in place already.

    Args:
        external_comfort (ExternalComfort):
            An ExternalComfort object to modify.
        additional_shelters (tuple[Shelter], optional):
            Add more shelters to the existing External Comfort case.
        target_wind_speed (tuple[float], optional):
            Override the target wind speed of the current typology.
        evaporative_cooling_effect (tuple[float], optional):
            Override the effectivess of the current evaporative cooling.
        radiant_temperature_adjustment (tuple[float], optional):
            The amount of radiant cooling to apply to the MRT.
        existing_shelters_wind_porosity (tuple[float], optional):
            Override the wind porosity of the existing shelters.
        existing_shelters_radiation_porosity (tuple[float], optional):
            Override the radiation porosity of the existing shelters.

    Returns:
        ExternalComfort:
            A modified object!
    """

    # check if any changes are needed, and return original object if not
    if all(
        [
            not additional_shelters,
            all(np.isnan(target_wind_speed)),
            all(np.isnan(evaporative_cooling_effect)),
            all(np.isnan(radiant_temperature_adjustment)),
            all(np.isnan(existing_shelters_wind_porosity)),
            all(np.isnan(existing_shelters_radiation_porosity)),
        ]
    ):
        warnings.warn("No changes made to the input ExternalComfort object.")
        return external_comfort

    # modify existing shelters if necessary
    modified_shelters = []
    for shelter in external_comfort.typology.shelters:
        _shelter = copy.copy(shelter)
        _shelter.wind_porosity = np.where(
            ~np.isnan(existing_shelters_wind_porosity),
            existing_shelters_wind_porosity,
            _shelter.wind_porosity,
        )
        _shelter.radiation_porosity = np.where(
            ~np.isnan(existing_shelters_radiation_porosity),
            existing_shelters_radiation_porosity,
            _shelter.radiation_porosity,
        )
        modified_shelters.append(_shelter)
    modified_shelters.extend(additional_shelters)

    # construct name
    modified_name = (
        f"{external_comfort.typology.name} + modified"  # TODO - add more info
    )

    # construct new typology
    modified_typology = Typology(
        name=modified_name,
        shelters=modified_shelters,
        target_wind_speed=np.where(
            ~np.isnan(target_wind_speed),
            target_wind_speed,
            external_comfort.typology.target_wind_speed,
        ).tolist(),
        evaporative_cooling_effect=np.where(
            ~np.isnan(evaporative_cooling_effect),
            evaporative_cooling_effect,
            external_comfort.typology.evaporative_cooling_effect,
        ).tolist(),
        radiant_temperature_adjustment=np.where(
            ~np.isnan(radiant_temperature_adjustment),
            radiant_temperature_adjustment,
            external_comfort.typology.radiant_temperature_adjustment,
        ).tolist(),
    )
    return ExternalComfort(
        simulation_result=external_comfort.simulation_result,
        typology=modified_typology,
    )
