from __future__ import annotations

import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug_comfort.collection.utci import UTCI
from ladybug_extension.datacollection import from_series, to_series
from ladybug.sunpath import Sunpath

from external_comfort.evaporative_cooling import get_evaporative_cooled_dbt_rh
from external_comfort.openfield import Openfield
from external_comfort.shelter import Shelter, coincident_shelters


class Typology:
    def __init__(
        self,
        openfield: Openfield,
        name: str = "Openfield",
        evaporative_cooling_effectiveness: float = 0,
        wind_speed_multiplier: float = 1,
        shelters: List[Shelter] = [],
    ) -> Typology:
        """Class for defining a specific external comfort typology, and calculating the resultant thermal comfort values.

        Args:
            openfield (Openfield): An Openfield object.
            name (str, optional): A string for the name of the typology. Defaults to "Openfield".
            evaporative_cooling_effectiveness (float, optional): A float between 0 and 1 for the effectiveness of the contextual evaporative cooling modifying air temperature. Defaults to 0.
            shelters (List[ShelterNew], optional): A list ShelterNew objects defining the sheltered portions around the typology. Defaults to no shelters.
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

        if coincident_shelters(shelters):
            raise ValueError(f"shelters overlap")
        else:
            self.shelters = shelters

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, openfield={self.openfield}, evaporative_cooling_effectiveness={self.evaporative_cooling_effectiveness:0.2%}, shelters={[str(i) for i in self.shelters]})"

    def effective_sky_visibility(self) -> float:
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

    def effective_wind_speed(self) -> HourlyContinuousCollection:
        """Based on the shelters in-place, create a composity wind-speed collection affected by those shelters."""
        if len(self.shelters) == 0:
            return self.openfield.epw.wind_speed * self.wind_speed_multiplier

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

    def effective_dry_bulb_temperature(self) -> HourlyContinuousCollection:
        """Based on the evaporative cooling configuration, calculate the effective dry bulb temperature for each hour of the year."""
        return get_evaporative_cooled_dbt_rh(
            self.openfield.epw, self.evaporative_cooling_effectiveness
        )["dry_bulb_temperature"]

    def effective_relative_humidity(self) -> HourlyContinuousCollection:
        """Based on the evaporative cooling configuration, calculate the effective dry bulb temperature for each hour of the year."""
        return get_evaporative_cooled_dbt_rh(
            self.openfield.epw, self.evaporative_cooling_effectiveness
        )["relative_humidity"]

    def effective_mean_radiant_temperature(self) -> HourlyContinuousCollection:

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

        # TODO - ADD ROLLING WINDOW MEAN TO ACCOUNT FOR ONWARDS WEIGHTING OF CHANGING VALUES - IES DONT CENTRE IT ABOUT THE CURRENT VALUE
        mrt_series = pd.Series(
            mrts, index=shaded_mrt.index, name=shaded_mrt.name
        ).interpolate()

        return from_series(mrt_series)

    def universal_thermal_climate_index(self) -> HourlyContinuousCollection:
        """Return the effective UTCI for the given typology."""
        return UTCI(
            air_temperature=self.effective_dry_bulb_temperature(),
            rel_humidity=self.effective_relative_humidity(),
            rad_temperature=self.effective_mean_radiant_temperature(),
            wind_speed=self.effective_wind_speed(),
        ).universal_thermal_climate_index

    @property
    def description(self) -> str:

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
            return f"Fully exposed" + wind_adj
        elif (len(shelter_descriptions) == 0) and (
            self.evaporative_cooling_effectiveness != 0
        ):
            return (
                f"Fully exposed, with {self.evaporative_cooling_effectiveness:0.0%} effective evaporative cooling"
                + wind_adj
            )
        elif (len(shelter_descriptions) != 0) and (
            self.evaporative_cooling_effectiveness == 0
        ):
            return f"{' and '.join(shelter_descriptions).capitalize()}" + wind_adj
        else:
            return (
                f"{' and '.join(shelter_descriptions).capitalize()}, with {self.evaporative_cooling_effectiveness:0.0%} effective evaporative cooling"
                + wind_adj
            )


def create_typologies(
    epw: EPW,
    ground_material: Union[str, _EnergyMaterialOpaqueBase],
    shade_material: Union[str, _EnergyMaterialOpaqueBase],
) -> Dict[str, Typology]:
    """Create a dictionary of typologies for a given epw file, with all requisite simulations and calculations completed"""
    openfield = Openfield(epw, ground_material, shade_material, True)
    typologies = {
        "Openfield": Typology(
            openfield,
            name="Openfield",
            evaporative_cooling_effectiveness=0,
            shelters=[],
            wind_speed_multiplier=1,
        ),
        "Enclosed": Typology(
            openfield,
            name="Enclosed",
            evaporative_cooling_effectiveness=0,
            shelters=[
                Shelter(altitude_range=[0, 90], azimuth_range=[0, 360], porosity=0)
            ],
            wind_speed_multiplier=1,
        ),
        "PartiallyEnclosed": Typology(
            openfield,
            name="PartiallyEnclosed",
            evaporative_cooling_effectiveness=0,
            shelters=[
                Shelter(altitude_range=[0, 90], azimuth_range=[0, 360], porosity=0.5)
            ],
            wind_speed_multiplier=1,
        ),
        "SkyShelter": Typology(
            openfield,
            name="SkyShelter",
            evaporative_cooling_effectiveness=0,
            shelters=[
                Shelter(altitude_range=[45, 90], azimuth_range=[0, 360], porosity=0)
            ],
            wind_speed_multiplier=1,
        ),
        "FrittedSkyShelter": Typology(
            openfield,
            name="FrittedSkyShelter",
            evaporative_cooling_effectiveness=0,
            shelters=[
                Shelter(altitude_range=[45, 90], azimuth_range=[0, 360], porosity=0.5),
            ],
            wind_speed_multiplier=1,
        ),
        "NearWater": Typology(
            openfield,
            name="NearWater",
            evaporative_cooling_effectiveness=0.15,
            shelters=[
                Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
            ],
            wind_speed_multiplier=1.2,
        ),
        "Misting": Typology(
            openfield,
            name="Misting",
            evaporative_cooling_effectiveness=0.3,
            shelters=[
                Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
            ],
            wind_speed_multiplier=0.5,
        ),
        "PDEC": Typology(
            openfield,
            name="PDEC",
            evaporative_cooling_effectiveness=0.7,
            shelters=[
                Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
            ],
            wind_speed_multiplier=0.5,
        ),
        "NorthShelter": Typology(
            openfield,
            name="NorthShelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[337.5, 22.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
        "NortheastShelter": Typology(
            openfield,
            name="NortheastShelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(altitude_range=[0, 70], azimuth_range=[22.5, 67.5], porosity=0),
            ],
            wind_speed_multiplier=1,
        ),
        "EastShelter": Typology(
            openfield,
            name="EastShelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[67.5, 112.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
        "SoutheastShelter": Typology(
            openfield,
            name="SoutheastShelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[112.5, 157.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
        "SouthShelter": Typology(
            openfield,
            name="SouthShelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[157.5, 202.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
        "SouthwestShelter": Typology(
            openfield,
            name="SouthwestShelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[202.5, 247.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
        "WestShelter": Typology(
            openfield,
            name="WestShelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[247.5, 292.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
        "NorthwestShelter": Typology(
            openfield,
            name="NorthwestShelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[292.5, 337.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
        "NorthShelterWithCanopy": Typology(
            openfield,
            name="NorthShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[337.5, 22.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
        "NortheastShelterWithCanopy": Typology(
            openfield,
            name="NortheastShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(altitude_range=[0, 90], azimuth_range=[22.5, 67.5], porosity=0),
            ],
            wind_speed_multiplier=1,
        ),
        "EastShelterWithCanopy": Typology(
            openfield,
            name="EastShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[67.5, 112.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
        "SoutheastShelterWithCanopy": Typology(
            openfield,
            name="SoutheastShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[112.5, 157.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
        "SouthShelterWithCanopy": Typology(
            openfield,
            name="SouthShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[157.5, 202.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
        "SouthwestShelterWithCanopy": Typology(
            openfield,
            name="SouthwestShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[202.5, 247.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
        "WestShelterWithCanopy": Typology(
            openfield,
            name="WestShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[247.5, 292.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
        "NorthwestShelterWithCanopy": Typology(
            openfield,
            name="NorthwestShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[292.5, 337.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
        ),
    }
    return typologies


if __name__ == "__main__":

    from external_comfort.material import MATERIALS
    from external_comfort.plot.plot import utci_heatmap, utci_pseudo_journey

    epw = EPW(
        r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\test\GBR_London.Gatwick.037760_IWEC.epw"
    )
    ground_material = MATERIALS["CONCRETE_LIGHTWEIGHT"]
    shade_material = MATERIALS["FABRIC"]

    typologies = create_typologies(epw, ground_material, shade_material)

    utcis = []
    descriptions = []
    names = []
    for n, (typology_name, typology) in enumerate(typologies.items()):
        if n > 5:
            continue
        print(f"Calculating UTCI for {typology_name}")
        utci = typology._universal_thermal_climate_index()
        utcis.append(utci)
        descriptions.append(typology.description)
        names.append(typology.name)
        f = utci_heatmap(
            utci,
            title=f"{epw.location.country}-{epw.location.city}\n{typology.description}",
        )
        f.savefig(
            f"C:/Users/tgerrish/Downloads/heatmap_{typology.name}.png",
            transparent=True,
            dpi=300,
            bbox_inches="tight",
        )

    f = utci_pseudo_journey(utcis, month=5, hour=15, names=names)
    f.savefig(
        f"C:/Users/tgerrish/Downloads/utci_journey.png",
        transparent=True,
        dpi=300,
        bbox_inches="tight",
    )
