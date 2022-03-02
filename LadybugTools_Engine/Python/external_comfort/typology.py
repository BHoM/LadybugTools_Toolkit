from __future__ import annotations

import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from typing import Dict

import numpy as np
import pandas as pd
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug_comfort.collection.utci import UTCI
from ladybug_extension.datacollection import from_series, to_series

from external_comfort.evaporative_cooling import get_evaporative_cooled_dbt_rh
from external_comfort.openfield import Openfield
from external_comfort.shelter import Shelter


class Typology:
    def __init__(
        self,
        openfield: Openfield,
        name: str = "Openfield",
        evaporative_cooling_effectiveness: float = 0,
        shelter: Shelter = None,
    ) -> Typology:
        """Class for defining a specific external comfort typology, and calculating the resultant thermal comfort values.

        Args:
            openfield (Openfield): An Openfield object.
            name (str, optional): A string for the name of the typology. Defaults to "Openfield".
            evaporative_cooling_effectiveness (float, optional): A float between 0 and 1 for the effectiveness of the contextual evaporative cooling modifying air temperature. Defaults to 0.
            shelter_polygon (Polygon, optional): A Polygon object defining the shelter polygon. Defaults to a shelter covering the top 1/3 of the sky.
        """
        self.name = name
        self.openfield = openfield
        self.evaporative_cooling_effectiveness = evaporative_cooling_effectiveness
        if not shelter:
            self.shelter = Shelter(
                azimuth_range=[0, 0],
                altitude_range=[0, 0],
                porosity=1,
            )
        else:
            self.shelter = shelter

        if not (
            (0 <= self.evaporative_cooling_effectiveness <= 1)
            or isinstance(self.evaporative_cooling_effectiveness, (int, float))
        ):
            raise ValueError(
                f"evaporative_cooling_effectiveness must be a number between 0 and 1"
            )

    @property
    def description(self) -> str:
        """Return a text description of this external comfort typology."""

        descriptive_str = ""
        descriptive_str += self.shelter.description
        if self.evaporative_cooling_effectiveness > 0:
            descriptive_str += f", with {self.evaporative_cooling_effectiveness:0.0%} effective evaporative cooling"
        return descriptive_str

    def __str__(self) -> str:
        return f"{self.name} - {self.description}"

    def __repr__(self) -> str:
        return str(self)

    def _dry_bulb_temperature(self) -> HourlyContinuousCollection:
        """Return the effective DBT for the given typology."""
        return get_evaporative_cooled_dbt_rh(
            self.openfield.epw, self.evaporative_cooling_effectiveness
        )["dry_bulb_temperature"]

    def _relative_humidity(self) -> HourlyContinuousCollection:
        """Return the effective RH for the given typology."""
        return get_evaporative_cooled_dbt_rh(
            self.openfield.epw, self.evaporative_cooling_effectiveness
        )["relative_humidity"]

    def _wind_speed(self) -> HourlyContinuousCollection:
        """Return the effective WS for the given typology (taking account of reductions from shelter location and porosity)."""
        return self.shelter.effective_wind_speed(self.openfield.epw)

    def _mean_radiant_temperature(self) -> HourlyContinuousCollection:
        """Return the effective MRT for the given typology (taking account of reductions from shelter location and porosity)."""
        shaded_series = to_series(self.openfield.shaded_mean_radiant_temperature)
        unshaded_series = to_series(self.openfield.unshaded_mean_radiant_temperature)

        # TODO - Calibrate the rolling/ewm method for transition between shaded/unshaded interpolants

        factors = self.shelter._unshaded_shaded_interpolant(self.openfield.epw)
        factors_series = (
            pd.Series(factors, index=shaded_series.index).ewm(halflife=1).mean()
        )

        mrt_values = []
        for factor, shaded, unshaded in list(
            zip(*[factors_series, shaded_series, unshaded_series])
        ):
            mrt_values.append(np.interp(factor, [0, 1], [shaded, unshaded]))
        mrt_series = pd.Series(
            mrt_values, index=shaded_series.index, name="Mean Radiant Temperature (C)"
        )

        return from_series(mrt_series)

    def _universal_thermal_climate_index(self) -> HourlyContinuousCollection:
        """Return the effective UTCI for the given typology."""
        return UTCI(
            air_temperature=self._dry_bulb_temperature(),
            rel_humidity=self._relative_humidity(),
            rad_temperature=self._mean_radiant_temperature(),
            wind_speed=self._wind_speed(),
        ).universal_thermal_climate_index


def create_typologies(
    epw: EPW,
    ground_material: _EnergyMaterialOpaqueBase,
    shade_material: _EnergyMaterialOpaqueBase,
) -> Dict[str, Typology]:
    """Create a dictionary of typologies for a given epw file, with all requisite simulations and calculations completed"""
    openfield = Openfield(epw, ground_material, shade_material, True)
    typologies = {
        "Openfield": Typology(
            openfield,
            name="Openfield",
            evaporative_cooling_effectiveness=0,
            shelter=Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
        ),
        "Enclosed": Typology(
            openfield,
            name="Enclosed",
            evaporative_cooling_effectiveness=0,
            shelter=Shelter(altitude_range=[0, 90], azimuth_range=[0, 360], porosity=0),
        ),
        "PartiallyEnclosed": Typology(
            openfield,
            name="PartiallyEnclosed",
            evaporative_cooling_effectiveness=0,
            shelter=Shelter(
                altitude_range=[0, 90], azimuth_range=[0, 360], porosity=0.5
            ),
        ),
        "SkyShelter": Typology(
            openfield,
            name="SkyShelter",
            evaporative_cooling_effectiveness=0,
            shelter=Shelter(
                altitude_range=[45, 90], azimuth_range=[0, 360], porosity=0
            ),
        ),
        "FrittedSkyShelter": Typology(
            openfield,
            name="FrittedSkyShelter",
            evaporative_cooling_effectiveness=0,
            shelter=Shelter(
                altitude_range=[45, 90], azimuth_range=[0, 360], porosity=0.5
            ),
        ),
        "Misting": Typology(
            openfield,
            name="Misting",
            evaporative_cooling_effectiveness=0.3,
            shelter=Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
        ),
        "PDEC": Typology(
            openfield,
            name="PDEC",
            evaporative_cooling_effectiveness=0.7,
            shelter=Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
        ),
        "NorthShelter": Typology(
            openfield,
            name="NorthShelter",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 70], azimuth_range=[337.5, 22.5], porosity=0
            ),
        ),
        "NortheastShelter": Typology(
            openfield,
            name="NortheastShelter",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 70], azimuth_range=[22.5, 67.5], porosity=0
            ),
        ),
        "EastShelter": Typology(
            openfield,
            name="EastShelter",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 70], azimuth_range=[67.5, 112.5], porosity=0
            ),
        ),
        "SoutheastShelter": Typology(
            openfield,
            name="SoutheastShelter",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 70], azimuth_range=[112.5, 157.5], porosity=0
            ),
        ),
        "SouthShelter": Typology(
            openfield,
            name="SouthShelter",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 70], azimuth_range=[157.5, 202.5], porosity=0
            ),
        ),
        "SouthwestShelter": Typology(
            openfield,
            name="SouthwestShelter",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 70], azimuth_range=[202.5, 247.5], porosity=0
            ),
        ),
        "WestShelter": Typology(
            openfield,
            name="WestShelter",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 70], azimuth_range=[247.5, 292.5], porosity=0
            ),
        ),
        "NorthwestShelter": Typology(
            openfield,
            name="NorthwestShelter",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 70], azimuth_range=[292.5, 337.5], porosity=0
            ),
        ),
        "NorthShelterWithCanopy": Typology(
            openfield,
            name="NorthShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 90], azimuth_range=[337.5, 22.5], porosity=0
            ),
        ),
        "NortheastShelterWithCanopy": Typology(
            openfield,
            name="NortheastShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 90], azimuth_range=[22.5, 67.5], porosity=0
            ),
        ),
        "EastShelterWithCanopy": Typology(
            openfield,
            name="EastShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 90], azimuth_range=[67.5, 112.5], porosity=0
            ),
        ),
        "SoutheastShelterWithCanopy": Typology(
            openfield,
            name="SoutheastShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 90], azimuth_range=[112.5, 157.5], porosity=0
            ),
        ),
        "SouthShelterWithCanopy": Typology(
            openfield,
            name="SouthShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 90], azimuth_range=[157.5, 202.5], porosity=0
            ),
        ),
        "SouthwestShelterWithCanopy": Typology(
            openfield,
            name="SouthwestShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 90], azimuth_range=[202.5, 247.5], porosity=0
            ),
        ),
        "WestShelterWithCanopy": Typology(
            openfield,
            name="WestShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 90], azimuth_range=[247.5, 292.5], porosity=0
            ),
        ),
        "NorthwestShelterWithCanopy": Typology(
            openfield,
            name="NorthwestShelterWithCanopy",
            evaporative_cooling_effectiveness=0.0,
            shelter=Shelter(
                altitude_range=[0, 90], azimuth_range=[292.5, 337.5], porosity=0
            ),
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
