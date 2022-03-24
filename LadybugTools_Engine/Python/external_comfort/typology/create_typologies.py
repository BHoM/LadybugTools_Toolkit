from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from typing import List, Union

from external_comfort.openfield import Openfield
from external_comfort.shelter import Shelter
from external_comfort.typology import Typology
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug.epw import EPW


def create_typologies(
    epw: EPW,
    ground_material: Union[str, _EnergyMaterialOpaqueBase],
    shade_material: Union[str, _EnergyMaterialOpaqueBase],
    calculate: bool = False,
) -> List[Typology]:
    """Create a dictionary of typologies for a given epw file and context configuration, with all requisite simulations and calculations completed

    Args:
        epw (EPW): The epw file to create typologies for
        ground_material (Union[str, _EnergyMaterialOpaqueBase]): The ground material to use for the typologies
        shade_material (Union[str, _EnergyMaterialOpaqueBase]): The shade material to use for the typologies
        calculate (bool, optional): Whether to pre-process the typologies generated. Defaults to False.

    Returns:
        List[Typology]: A list of typologies
    """

    openfield = Openfield(epw, ground_material, shade_material, True)
    typologies = [
        Typology(
            openfield,
            name="Openfield",
            evaporative_cooling_effectiveness=0,
            shelters=[],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Enclosed",
            evaporative_cooling_effectiveness=0,
            shelters=[
                Shelter(altitude_range=[0, 90], azimuth_range=[0, 360], porosity=0)
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Partially enclosed",
            evaporative_cooling_effectiveness=0,
            shelters=[
                Shelter(altitude_range=[0, 90], azimuth_range=[0, 360], porosity=0.5)
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Sky-shelter",
            evaporative_cooling_effectiveness=0,
            shelters=[
                Shelter(altitude_range=[45, 90], azimuth_range=[0, 360], porosity=0)
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Fritted sky-shelter",
            evaporative_cooling_effectiveness=0,
            shelters=[
                Shelter(altitude_range=[45, 90], azimuth_range=[0, 360], porosity=0.5),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Near water",
            evaporative_cooling_effectiveness=0.15,
            shelters=[
                Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
            ],
            wind_speed_multiplier=1.2,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Misting",
            evaporative_cooling_effectiveness=0.3,
            shelters=[
                Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
            ],
            wind_speed_multiplier=0.5,
            calculate=False,
        ),
        Typology(
            openfield,
            name="PDEC",
            evaporative_cooling_effectiveness=0.7,
            shelters=[
                Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
            ],
            wind_speed_multiplier=0.5,
            calculate=False,
        ),
        Typology(
            openfield,
            name="North shelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[337.5, 22.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Northeast shelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(altitude_range=[0, 70], azimuth_range=[22.5, 67.5], porosity=0),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="East shelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[67.5, 112.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Southeast shelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[112.5, 157.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="South shelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[157.5, 202.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Southwest shelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[202.5, 247.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="West shelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[247.5, 292.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Northwest shelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[292.5, 337.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="North shelter (with canopy)",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[337.5, 22.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Northeast shelter (with canopy)",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(altitude_range=[0, 90], azimuth_range=[22.5, 67.5], porosity=0),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="East shelter (with canopy)",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[67.5, 112.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Southeast shelter (with canopy)",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[112.5, 157.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="South shelter (with canopy)",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[157.5, 202.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Southwest shelter (with canopy)",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[202.5, 247.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="West shelter (with canopy)",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[247.5, 292.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="Northwest shelter (with canopy)",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[292.5, 337.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="East-west shelter",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[67.5, 112.5], porosity=0
                ),
                Shelter(
                    altitude_range=[0, 70], azimuth_range=[247.5, 292.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
        Typology(
            openfield,
            name="East-west shelter (with canopy)",
            evaporative_cooling_effectiveness=0.0,
            shelters=[
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[67.5, 112.5], porosity=0
                ),
                Shelter(
                    altitude_range=[0, 90], azimuth_range=[247.5, 292.5], porosity=0
                ),
            ],
            wind_speed_multiplier=1,
            calculate=False,
        ),
    ]

    if calculate:
        t = []
        with ThreadPoolExecutor() as executor:
            fut = [executor.submit(typology._calculate) for typology in typologies]
            for r in as_completed(fut):
                t.append(r.result())
        return t            

    return typologies
