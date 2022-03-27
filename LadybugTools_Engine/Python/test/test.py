import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from external_comfort import ExternalComfort, ExternalComfortResult
from external_comfort.material import MATERIALS
from external_comfort.shelter import Shelter
from external_comfort.typology import (
    TYPOLOGIES,
    Typology,
    TypologyResult,
    calculate_typology_results,
)
from ladybug.epw import EPW


def main() -> None:
    epw = EPW(
        r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\test\GBR_London.Gatwick.037760_IWEC.epw"
    )

    ec = ExternalComfort(
        epw=epw,
        ground_material=MATERIALS["Asphalt"],
        shade_material=MATERIALS["Fabric"],
    )
    ecr = ExternalComfortResult(ec)

    sh1 = Shelter(porosity=0.5, altitude_range=[60, 90], azimuth_range=[0, 360])
    sh2 = Shelter(porosity=0, altitude_range=[0, 20], azimuth_range=[0, 90])
    sh3 = Shelter(porosity=0, altitude_range=[0, 20], azimuth_range=[270, 360])

    tyrs = calculate_typology_results(list(TYPOLOGIES.values())[0:5], ecr)

    for tyr in tyrs:
        print(
            f"{tyr.typology.name}: {tyr.universal_thermal_climate_index.average:0.2f}C"
        )


if __name__ == "__main__":
    main()
