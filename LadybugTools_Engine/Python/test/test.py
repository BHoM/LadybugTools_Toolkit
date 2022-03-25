import sys
sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from external_comfort.openfield import EPW, Openfield, OpenfieldResult
from external_comfort.typology import TYPOLOGIES
from external_comfort.typology.create_typologies import create_typologies

def main() -> Openfield:
    epw = EPW(r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\test\GBR_London.Gatwick.037760_IWEC.epw")
    
    of = Openfield(epw, "Asphalt", "Fabric")
    ofr = OpenfieldResult(of)

    typology_results = create_typologies(ofr, TYPOLOGIES.values())

    for t in typology_results:
        print(f"{t.typology.name}: {t.universal_thermal_climate_index.average:0.2f}C")

if __name__ == "__main__":
    main()
