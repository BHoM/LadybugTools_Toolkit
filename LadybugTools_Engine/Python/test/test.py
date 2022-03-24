import sys
sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from external_comfort.openfield import EPW, Openfield
from external_comfort.typology.create_typologies import create_typologies

def main() -> Openfield:
    epw = EPW(r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\test\GBR_London.Gatwick.037760_IWEC.epw")
    
    # of = Openfield(epw, "ASPHALT", "FABRIC", True)

    typologies = create_typologies(epw, "ASPHALT", "FABRIC", calculate=True)

    for typology in typologies:
        print(f"{typology.name} : {typology.universal_thermal_climate_index.average:0.2f}C")

if __name__ == "__main__":
    main()
