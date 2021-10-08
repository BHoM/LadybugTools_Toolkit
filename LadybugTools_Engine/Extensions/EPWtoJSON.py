import sys

sys.path.append(r"C:\ProgramData\BHoM\Extensions")

from LadybugTools.epw import BH_EPW

if __name__ == "__main__":
    print(BH_EPW(sys.argv[1]).to_json())
