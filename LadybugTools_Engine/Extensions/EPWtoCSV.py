from pathlib import Path
import sys

sys.path.append(r"C:\ProgramData\BHoM\Extensions")

from LadybugTools.epw import BH_EPW

if __name__ == "__main__":
    epw_path = Path(sys.argv[1])
    csv_path = epw_path.with_suffix(".csv")
    print(BH_EPW(epw_path.as_posix()).to_csv(csv_path.as_posix()))
