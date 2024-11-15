"""Base module for the ladybugtools_toolkit package."""

# pylint: disable=E0401
import getpass
import os
from pathlib import Path

import matplotlib.pyplot as plt

# pylint: disable=E0401

# get common paths
DATA_DIRECTORY = (Path(__file__).parent.parent / "data").absolute()
BHOM_DIRECTORY = (Path(__file__).parent / "bhom").absolute()
HOME_DIRECTORY = (Path("C:/Users/") / getpass.getuser()).absolute()

# override "HOME" in case IT has set this to something other than default
os.environ["HOME"] = HOME_DIRECTORY.as_posix()

# set plotting style for modules within this toolkit
plt.style.use(
    Path(
        "C:/ProgramData/BHoM/Extensions/PythonCode/Python_Toolkit/src/python_toolkit/bhom/bhom.mplstyle"
    )
)

# get dataset paths
SRI_DATA = DATA_DIRECTORY / "sri_data.csv"
KOEPPEN_DATA = DATA_DIRECTORY / "koeppen.csv"
ICE_MATERIALS_DATA = DATA_DIRECTORY / "ICE_database_sources.xlsx.csv"
VEGETATION_DATA = DATA_DIRECTORY / "vegetation.json"
