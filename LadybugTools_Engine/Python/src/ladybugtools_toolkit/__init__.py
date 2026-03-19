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

TOOLKIT_NAME = "LadybugTools_Toolkit"

if os.name == "nt":
    # override "HOME" in case this is set to something other than default for windows
    os.environ["HOME"] = (Path("C:/Users/") / getpass.getuser()).as_posix()

# get dataset paths
SRI_DATA = DATA_DIRECTORY / "sri_data.csv"
KOEPPEN_DATA = DATA_DIRECTORY / "koeppen.csv"
ICE_MATERIALS_DATA = DATA_DIRECTORY / "ICE_database_sources.xlsx.csv"
VEGETATION_DATA = DATA_DIRECTORY / "vegetation.json"
