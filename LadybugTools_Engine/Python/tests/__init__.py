import getpass
from pathlib import Path

from honeybee.config import folders as hb_folders
from honeybee.model import Model
from ladybug.epw import EPW

# set identifier for all downstream processes
BASE_IDENTIFIER = "pytest"

# set simulation directory for downstream processes
hb_folders.default_simulation_folder = f"C:/Users/{getpass.getuser()}/simulation"

# set identifiers for ExternalComfort simulation and SpatialComfort simulation
EXTERNAL_COMFORT_IDENTIFIER = f"{BASE_IDENTIFIER}_external_comfort"
EXTERNAL_COMFORT_DIRECTORY = (
    Path(hb_folders.default_simulation_folder) / EXTERNAL_COMFORT_IDENTIFIER
)
SPATIAL_COMFORT_IDENTIFIER = f"{BASE_IDENTIFIER}_spatial_comfort"
SPATIAL_COMFORT_DIRECTORY = (
    Path(hb_folders.default_simulation_folder) / SPATIAL_COMFORT_IDENTIFIER
)

# set source files used in testing
SPATIAL_COMFORT_MODEL_FILE = (
    Path(__file__).parent / "assets" / "example_spatial_comfort.hbjson"
)
SPATIAL_COMFORT_MODEL_OBJ: Model = Model.from_hbjson(SPATIAL_COMFORT_MODEL_FILE)

EPW_FILE = Path(__file__).parent / "assets" / "example.epw"
EPW_CSV_FILE = (Path(__file__).parent / "assets" / "example.csv",)
EPW_OBJ = EPW(EPW_FILE)

RES_FILE = Path(__file__).parent / "assets" / "example.res"
SQL_FILE = Path(__file__).parent / "assets" / "example.sql"
ILL_FILE = Path(__file__).parent / "assets" / "example.ill"
NPY_FILE = Path(__file__).parent / "assets" / "example.npy"
PTS_FILE = Path(__file__).parent / "assets" / "example.pts"
CFD_DIRECTORY = Path(__file__).parent / "assets" / "cfd"
