from pathlib import Path

from honeybee.config import folders as hb_folders

DATA_DIR = Path(hb_folders.default_simulation_folder) / "interactive_thermal_comfort_data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

PERSON_HEIGHT = 1.65
PERSON_AGE = 36
PERSON_SEX = 0.5
PERSON_HEIGHT = 1.65
PERSON_MASS = 62
PERSON_POSITION = "standing"

TERRAIN_ROUGHNESS_LENGTH = 0.03
ATMOSPHERIC_PRESSURE = 101325
