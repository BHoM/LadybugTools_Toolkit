import getpass

from honeybee.config import folders as hb_folders

from .external_comfort import ExternalComfort
from .external_comfort_result import ExternalComfortResult
from .materials import Materials
from .shelter import Shelter

# from .typology import Typologies, Typology, TypologyResult

hb_folders.default_simulation_folder = f"C:/Users/{getpass.getuser()}/simulation"
