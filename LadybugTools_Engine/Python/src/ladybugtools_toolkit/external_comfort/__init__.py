import getpass

from honeybee.config import folders as hb_folders

# from ladybugtools_toolkit.external_comfort.external_comfort import ExternalComfort
from ladybugtools_toolkit.external_comfort.materials import Materials
from ladybugtools_toolkit.external_comfort.shelter import Shelter
from ladybugtools_toolkit.external_comfort.typology.typologies import (
    Typologies,
    Typology,
)

hb_folders.default_simulation_folder = f"C:/Users/{getpass.getuser()}/simulation"
