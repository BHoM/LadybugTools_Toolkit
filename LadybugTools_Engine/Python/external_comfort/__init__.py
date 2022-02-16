import getpass
from pathlib import Path

from honeybee.config import folders as hb_folders
from honeybee_energy.config import folders as hbe_folders
from honeybee_radiance.config import folders as hbr_folders

USERNAME = getpass.getuser()
TOOLKIT_ENVIRONMENT = Path(
    "C:/ProgramData/BHoM/Extensions/PythonEnvironments/LadybugTools_Toolkit"
)

ladybug_tools_folder = Path(f"C:/Users/{USERNAME}/ladybug_tools")

hb_folders.default_simulation_folder = f"C:/Users/{USERNAME}/simulation"
hb_folders._python_exe_path = (TOOLKIT_ENVIRONMENT / "python.exe").as_posix()
hb_folders._python_package_path = (TOOLKIT_ENVIRONMENT / "Lib/site-packages").as_posix()
hb_folders._python_scripts_path = (TOOLKIT_ENVIRONMENT / "Scripts").as_posix()
QUEENBEE_EXE = (TOOLKIT_ENVIRONMENT / "Scripts/queenbee.exe").as_posix()

hbe_folders.openstudio_path = (ladybug_tools_folder / "openstudio/bin").as_posix()
hbe_folders.energyplus_path = (
    ladybug_tools_folder / "openstudio/EnergyPlus"
).as_posix()
hbe_folders.honeybee_openstudio_gem_path = (
    ladybug_tools_folder / "resources/measures/honeybee_openstudio_gem/lib"
).as_posix()

hbr_folders.radiance_path = (ladybug_tools_folder / "radiance").as_posix()
