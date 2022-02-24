import sys
sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import getpass
from pathlib import Path

from honeybee.config import folders as hb_folders
from honeybee_energy.config import folders as hbe_folders
from honeybee_radiance.config import folders as hbr_folders

USERNAME = getpass.getuser()

ladybug_tools_folder = Path(f"C:/Users/{USERNAME}/ladybug_tools")

hb_folders.default_simulation_folder = f"C:/Users/{USERNAME}/simulation"
hb_folders._python_exe_path = (ladybug_tools_folder / "python/python.exe").as_posix()
hb_folders._python_package_path = (ladybug_tools_folder / "python/Lib/site-packages").as_posix()
hb_folders._python_scripts_path = (ladybug_tools_folder / "python/Scripts").as_posix()

QUEENBEE_EXE = (ladybug_tools_folder / "python/Scripts/queenbee.exe").as_posix()

hbe_folders.openstudio_path = (ladybug_tools_folder / "openstudio/bin").as_posix()
hbe_folders.energyplus_path = (ladybug_tools_folder / "openstudio/EnergyPlus").as_posix()
hbe_folders.honeybee_openstudio_gem_path = (ladybug_tools_folder / "resources/measures/honeybee_openstudio_gem/lib").as_posix()

hbr_folders.radiance_path = (ladybug_tools_folder / "radiance").as_posix()

assert (Path(hbe_folders.openstudio_path) / "openstudio.exe").exists(), \
    f"openstudio.exe not found in {hbe_folders.openstudio_path}. Ensure that the Openstudio installation is located in this directory."

assert Path(hbe_folders.honeybee_openstudio_gem_path).exists(), \
    f"honeybee_openstudio_gem measures not found in {hbe_folders.honeybee_openstudio_gem_path}. Ensure that a Ladyubg-tools installation has been completed installation is located in this directory."

assert (Path(hbr_folders.radiance_path) / "bin/rtrace.exe").exists(), \
    f"Radiance binaries not found in {hbr_folders.radiance_path}. Ensure that the Radiance installation is located in this directory."
