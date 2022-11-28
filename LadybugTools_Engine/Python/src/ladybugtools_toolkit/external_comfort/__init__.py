# pylint: skip-file

import getpass

from honeybee.config import folders as hb_folders
from honeybee_energy.config import folders as hbe_folders
from honeybee_radiance.config import folders as hbr_folders

USER = getpass.getuser()
hb_folders.default_simulation_folder = f"C:/Users/{USER}/simulation"

# SET HONEYBEE ENERGY PATHS
hbe_folders._openstudio_csharp_path = (
    "C:/Program Files/ladybug_tools/openstudio/CSharp/openstudio"
)
hbe_folders._openstudio_path = "C:/Program Files/ladybug_tools/openstudio/bin"
hbe_folders._openstudio_exe = (
    "C:/Program Files/ladybug_tools/openstudio/bin/openstudio.exe"
)
hbe_folders.energyplus_path = "C:/Program Files/ladybug_tools/openstudio/EnergyPlus"
hbe_folders._energyplus_exe = (
    "C:/Program Files/ladybug_tools/openstudio/EnergyPlus/energyplus.exe"
)
hbe_folders._openstudio_results_measure_path = (
    "C:/Program Files/ladybug_tools/resources/measures/openstudio_results"
)
hbe_folders._view_data_measure_path = (
    "C:/Program Files/ladybug_tools/resources/measures/view_data"
)
hbe_folders._inject_idf_measure_path = (
    "C:/Program Files/ladybug_tools/resources/measures/inject_idf"
)
hbe_folders.lbt_measures_path = "C:/Program Files/ladybug_tools/resources/measures"
hbe_folders._honeybee_adapter_path = "C:/Program Files/ladybug_tools/resources/measures/honeybee_openstudio_gem/lib/files/honeybee_adapter.rb"
hbe_folders.honeybee_openstudio_gem_path = (
    "C:/Program Files/ladybug_tools/resources/measures/honeybee_openstudio_gem/lib"
)
hbe_folders._construction_lib = (
    f"C:/Users/{USER}/AppData/Roaming/ladybug_tools/standards/constructions"
)
hbe_folders._constructionset_lib = (
    f"C:/Users/{USER}/AppData/Roaming/ladybug_tools/standards/constructionsets"
)
hbe_folders._schedule_lib = (
    "C:/Users/tgerrish/AppData/Roaming/ladybug_tools/standards/schedules"
)
hbe_folders._programtype_lib = (
    f"C:/Users/{USER}/AppData/Roaming/ladybug_tools/standards/programtypes"
)
hbe_folders.standards_data_folder = (
    f"C:/Users/{USER}/AppData/Roaming/ladybug_tools/standards"
)
hbe_folders.defaults_file = "C:/Program Files/ladybug_tools/resources/standards/honeybee_standards/energy_default.json"
hbe_folders.standards_extension_folders = [
    "C://Program Files//ladybug_tools//resources//standards//honeybee_energy_standards"
]
HBE_FOLDERS = hbe_folders

# SET HONEYBEE RADIANCE PATHS
hbr_folders._radbin_path = "C:/Program Files/ladybug_tools/radiance/bin"
hbr_folders.radiance_path = "C:/Program Files/ladybug_tools/radiance"
hbr_folders._radlib_path = "C:/Program Files/ladybug_tools/radiance/lib"
hbr_folders._modifier_lib = (
    f"C:/Users/{USER}/AppData/Roaming/ladybug_tools/standards/modifiers"
)
hbr_folders._modifierset_lib = (
    f"C:/Users/{USER}/AppData/Roaming/ladybug_tools/standards/modifiersets"
)
hbr_folders._standards_data_folder = (
    f"C:/Users/{USER}/AppData/Roaming/ladybug_tools/standards"
)
hbr_folders.defaults_file = "C:/Program Files/ladybug_tools/resources/standards/honeybee_standards/radiance_default.json"
HBR_FOLDERS = hbr_folders
