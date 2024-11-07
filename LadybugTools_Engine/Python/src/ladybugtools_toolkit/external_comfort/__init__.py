"""Root package for external_comfort."""

# pylint: disable=E0401
# import getpass
# from pathlib import Path

# pylint: enable=E0401

# from honeybee.config import folders as hb_folders
# from honeybee_energy.config import folders as hbe_folders
# from honeybee_radiance.config import folders as hbr_folders


# USER = getpass.getuser()
# LB_FOLDER = Path("C:/Program Files/ladybug_tools")
# ROAMING_FOLDER = Path(f"C:/Users/{USER}/AppData/Roaming/ladybug_tools")
# hb_folders.default_simulation_folder = Path(f"C:/Users/{USER}/simulation")

# # pytlint: disable=W0212

# hbe_folders._openstudio_csharp_path = LB_FOLDER / "openstudio/CSharp/openstudio"
# hbe_folders._openstudio_path = LB_FOLDER / "openstudio/bin"
# hbe_folders._openstudio_exe = LB_FOLDER / "openstudio/bin/openstudio.exe"
# hbe_folders.energyplus_path = LB_FOLDER / "openstudio/EnergyPlus"
# hbe_folders._energyplus_exe = LB_FOLDER / "openstudio/EnergyPlus/energyplus.exe"
# hbe_folders._openstudio_results_measure_path = (
#     LB_FOLDER / "resources/measures/openstudio_results"
# )
# hbe_folders._view_data_measure_path = LB_FOLDER / "resources/measures/view_data"
# hbe_folders._inject_idf_measure_path = LB_FOLDER / "resources/measures/inject_idf"
# hbe_folders.lbt_measures_path = LB_FOLDER / "resources/measures"
# hbe_folders._honeybee_adapter_path = (
#     LB_FOLDER
#     / "resources/measures/honeybee_openstudio_gem/lib/files/honeybee_adapter.rb"
# )
# hbe_folders.honeybee_openstudio_gem_path = (
#     LB_FOLDER / "resources/measures/honeybee_openstudio_gem/lib"
# )
# hbe_folders._construction_lib = ROAMING_FOLDER / "standards/constructions"
# hbe_folders._constructionset_lib = ROAMING_FOLDER / "standards/constructionsets"
# hbe_folders._schedule_lib = (
#     "C:/Users/tgerrish/AppData/Roaming/ladybug_tools/standards/schedules"
# )
# hbe_folders._programtype_lib = ROAMING_FOLDER / "standards/programtypes"
# hbe_folders.standards_data_folder = ROAMING_FOLDER / "standards"
# hbe_folders.defaults_file = (
#     LB_FOLDER / "resources/standards/honeybee_standards/energy_default.json"
# )
# hbe_folders.standards_extension_folders = [
#     LB_FOLDER / "resources/standards/honeybee_energy_standards"
# ]
# HBE_FOLDERS = hbe_folders

# hbr_folders._radbin_path = LB_FOLDER / "radiance/bin"
# hbr_folders.radiance_path = LB_FOLDER / "radiance"
# hbr_folders._radlib_path = LB_FOLDER / "radiance/lib"
# hbr_folders._modifier_lib = ROAMING_FOLDER / "standards/modifiers"
# hbr_folders._modifierset_lib = ROAMING_FOLDER / "standards/modifiersets"
# hbr_folders._standards_data_folder = ROAMING_FOLDER / "standards"
# hbr_folders.defaults_file = (
#     LB_FOLDER / "resources/standards/honeybee_standards/radiance_default.json"
# )
# HBR_FOLDERS = hbr_folders

# # SET QUEENBEE PATH
# QUEENBEE_PATH = LB_FOLDER / "python/Scripts/queenbee.exe"

# # pytlint: enable=W0212
