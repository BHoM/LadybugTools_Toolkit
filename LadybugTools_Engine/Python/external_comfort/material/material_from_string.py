import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from external_comfort.material import MATERIALS
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase


def material_from_string(material_string: str) -> _EnergyMaterialOpaqueBase:
    """
    Return the EnergyMaterial object associated with the given string.
    """
    try:
        return MATERIALS[material_string]
    except KeyError:
        raise ValueError(
            f"Unknown material: {material_string}. Choose from one of {list(MATERIALS.keys())}."
        )
