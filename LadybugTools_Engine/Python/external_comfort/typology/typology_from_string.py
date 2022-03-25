import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from external_comfort.typology import Typology, TYPOLOGIES


def typology_from_string(typology_string: str) -> Typology:
    """
    Return the Typology object associated with the given string.
    """
    try:
        return TYPOLOGIES[typology_string]
    except KeyError:
        raise ValueError(
            f"Unknown typology: {typology_string}. Choose from one of {list(TYPOLOGIES.keys())}."
        )
