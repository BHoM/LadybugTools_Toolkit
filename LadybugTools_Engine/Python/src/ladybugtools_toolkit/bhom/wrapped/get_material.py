"""Method to wrap for access to pre-defined materials."""
# pylint: disable=C0415,E0401,W0703
import traceback
import json
from ladybugtools_toolkit.external_comfort.material import Materials
from python_toolkit.bhom.decorators import bhom_wrapper

@bhom_wrapper.bhom_callable("get_material")
def get_material(json_file: str, **kwargs) -> None:
    """Create a file containing all default materials."""
    try:
        json_str = json.dumps([material.value.to_dict() for material in Materials])

        with open(json_file, "w") as f:
            f.write(json_str)

        return json_str

    except Exception as e:
        return traceback.format_exc()
