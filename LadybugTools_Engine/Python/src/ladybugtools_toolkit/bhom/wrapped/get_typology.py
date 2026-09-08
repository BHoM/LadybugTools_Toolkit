"""Method to wrap for access to pre-defined typologies."""
# pylint: disable=C0415,E0401,W0703
import traceback
import json
from ladybugtools_toolkit.external_comfort.typology import Typologies
from python_toolkit.bhom.decorators import bhom_wrapper

@bhom_wrapper.bhom_callable("get_typology")
def get_typology(json_file: str, **kwargs) -> None:
    """Create a file containing all default typologies."""
    try:
        json_str = json.dumps([typology.value.to_dict() for typology in Typologies])

        with open(json_file, "w") as f:
            f.write(json_str)

        return json_str

    except Exception as e:
        return traceback.format_exc()
