"""Method to wrap for conversion of HBJSON to GEM file."""
# pylint: disable=C0415,E0401,W0703
import json
import traceback
from pathlib import Path
import uuid
from ..logger import CONSOLE_LOGGER
import tempfile
from honeybee.model import Model
from honeybee_ies.writer import model_to_ies
from python_toolkit.bhom.decorators import bhom_wrapper

@bhom_wrapper.bhom_callable("hbjson_to_gem")
def hbjson_to_gem(hbjson_file: str, **kwargs) -> None:
    """Create an IES GEM file from an HBJSON file."""
    try:
        hbjson_dict = None

        try:
            file_path = Path(hbjson_file)
            if file_path.is_file():
                hbjson_dict = json.loads(file_path.read_text())
        except:
            CONSOLE_LOGGER.info("gem file provided was not a path, trying to read as json...")

        if hbjson_dict is None:
            hbjson_dict = json.loads(hbjson_file)

        model = Model.from_dict(hbjson_dict)
        name = str(uuid.uuid4()) + ".gem"
        model_to_ies(
            model, folder=tempfile.gettempdir(), name=name
        )

        gem_file = (Path(tempfile.gettempdir()) / name)
        gem = gem_file.read_text()
        gem_file.unlink()

        return gem
    except Exception:
        CONSOLE_LOGGER.error("Could not convert the hbjson file to a gem file.", exc_info=1)
        return traceback.format_exc()
