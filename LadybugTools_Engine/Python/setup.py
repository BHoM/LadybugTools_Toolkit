﻿from pathlib import Path

import setuptools
from win32api import HIWORD, LOWORD, GetFileVersionInfo

TOOLKIT_NAME = "LadybugTools_Toolkit"


def _bhom_version() -> str:
    """Return the version of BHoM installed (using the BHoM.dll in the root BHoM directory."""
    info = GetFileVersionInfo(
        "C:/ProgramData/BHoM/Assemblies/BHoM.dll", "\\"
    )  # pylint: disable=[no-name-in-module]
    ms = info["FileVersionMS"]
    ls = info["FileVersionLS"]
    return f"{HIWORD(ms)}.{LOWORD(ms)}.{HIWORD(ls)}.{LOWORD(ls)}"  # pylint: disable=[no-name-in-module]


BHOM_VERSION = _bhom_version()

here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    description=f"A Python library that enables usage of the Python code within {TOOLKIT_NAME} as part of BHoM workflows.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=BHOM_VERSION,
)
