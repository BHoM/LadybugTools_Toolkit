from pathlib import Path
import setuptools

TOOLKIT_NAME = "LadybugTools_Toolkit"

here = Path(__file__).parent.resolve()

def _bhom_version() -> str:
    """Return the version of BHoM installed (using the BHoM.dll in the root BHoM directory."""
    versionFile = here / "VERSION.txt" #version file is created in a pre-build event in LadybugTools_Engine.csproj
    return versionFile.read_text();

BHOM_VERSION = _bhom_version()

long_description = (here / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    description=f"A Python library that enables usage of the Python code within {TOOLKIT_NAME} as part of BHoM workflows.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=BHOM_VERSION,
)
