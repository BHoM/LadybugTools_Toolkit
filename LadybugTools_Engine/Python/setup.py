import pathlib

import setuptools

TOOLKIT_NAME = "LadybugTools_Toolkit"

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")
requirements = [
    i.strip()
    for i in (here / "requirements.txt").read_text(encoding="utf-8-sig").splitlines()
]

setuptools.setup(
    name=TOOLKIT_NAME.lower(),
    author="BHoM",
    author_email="bhombot@burohappold.com",
    description=f"A Python library that enables usage of the Python code within {TOOLKIT_NAME} as part of BHoM workflows.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/BHoM/{TOOLKIT_NAME}",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", exclude=["tests"]),
    package_data={
        "data": ["data/*"],
    },
    install_requires=requirements,
)
