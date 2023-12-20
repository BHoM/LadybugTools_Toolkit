import shutil
from ladybug.futil import nukedir

from . import EXTERNAL_COMFORT_DIRECTORY, SPATIAL_COMFORT_DIRECTORY


def pytest_sessionstart(session):
    """_"""
    if SPATIAL_COMFORT_DIRECTORY.exists():
        print(f"Removing existing test files from {SPATIAL_COMFORT_DIRECTORY}")
        try:
            shutil.rmtree(SPATIAL_COMFORT_DIRECTORY)
        except PermissionError:
            nukedir(SPATIAL_COMFORT_DIRECTORY, rmdir=True)
    if EXTERNAL_COMFORT_DIRECTORY.exists():
        print(f"Removing existing test files from {EXTERNAL_COMFORT_DIRECTORY}")
        try:
            shutil.rmtree(EXTERNAL_COMFORT_DIRECTORY)
        except PermissionError:
            nukedir(EXTERNAL_COMFORT_DIRECTORY, rmdir=True)
