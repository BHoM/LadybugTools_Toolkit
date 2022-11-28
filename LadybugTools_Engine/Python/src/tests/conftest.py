import shutil

from ..tests import EXTERNAL_COMFORT_DIRECTORY, SPATIAL_COMFORT_DIRECTORY


def pytest_sessionstart(session):
    """_"""
    if SPATIAL_COMFORT_DIRECTORY.exists():
        print(f"Removing existing test files from {SPATIAL_COMFORT_DIRECTORY}")
        shutil.rmtree(SPATIAL_COMFORT_DIRECTORY)
    if EXTERNAL_COMFORT_DIRECTORY.exists():
        print(f"Removing existing test files from {EXTERNAL_COMFORT_DIRECTORY}")
        shutil.rmtree(EXTERNAL_COMFORT_DIRECTORY)
