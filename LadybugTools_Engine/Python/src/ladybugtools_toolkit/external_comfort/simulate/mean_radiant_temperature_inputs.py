from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from honeybee.model import Model
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW

from .solar_radiation import solar_radiation
from .surface_temperature import surface_temperature
from .working_directory import working_directory as wd


def mean_radiant_temperature_inputs(
    model: Model, epw: EPW
) -> Dict[str, HourlyContinuousCollection]:
    """Run both a surface temperature and solar radiation simulation concurrently and return the combined results.

    Args:
        model (Model): A model used as part of the External Comfort workflow.
        epw (EPW): The EPW file in which to simulate the model.

    Returns:
        Dict[str, HourlyContinuousCollection]: A dictionary containing all collections necessary to create shaded and unshaded mean-radiant-temperature collections.
    """

    working_directory = wd(model)

    # save EPW to working directory for later use
    if not (working_directory / Path(epw.file_path).name).exists():
        epw.save((working_directory / Path(epw.file_path).name).as_posix())

    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for func in [solar_radiation, surface_temperature]:
            results.append(executor.submit(func, model, epw))

    return {k: v for d in [r.result() for r in results] for k, v in d.items()}
