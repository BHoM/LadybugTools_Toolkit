"""Method to run ALL(?) combinations of given parameters in parallel."""

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ladybugtools_toolkit.interactive.generate_data import (
    DATA_DIR,
    calculate_metrics,
    create_dataset,
)
from tqdm import tqdm


def create_process_save(
    air_temperature: float,
    mean_radiant_temperature: float,
    air_velocitys: list[float],
    relative_humiditys: list[float],
    solar_radiations: list[float],
    clo_value: float,
    metabolic_rate: float,
) -> Path:
    """Calculate and save results for a given set of parameters."""

    sp = (
        DATA_DIR
        / f"{air_temperature}_{mean_radiant_temperature}_{clo_value}_{metabolic_rate}.parquet"
    )
    if sp.exists():
        return sp

    # create dataset to process
    df = create_dataset(
        air_temperatures=[air_temperature],
        mean_radiant_temperatures=[mean_radiant_temperature],
        air_velocitys=air_velocitys,
        relative_humiditys=relative_humiditys,
        solar_radiations=solar_radiations,
        clo_values=[clo_value],
        metabolic_rates=[metabolic_rate],
    )
    # calculate and save
    calculate_metrics(dataframe=df).to_parquet(sp, compression="brotli", index=False)
    return sp


if __name__ == "__main__":

    # generate the data - need to be kep tsame across files!
    _air_temperatures = np.arange(-10, 50, 2).tolist()
    _mean_radiant_temperatures = np.arange(-10, 101, 5).tolist()
    _relative_humiditys = np.arange(0, 110, 5).tolist()
    _solar_radiations = np.arange(0, 1401, 50).tolist()
    _air_velocitys = np.arange(0, 33, 1).tolist()
    _clo_values = [0.46, 0.65, 0.83, 1, 1.2]
    _metabolic_rates = [0.8, 1.2, 1.6, 2, 4, 6, 8]

    n_iterations = (
        len(_air_temperatures)
        * len(_mean_radiant_temperatures)
        * len(_relative_humiditys)
        * len(_solar_radiations)
        * len(_air_velocitys)
        * len(_clo_values)
        * len(_metabolic_rates)
    )
    print(f"Total number of iterations: {n_iterations}")

    local_iters = itertools.product(
        _air_temperatures, _mean_radiant_temperatures, _clo_values, _metabolic_rates
    )
    n_local_iterations = (
        len(_air_temperatures)
        * len(_mean_radiant_temperatures)
        * len(_clo_values)
        * len(_metabolic_rates)
    )

    with tqdm(total=n_local_iterations) as pbar:
        with ProcessPoolExecutor() as executor:
            for air_temperature, mean_radiant_temperature, clo_value, metabolic_rate in local_iters:
                results = []
                results.append(
                    executor.submit(
                        create_process_save,
                        air_temperature=air_temperature,
                        mean_radiant_temperature=mean_radiant_temperature,
                        air_velocitys=_air_velocitys,
                        relative_humiditys=_relative_humiditys,
                        solar_radiations=_solar_radiations,
                        clo_value=clo_value,
                        metabolic_rate=metabolic_rate,
                    )
                )

            for future in as_completed(results):
                future.result()
                pbar.update(1)
