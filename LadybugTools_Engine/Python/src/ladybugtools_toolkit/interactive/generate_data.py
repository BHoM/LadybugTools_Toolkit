"""Methods to generate datasets decribing thermal comfort according to varying 
indices, and an interactive web-page to play with the data."""

import inspect
import itertools
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from ladybug.epw import EPW, Header
from ladybug_comfort.collection.solarcal import OutdoorSolarCal
from ladybugtools_toolkit.helpers import chunks
from tqdm import tqdm

from . import DATA_DIR, PERSON_HEIGHT, TERRAIN_ROUGHNESS_LENGTH
from .wrapped_methods import (
    _actual_sensation_vote,
    _apparent_temperature,
    _discomfort_index,
    _heat_index,
    _humidex,
    _physiologic_equivalent_temperature,
    _standard_effective_temperature,
    _thermal_sensation,
    _universal_thermal_climate_index,
    _wet_bulb_globe_temperature,
    _windchill_temp,
)


def calculate_metrics(
    dataframe: pd.DataFrame, metrics_to_calculate: list[Callable] = None
) -> pd.DataFrame:
    """Calculate a set of thermal comfort metrics for a given dataframe.

    The dataframe should contain columns that match the arguments of the callables
    which calculate the metrics. The metrics will be calculated for each row of the
    dataframe.

    Args:
        dataframe (pd.DataFrame):
            The dataframe containing the inputs to calculate the metrics.
        metrics_to_calculate (list[Callable], optional):
            A list of callables that calculate the metrics. The callables have
            arguments that match the columns in the dataframe.
            If this list is None, then all metrics will be calculated.

    Returns:
        pd.DataFrame:
            The original dataframe with additional columns for each metric calculated.
    """

    # validate
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The dataframe should be a pandas DataFrame.")

    if metrics_to_calculate is None:
        metrics_to_calculate = [
            _actual_sensation_vote,
            _apparent_temperature,
            _discomfort_index,
            _heat_index,
            _humidex,
            _physiologic_equivalent_temperature,
            _standard_effective_temperature,
            _thermal_sensation,
            _universal_thermal_climate_index,
            _wet_bulb_globe_temperature,
            _windchill_temp,
        ]
    if len(metrics_to_calculate) == 0:
        raise ValueError("At least one metric should be provided.")

    # get all possible arguments from the callables
    all_args = set()
    for func in metrics_to_calculate:
        all_args.update(inspect.getfullargspec(func).args)

    # check arguments are present in the given dataframe
    for arg in all_args:
        if arg not in dataframe.columns:
            raise ValueError(f"Argument {arg} is missing from the dataframe.")

    # run each metric
    r = []
    pbar = tqdm(metrics_to_calculate, desc="Calculating metrics")
    for func in pbar:
        pbar.set_description(f"Calculating {func.__name__[1:]} for {len(dataframe)} conditions")
        r.append(
            dataframe[list(inspect.getfullargspec(func).args)]
            .apply((lambda row: func(**row)), axis=1)
            .rename(func.__name__[1:])
        )
    return pd.concat([dataframe] + r, axis=1)


def create_dataset(
    air_temperatures: list[float],  # pylint: disable=unused-argument
    mean_radiant_temperatures: list[float],  # pylint: disable=unused-argument
    air_velocitys: list[float],  # pylint: disable=unused-argument
    relative_humiditys: list[float],  # pylint: disable=unused-argument
    solar_radiations: list[float],  # pylint: disable=unused-argument
    metabolic_rates: list[float],  # pylint: disable=unused-argument
    clo_values: list[float],  # pylint: disable=unused-argument
) -> list[pd.DataFrame]:
    """Create a dataframe from the given inputs, of all possible combinations of data.

    Args:
        air_temperatures (list[float]):
            The air temperatures to consider.
        mean_radiant_temperatures (list[float]):
            The mean radiant temperatures to consider.
        air_velocitys (list[float]):
            The air velocities to consider.
        relative_humiditys (list[float]):
            The relative humidities to consider.
        solar_radiations (list[float]):
            The solar radiations to consider.
        metabolic_rates (list[float]):
            The metabolic rates to consider.
        clo_values (list[float]):
            The clothing insulation values to consider.
    Returns:
        pd.DataFrame:
            The processed results, including inputs and outputs.
    """

    for i in [
        "air_temperatures",
        "mean_radiant_temperatures",
        "air_velocitys",
        "relative_humiditys",
        "solar_radiations",
        "metabolic_rates",
        "clo_values",
    ]:
        if not isinstance(locals()[i], (list, tuple)):
            raise ValueError(f"{i} should be a list.")

    n_iterations = (
        len(air_temperatures)
        * len(mean_radiant_temperatures)
        * len(air_velocitys)
        * len(relative_humiditys)
        * len(solar_radiations)
        * len(metabolic_rates)
        * len(clo_values)
    )

    prod = itertools.product(
        air_temperatures,
        mean_radiant_temperatures,
        air_velocitys,
        relative_humiditys,
        solar_radiations,
        metabolic_rates,
        clo_values,
    )

    df = pd.DataFrame(
        data=prod,
        columns=[
            "air_temperature",
            "mean_radiant_temperature",
            "air_velocity",
            "relative_humidity",
            "solar_radiation",
            "metabolic_rate",
            "clo_value",
        ],
    )

    return df


def calculate_metrics_for_epw(
    epw: EPW,
    metabolic_rate: float = 1,
    clo_value: float = 1,
    metrics_to_calculate: list[Callable] = None,
) -> Path:
    """Create a dataset of possible values based on the content of an EPW file.

    Note:
        This uses conditions present in the EPW file only. For example, if RH is
        only between 20-55% in the EPW, then that is the only range that will be
        calculated.

    Args:
        epw (EPW):
            The EPW file to generate the dataset from.
        metabolic_rate (float, optional):
            The metabolic rate to consider.
            Defaults to 1 which only calculates for sitting.
        clo_value (float, optional):
            The clothing insulation value to consider.
            Defaults to 1 which only calculates for Trousers,
            long-sleeved shirt, long-sleeved sweater, T-shirt.
        metrics_to_calculate (list[Callable], optional):
            A list of callables that calculate the metrics. If this list is None,
            then all metrics will be called :)

    Returns:
        pd.DataFrame:
            Possible conditions and thermal comfort indices for the given EPW file.
    """

    # create directory to store results in
    dataset = (
        DATA_DIR / f"{Path(epw.file_path).stem}_{metabolic_rate:0.1f}_{clo_value:0.1f}.parquet"
    )
    if dataset.exists():
        print(f"Reloading from {dataset}.")
        return pd.read_parquet(dataset)

    # calculate MRT for all hours of the EPW
    _unshaded_mrt = np.array(
        OutdoorSolarCal(
            location=epw.location,
            direct_normal_solar=epw.direct_normal_radiation,
            diffuse_horizontal_solar=epw.diffuse_horizontal_radiation,
            horizontal_infrared=epw.horizontal_infrared_radiation_intensity,
            surface_temperatures=epw.dry_bulb_temperature,
            sky_exposure=1,
        ).mean_radiant_temperature
    )
    _shaded_mrt = np.array(
        OutdoorSolarCal(
            location=epw.location,
            direct_normal_solar=epw.direct_normal_radiation,
            diffuse_horizontal_solar=epw.diffuse_horizontal_radiation,
            horizontal_infrared=epw.horizontal_infrared_radiation_intensity,
            surface_temperatures=epw.dry_bulb_temperature,
            sky_exposure=0,
        ).mean_radiant_temperature
    )

    # create a dataframe from the EPW file
    # translate air velocity to person height
    _ws = np.array(epw.wind_speed.values) * (
        np.log(PERSON_HEIGHT / TERRAIN_ROUGHNESS_LENGTH) / np.log(10 / TERRAIN_ROUGHNESS_LENGTH)
    )
    df = pd.DataFrame.from_dict(
        data={
            "air_velocity": np.tile(_ws, 2),
            "air_temperature": np.tile(epw.dry_bulb_temperature.values, 2),
            "relative_humidity": np.tile(epw.relative_humidity.values, 2),
            "solar_radiation": np.tile(epw.global_horizontal_radiation.values, 2),
            "mean_radiant_temperature": np.concatenate((_unshaded_mrt, _shaded_mrt)),
        }
    )
    df["clo_value"] = clo_value
    df["metabolic_rate"] = metabolic_rate

    # add date and time data for fun things later (e.g. filter by time of day period)
    dts = epw.dry_bulb_temperature.header.analysis_period.datetimes
    df["month_of_year"] = np.tile([dt.month for dt in dts], 2)
    df["hour_of_day"] = np.tile([dt.hour for dt in dts], 2)

    # process!
    df = calculate_metrics(dataframe=df, metrics_to_calculate=metrics_to_calculate)

    # try:
    #     with ProcessPoolExecutor() as executor:
    #         results = []
    #         for _, row in tqdm(
    #             df.iterrows(),
    #             total=df.shape[0],
    #             desc=f"Calculating thermal comfort indices for {Path(epw.file_path).name}",
    #         ):
    #             results.append(
    #                 executor.submit(
    #                     calculate_all_metrics,
    #                     **row.to_dict(),
    #                 ).result()
    #             )
    # except Exception as e:
    #     warnings.warn(f"Failed to run in parallel. Running in series. Error: {e}")
    #     results = []
    #     for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Running in series"):
    #         results.append(calculate_all_metrics(**row.to_dict()))

    # # convert results into dataframe and return
    # df = pd.DataFrame.from_dict(results)

    # save to dataset
    print(f"Writing results to {dataset}.")
    df.to_parquet(dataset, compression="brotli", index=False)

    return df


# def calculate_all_metrics(
#     air_temperature: float,
#     mean_radiant_temperature: float,
#     air_velocity: float,
#     relative_humidity: float,
#     solar_radiation: float = 0,
#     metabolic_rate: float = 1,
#     clo_value: float = 1,
#     atmospheric_pressure: float = 101325,
# ) -> dict[str, float]:
#     """Calculate all thermal comfort indices for a given set of inputs.

#     Args:
#         air_temperature (float):
#             The air temperature in degree Celsius.
#         mean_radiant_temperature (float):
#             The mean radiant temperature in degree Celsius.
#         air_velocity (float):
#             The air velocity in m/s. This is assumed to be given at human level,
#             and is translated to the relevant height per thermal comfort index
#             within this method.
#         relative_humidity (float):
#             The relative humidity in percentage.
#         solar_radiation (float, optional):
#             The solar radiation in W/m2.
#             Default is 0, which is equivalent to no solar radiation.
#         metabolic_rate (float, optional):
#             The metabolic rate in met.
#             Default is 1 met, which is equivalent to sitting.
#         clo_value (float, optional):
#             The clothing insulation in clo.
#             Default is 1 clo, which is equivalent to a business suit.
#         atmospheric_pressure (float, optional):
#             The atmospheric pressure in Pa.
#             Default is 101325 Pa, which is equivalent to standard atmospheric pressure.
#     Returns:
#         dict[str, float]:
#             A dictionary of all thermal comfort indices calculated for the given inputs.
#     """

#     # global assumptions
#     person_age = 36
#     person_sex = 0.5
#     person_mass = 62
#     person_position = "standing"
#     person_additional_work = 0

#     # validations
#     if not 0 <= relative_humidity <= 110:
#         raise ValueError("Relative humidity should be between 0 and 110%.")
#     if not 0 <= solar_radiation <= 1400:
#         raise ValueError("Solar radiation should be between 0 and 1400 W/m2.")
#     if metabolic_rate < 0:
#         raise ValueError("Metabolic rate should be positive.")
#     if clo_value < 0:
#         raise ValueError("Clothing insulation should be positive.")
#     if air_velocity < 0:
#         raise ValueError("Air velocity should be positive.")

#     # pre-calculations - for inputs not provided
#     dewpoint_temperature = dew_point_from_db_rh_fast(
#         db_temp=air_temperature, rel_humid=relative_humidity
#     )
#     air_velocity_10m = air_velocity * (
#         np.log(10 / TERRAIN_ROUGHNESS_LENGTH) / np.log(PERSON_HEIGHT / TERRAIN_ROUGHNESS_LENGTH)
#     )
#     # TODO - create better way of determine ground surface temperature here ... but FAST!
#     # for now it just assumes its linearly correlated with solar radiation to a max of 70C
#     ground_surface_temperature = float(np.interp(solar_radiation, [0, 1400], [air_temperature, 70]))

#     d = {
#         "air_temperature": air_temperature,
#         "mean_radiant_temperature": mean_radiant_temperature,
#         "air_velocity": air_velocity,
#         "relative_humidity": relative_humidity,
#         "solar_radiation": solar_radiation,
#         "metabolic_rate": metabolic_rate,
#         "clo_value": clo_value,
#         "dewpoint_temperature": dewpoint_temperature,
#         "air_velocity_10m": air_velocity_10m,
#     }

#     # do calculations
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")

#         d["actual_sensation_vote"] = actual_sensation_vote(
#             ta=air_temperature, ws=air_velocity, rh=relative_humidity, sr=solar_radiation
#         )
#         d["apparent_temperature"] = apparent_temperature(
#             ta=air_temperature, rh=relative_humidity, ws=air_velocity * 3.6
#         )
#         d["discomfort_index"] = discomfort_index(ta=air_temperature, rh=relative_humidity)
#         d["heat_index"] = heat_index(ta=air_temperature, rh=relative_humidity)

#         d["humidex"] = humidex(
#             ta=air_temperature, tdp=max(0.001, dewpoint_temperature)
#         )  # errors when rh and tdp is 0

#         try:
#             d["physiologic_equivalent_temperature"] = _pet(
#                 ta=air_temperature,
#                 tr=mean_radiant_temperature,
#                 vel=air_velocity,
#                 rh=max(0.001, relative_humidity),  # errors when rh is 0
#                 met=metabolic_rate,
#                 clo=max(0.001, clo_value),  # errors when clo is 0
#                 age=person_age,
#                 sex=person_sex,
#                 ht=PERSON_HEIGHT,
#                 m_body=person_mass,
#                 pos=person_position,
#                 b_press=atmospheric_pressure,
#             )
#         except TimeoutException:
#             d["physiologic_equivalent_temperature"] = np.nan

#         d["standard_effective_temperature"] = predicted_mean_vote(
#             ta=air_temperature,
#             tr=mean_radiant_temperature,
#             vel=air_velocity,
#             rh=relative_humidity,
#             met=metabolic_rate,
#             clo=clo_value,
#             wme=person_additional_work,
#         )["set"]
#         d["thermal_sensation"] = thermal_sensation(
#             ta=air_temperature,
#             ws=air_velocity,
#             rh=relative_humidity,
#             sr=solar_radiation,
#             tground=ground_surface_temperature,
#         )
#         d["universal_thermal_climate_index"] = universal_thermal_climate_index(
#             ta=air_temperature,
#             tr=mean_radiant_temperature,
#             vel=air_velocity_10m,
#             rh=relative_humidity,
#         )
#         d["wet_bulb_globe_temperature"] = wet_bulb_globe_temperature(
#             ta=air_temperature, mrt=mean_radiant_temperature, ws=air_velocity, rh=relative_humidity
#         )
#         d["windchill_temperature"] = windchill_temp(ta=air_temperature, ws=air_velocity)

#     # convert datatypes to smaller float values
#     d = {k: np.float16(v) for k, v in d.items()}

#     return d


# def create_dataset(
#     air_temperatures: list[float],  # pylint: disable=unused-argument
#     mean_radiant_temperatures: list[float],  # pylint: disable=unused-argument
#     air_velocitys: list[float],  # pylint: disable=unused-argument
#     relative_humiditys: list[float],  # pylint: disable=unused-argument
#     solar_radiations: list[float],  # pylint: disable=unused-argument
#     metabolic_rates: list[float],  # pylint: disable=unused-argument
#     clo_values: list[float],  # pylint: disable=unused-argument
#     atmospheric_pressures: list[float],  # pylint: disable=unused-argument
# ) -> pd.DataFrame:
#     """Process the iterations for the given inputs.

#     Args:
#         air_temperatures (list[float]):
#             The air temperatures to consider.
#         mean_radiant_temperatures (list[float]):
#             The mean radiant temperatures to consider.
#         air_velocitys (list[float]):
#             The air velocities to consider.
#         relative_humiditys (list[float]):
#             The relative humidities to consider.
#         solar_radiations (list[float]):
#             The solar radiations to consider.
#         metabolic_rates (list[float]):
#             The metabolic rates to consider.
#         clo_values (list[float]):
#             The clothing insulation values to consider.
#         atmospheric_pressure (list[float]):
#             The atmospheric pressures to consider.
#     Returns:
#         pd.DataFrame:
#             The processed results, including inputs and outputs.
#     """

#     # obtain the variables passed to this function
#     inputs = locals()

#     # validate inputs - all present and correct? (also sort in order of expected downstream args)
#     if not all(isinstance(i, (list, tuple)) for i in inputs.values()):
#         raise ValueError("All inputs should be lists.")

#     arg_names = inspect.getfullargspec(calculate_all_metrics).args
#     variables = {}
#     for arg in arg_names:
#         if arg + "s" not in inputs.keys():
#             raise ValueError(f"{arg} argument is missing in this function.")
#         variables[arg] = list(set(inputs[arg + "s"]))

#     # ask for confirmation before running
#     num_iterations = functools.reduce(operator.mul, map(len, variables.values()), 1)
#     print(f"About to run {num_iterations:,} iterations.")
#     # cont = input(f"About to run {num_iterations:,} iterations. Continue? [y/n]: ")
#     # if not cont.lower().strip() == "y":
#     #     print("You coward! Exiting.")
#     #     return

#     # create set of all possible combinations
#     iterations = itertools.product(*variables.values())

#     # convert to dataframe
#     df = pd.DataFrame(iterations, columns=INPUT_VARIABLES)

#     # process inputs
#     try:
#         with ProcessPoolExecutor() as executor:
#             results = []
#             for _, row in tqdm(
#                 df.iterrows(),
#                 total=df.shape[0],
#                 desc="Calculating thermal comfort indices in parallel.",
#             ):
#                 results.append(executor.submit(calculate_all_metrics, **row.to_dict()).result())
#     except Exception as e:
#         warnings.warn(f"Failed to run in parallel. Running in series. Error: {e}")
#         results = []
#         for _, row in tqdm(
#             df.iterrows(), total=df.shape[0], desc="Calculating thermal comfort indices in series"
#         ):
#             results.append(
#                 calculate_all_metrics(
#                     **row.to_dict(),
#                 )
#             )

#     # convert results into dataframe and return
#     df = pd.DataFrame.from_dict(results)

#     # save to dataset
#     df.to_parquet(
#         DATASET.with_name(f"{DATASET.stem}_{pd.Timestamp.now():%Y%m%d%H%M%S}"),
#         compression="brotli",
#         index=False,
#     )

#     return df
