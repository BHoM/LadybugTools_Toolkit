"""Methods for determining the feasible ranges of thermal comfort indices 
based on simple modification of inputs to their calculation."""

# pylint: disable=C0302,E0401,E1101
import json
import warnings
from calendar import month_abbr
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug_comfort.collection.pet import PET, OutdoorSolarCal, PhysiologicalEquivalentTemperature
from ladybug_comfort.collection.pmv import PMV, StandardEffectiveTemperature
from ladybug_comfort.collection.utci import UTCI
from python_toolkit.bhom.logging import CONSOLE_LOGGER
from tqdm import tqdm

from ..helpers import evaporative_cooling_effect_collection
from ..ladybug_extension.datacollection import average, collection_to_series

# pylint: enable=E0401


class ThermalComfortIndex(Enum):
    """An enumeration of the comfort indices that can be calculated."""

    UTCI = UniversalThermalClimateIndex
    PET = PhysiologicalEquivalentTemperature
    SET = StandardEffectiveTemperature

    @property
    def default_met_rate(self) -> float:
        """Get the default MET rate for the comfort index."""
        match self.value:
            case ThermalComfortIndex.UTCI.value:
                return None
            case ThermalComfortIndex.PET.value:
                return 2.4
            case ThermalComfortIndex.SET.value:
                return 2.4
            case _:
                raise NotImplementedError(f"{self.name} is not supported.")

    @property
    def default_clo_value(self) -> float:
        """Get the default CLO value for the comfort index."""
        match self.value:
            case ThermalComfortIndex.UTCI.value:
                return None
            case ThermalComfortIndex.PET.value:
                return 0.7
            case ThermalComfortIndex.SET.value:
                return 0.7
            case _:
                raise NotImplementedError(f"{self.name} is not supported.")

    @property
    def default_comfort_limits(self) -> tuple[float]:
        """Get the default comfort limits for the comfort index.

        Reference:
            Blazejczyk, Krzysztof, Yoram Epstein, Gerd Jendritzky,
            Henning Staiger, and Birger Tinz. “Comparison of UTCI to
            Selected Thermal Indices.” International Journal of
            Biometeorology 56, no. 3 (May 2012): 515–35.
            https://doi.org/10.1007/s00484-011-0453-2.

        """
        match self.value:
            case ThermalComfortIndex.UTCI.value:
                return (9, 26)
            case ThermalComfortIndex.PET.value:
                return (18, 23)
            case ThermalComfortIndex.SET.value:
                return (17, 30)
            case _:
                raise NotImplementedError(f"{self.name} is not supported.")

    @property
    def default_wind_modifier(self) -> float:
        """Get the default wind modifier for the comfort index. This
        value denotes the factor applied to wind in order to make it
        applicable to the different comfort index - usually due to the
        different wind speed height for that index."""
        match self.value:
            case ThermalComfortIndex.UTCI.value:
                return 1.0
            case ThermalComfortIndex.PET.value:
                return 2 / 3
            case ThermalComfortIndex.SET.value:
                return 2 / 3
            case _:
                raise NotImplementedError(f"{self.name} is not supported.")


def thermal_comfort_data(
    epw: EPW,
    thermal_comfort_index: ThermalComfortIndex,
    shade_proportion: float = 0,
    additional_air_moisture: float = 0,
    wind_multiplier: float = 1,
    met_rate: float = None,
    clo_value: float = None,
) -> pd.DataFrame:
    """Calculate the thermal comfort index for a given EPW file and parameters.

    Args:
        epw (EPW):
            An EPW object with the weather data.
        thermal_comfort_index (ThermalComfortIndex):
            The thermal comfort index to calculate.
        shade_proportion (float, optional):
            A number between 0 and 1 that represents the proportion of shade
            given to an abstract point. Default is 0.
        additional_air_moisture (float, optional):
            A number between 0 and 1 that represents the effectiveness of
            evaporative cooling on the air. Default is 0. 1 would be fully
            saturated air.
        wind_multiplier (float, optional):
            A multiplier for the wind speed. Default is 1.
        met_rate (float, optional):
            The metabolic rate of the person in met. Default is None, which
            would use the default value for the provided comfort index.
        clo_value (float, optional):
            The clothing insulation value in clo. Default is None, which would
            use the default value for the provided comfort index.

    Returns:
        pd.DataFrame: A pandas DataFrame with the thermal comfort index values.

    Note:
        - This method does not account for surface temperature heating from
        solar radiation.
        - The method will save the results or the provided EPW file in the
        default Ladybug simulation folder named "thermal_comfort".
        - UTCI cannot accept MET and CLO values and these will be ignored if
        provided.
    """

    # get the met and clo values if needed
    if met_rate is None:
        met_rate = thermal_comfort_index.default_met_rate
    if clo_value is None:
        clo_value = thermal_comfort_index.default_clo_value

    if thermal_comfort_index.value == ThermalComfortIndex.UTCI.value:
        if met_rate is not None:
            CONSOLE_LOGGER.warning("UTCI does not accept MET rate. It will be ignored.")
        if clo_value is not None:
            CONSOLE_LOGGER.warning("UTCI does not accept CLO value. It will be ignored.")

    # validate inputs
    if not isinstance(epw, EPW):
        raise ValueError("epw must be of type EPW.")
    if not isinstance(thermal_comfort_index, ThermalComfortIndex):
        raise ValueError("thermal_comfort_index must be of type ThermalComfortIndex.")
    if not (shade_proportion >= 0) & (shade_proportion <= 1):
        raise ValueError("shade_proportion must be between 0 and 1.")
    if not (additional_air_moisture >= 0) & (additional_air_moisture <= 1):
        raise ValueError("additional_air_moisture must be between 0 and 1.")
    if wind_multiplier < 0:
        raise ValueError("wind_multiplier must be greater than 0.")
    if not (met_rate is None or met_rate >= 0):
        raise ValueError("met_rate must be greater than or equal to 0.")
    if not (clo_value is None or clo_value >= 0):
        raise ValueError("clo_value must be greater than or equal to 0.")

    # create folder to store results in
    root_dir = Path(hb_folders.default_simulation_folder) / "thermal_comfort"
    out_dir = root_dir / Path(epw.file_path).name
    out_dir.mkdir(parents=True, exist_ok=True)

    # save epw file to the folder
    epw_file = out_dir / Path(epw.file_path).name
    if not epw_file.exists():
        epw.save(epw_file)

    # create config identifier
    input_id = f"{shade_proportion:0.1f}_{additional_air_moisture:0.1f}_{wind_multiplier:0.1f}"
    config_id = f"{thermal_comfort_index.name}_{input_id}_{met_rate}_{clo_value}"

    # load existing file if it exists
    collection = None
    comfort_index_file = out_dir / f"{config_id}.json"
    if comfort_index_file.exists():
        with open(comfort_index_file, "r", encoding="utf-8") as fp:
            collection = HourlyContinuousCollection.from_dict(json.load(fp))

    if collection is None:
        # calculate MRT components
        mrt_shaded_file = out_dir / "MRTshaded.json"
        mrt_unshaded_file = out_dir / "MRTunshaded.json"
        if mrt_unshaded_file.exists():
            with open(mrt_unshaded_file, "r", encoding="utf-8") as fp:
                mrt_unshaded = HourlyContinuousCollection.from_dict(json.load(fp))
        else:
            mrt_unshaded: HourlyContinuousCollection = OutdoorSolarCal(
                location=epw.location,
                direct_normal_solar=epw.direct_normal_radiation,
                diffuse_horizontal_solar=epw.diffuse_horizontal_radiation,
                horizontal_infrared=epw.horizontal_infrared_radiation_intensity,
                surface_temperatures=epw.dry_bulb_temperature,
            ).mean_radiant_temperature
            with open(mrt_unshaded_file, "w", encoding="utf-8") as fp:
                json.dump(mrt_unshaded.to_dict(), fp)
        if mrt_shaded_file.exists():
            with open(mrt_shaded_file, "r", encoding="utf-8") as fp:
                mrt_shaded = HourlyContinuousCollection.from_dict(json.load(fp))
        else:
            mrt_shaded = mrt_unshaded.get_aligned_collection(epw.dry_bulb_temperature.values)
            with open(mrt_shaded_file, "w", encoding="utf-8") as fp:
                json.dump(mrt_shaded.to_dict(), fp)
        mrt = average(
            [mrt_shaded, mrt_unshaded],
            [shade_proportion, 1 - shade_proportion],
        )

        # calculate DBT and RH components
        dbt_file = out_dir / f"DBT_{additional_air_moisture:0.1f}.json"
        rh_file = out_dir / f"RH_{additional_air_moisture:0.1f}.json"
        dbt, rh = None, None
        if dbt_file.exists():
            with open(dbt_file, "r", encoding="utf-8") as fp:
                dbt = HourlyContinuousCollection.from_dict(json.load(fp))
        if rh_file.exists():
            with open(rh_file, "r", encoding="utf-8") as fp:
                rh = HourlyContinuousCollection.from_dict(json.load(fp))
        if dbt is None or rh is None:
            dbt, rh = evaporative_cooling_effect_collection(
                epw=epw, evaporative_cooling_effectiveness=additional_air_moisture
            )
            with open(dbt_file, "w", encoding="utf-8") as fp:
                json.dump(dbt.to_dict(), fp)
            with open(rh_file, "w", encoding="utf-8") as fp:
                json.dump(rh.to_dict(), fp)

        # calculate wind speed components

        vel = epw.wind_speed * wind_multiplier * thermal_comfort_index.default_wind_modifier

        # calculate the thermal comfort index
        match thermal_comfort_index.value:
            case ThermalComfortIndex.UTCI.value:
                collection = UTCI(
                    air_temperature=dbt,
                    rel_humidity=rh,
                    wind_speed=vel,
                    rad_temperature=mrt,
                ).universal_thermal_climate_index
            case ThermalComfortIndex.PET.value:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    collection = PET(
                        air_temperature=dbt,
                        rel_humidity=rh,
                        rad_temperature=mrt,
                        air_speed=vel,
                        barometric_pressure=epw.atmospheric_station_pressure,
                        met_rate=met_rate,
                        clo_value=clo_value,
                    ).physiologic_equivalent_temperature
            case ThermalComfortIndex.SET.value:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    collection = PMV(
                        air_temperature=dbt,
                        rel_humidity=rh,
                        rad_temperature=mrt,
                        air_speed=vel,
                        met_rate=met_rate,
                        clo_value=clo_value,
                    ).standard_effective_temperature
            case _:
                raise NotImplementedError(f"{thermal_comfort_index.name} is not supported.")
        collection.header.metadata = {
            "EPW": Path(epw.file_path).name,
            "ComfortIndex": thermal_comfort_index.name,
            "ShadeProportion": shade_proportion,
            "AirMoisture": additional_air_moisture,
            "WindMultiplier": wind_multiplier,
            "MET": met_rate,
            "CLO": clo_value,
        }
        with open(comfort_index_file, "w", encoding="utf-8") as fp:
            json.dump(collection.to_dict(), fp)

    # convert collection to pandas Series
    df = collection_to_series(collection).to_frame()
    df.columns = pd.MultiIndex.from_tuples(
        [collection.header.metadata.values()],
        names=collection.header.metadata.keys(),
    )

    return df


def thermal_comfort_datas(
    epws: tuple[EPW],
    thermal_comfort_indices: tuple[ThermalComfortIndex] = (ThermalComfortIndex.UTCI,),
    shade_proportions: tuple[float] = (0, 1),
    additional_air_moistures: tuple[float] = (0, 0.7),
    wind_multipliers: tuple[float] = (0, 1, 1.1),
    met_rates: tuple[float] = (None,),
    clo_values: tuple[float] = (None,),
    run_parallel: bool = False,
) -> pd.DataFrame:
    """Calculate multiple thermal comfort indices - this is a wrapper around
    another function that makes parallelisation more efficient.

    Args:
        epws (tuple[EPW]):
            A tuple of EPW objects with the weather data.
        thermal_comfort_indices (tuple[ThermalComfortIndex], optional):
            A tuple of thermal comfort indices to calculate. Default is (UTCI).
        shade_proportions (tuple[float], optional):
            A tuple of numbers between 0 and 1 that represents the proportion of
            shade given to an abstract point. Default is (0, 1) which would
            simulate no-shade and full-shade.
        additional_air_moistures (tuple[float], optional):
            A tuple of numbers between 0 and 1 that represents the effectiveness
            of evaporative cooling on the air. Default is (0, 0.7) which would
            simulate no addiitonal moisture and 70% effective moisture
            addition to air - typical of PDEC tower.
        wind_multipliers (tuple[float], optional):
            A tuple of multipliers for the wind speed. Default is (0, 1.1)
            which would simulate no-wind and wind + 10%.
        met_rates (tuple[float], optional):
            A tuple of metabolic rates of the person in met. Default is (None)
            which just simulates the defualt for the provided thermal comfort
            index.
        clo_values (tuple[float], optional):
            A tuple of clothing insulation values in clo. Default is (None)
            which just simulates the defualt for the provided thermal comfort
            index.
        run_parallel (bool, optional):
            Set to True to run the calculations in parallel. Default is False.

    Returns:
        pd.DataFrame: A pandas DataFrame with the thermal comfort index values.
    """

    # validation
    for arg, val in locals().items():
        if arg == "run_parallel":
            continue
        if not isinstance(val, (list, tuple)):
            raise ValueError(f"{arg} must be iterable.")
    if not all(isinstance(i, EPW) for i in epws):
        raise ValueError("All epws must be of type EPW.")
    if not all(isinstance(i, ThermalComfortIndex) for i in thermal_comfort_indices):
        raise ValueError("All thermal_comfort_indices must be of type ThermalComfortIndex.")
    if not all((i >= 0) & (i <= 1) for i in shade_proportions):
        raise ValueError("All shade_proportions must be between 0 and 1.")
    if not all((i >= 0) & (i <= 1) for i in additional_air_moistures):
        raise ValueError("All additional_air_moistures must be between 0 and 1.")
    if not all(i >= 0 for i in wind_multipliers):
        raise ValueError("All wind_multipliers must be greater than 0.")

    # create a list of all possible combinations of the input values
    target_iterations = []
    for epw in epws:
        # create folder to store results in
        root_dir = Path(hb_folders.default_simulation_folder) / "thermal_comfort"
        out_dir = root_dir / Path(epw.file_path).name
        out_dir.mkdir(parents=True, exist_ok=True)

        # save epw file to the folder
        epw_file = out_dir / Path(epw.file_path).name
        if not epw_file.exists():
            epw.save(epw_file)

        for tci in thermal_comfort_indices:
            for sp in shade_proportions:
                # calculate MRT components - included here to allow parallelisation
                mrt_shaded_file = out_dir / "MRTshaded.json"
                mrt_unshaded_file = out_dir / "MRTunshaded.json"
                if not mrt_unshaded_file.exists():
                    mrt_unshaded: HourlyContinuousCollection = OutdoorSolarCal(
                        location=epw.location,
                        direct_normal_solar=epw.direct_normal_radiation,
                        diffuse_horizontal_solar=epw.diffuse_horizontal_radiation,
                        horizontal_infrared=epw.horizontal_infrared_radiation_intensity,
                        surface_temperatures=epw.dry_bulb_temperature,
                    ).mean_radiant_temperature
                    with open(mrt_unshaded_file, "w", encoding="utf-8") as fp:
                        json.dump(mrt_unshaded.to_dict(), fp)
                if not mrt_shaded_file.exists():
                    mrt_shaded = mrt_unshaded.get_aligned_collection(
                        epw.dry_bulb_temperature.values
                    )
                    with open(mrt_shaded_file, "w", encoding="utf-8") as fp:
                        json.dump(mrt_shaded.to_dict(), fp)
                for am in additional_air_moistures:
                    # calculate DBT and RH components - included here to allow parallelisation
                    dbt_file = out_dir / f"DBT_{am:0.1f}.json"
                    rh_file = out_dir / f"RH_{am:0.1f}.json"
                    if not dbt_file.exists() or not rh_file.exists():
                        dbt, rh = evaporative_cooling_effect_collection(
                            epw=epw, evaporative_cooling_effectiveness=am
                        )
                        with open(dbt_file, "w", encoding="utf-8") as fp:
                            json.dump(dbt.to_dict(), fp)
                        with open(rh_file, "w", encoding="utf-8") as fp:
                            json.dump(rh.to_dict(), fp)
                    for wm in wind_multipliers:
                        if tci.value == ThermalComfortIndex.UTCI.value:
                            target_iterations.append(
                                {
                                    "epw": epw,
                                    "thermal_comfort_index": tci,
                                    "shade_proportion": sp,
                                    "additional_air_moisture": am,
                                    "wind_multiplier": wm,
                                    "met_rate": None,
                                    "clo_value": None,
                                }
                            )
                        else:
                            for mr in met_rates:
                                for cv in clo_values:
                                    target_iterations.append(
                                        {
                                            "epw": epw,
                                            "thermal_comfort_index": tci,
                                            "shade_proportion": sp,
                                            "additional_air_moisture": am,
                                            "wind_multiplier": wm,
                                            "met_rate": mr,
                                            "clo_value": cv,
                                        }
                                    )
    # shuffle the iterations
    np.random.shuffle(target_iterations)

    df = []
    if run_parallel:
        # run calculations in parallel
        l = len(target_iterations)
        with tqdm(total=l) as pbar:
            pbar.set_description("Calculating thermal comfort indices")
            with ThreadPoolExecutor() as executor:
                CONSOLE_LOGGER.disabled = True
                futures = [
                    executor.submit(thermal_comfort_data, **kwargs) for kwargs in target_iterations
                ]
                CONSOLE_LOGGER.disabled = False
                for future in as_completed(futures):
                    df.append(future.result())
                    pbar.update(1)
    else:
        pbar = tqdm(target_iterations)
        for it in pbar:
            pbar.set_description(
                f"Processing {Path(it['epw'].file_path).stem} ({it['thermal_comfort_index'].name}-{it['shade_proportion']:0.1f}-{it['additional_air_moisture']:0.1f}-{it['wind_multiplier']:0.1f}-{it['met_rate']}-{it['clo_value']})"
            )
            df.append(thermal_comfort_data(**it))

    return pd.concat(df, axis=1).sort_index(axis=1)


def thermal_comfort_bounds(
    epw: EPW,
    thermal_comfort_index: ThermalComfortIndex,
    shade_limits: tuple[float] = (0, 1),
    wind_limits: tuple[float] = (0, 1.1),
    moisture_limits: tuple[float] = (0, 0.7),
    met_rate: float = None,
    clo_value: float = None,
    show_all: bool = False,
) -> pd.DataFrame:
    """Calculate the minimum and maximum values of the thermal comfort index
    for a given EPW file and parameters.

    Args:
        epw (EPW):
            An EPW object with the weather data.
        thermal_comfort_index (ThermalComfortIndex):
            The thermal comfort index to calculate.
        shade_limits (tuple[float], optional):
            A tuple of two numbers between 0 and 1 that represents the range of
            shade given to an abstract point. Default is (0, 1).
        wind_limits (tuple[float], optional):
            A tuple of two numbers between 0 and 1.1 that represents the range of
            wind speed multipliers. Default is (0, 1.1).
        moisture_limits (tuple[float], optional):
            A tuple of two numbers between 0 and 0.7 that represents the range of
            effectiveness of evaporative cooling on the air. Default is (0, 0.7).
        met_rate (float, optional):
            The metabolic rate of the person in met. Default is None, which would
            use the default value for the provided comfort index.
        clo_value (float, optional):
            The clothing insulation value in clo. Default is None, which would
            use the default value for the provided comfort index.
        show_all (bool, optional):
            Set to True to return all anual hourly values instead of just the
            min and max. Default is False. This is useful for debugging.

    Returns:
        pd.DataFrame: A pandas DataFrame with the min and max values of the thermal
        comfort index.
    """

    if len(shade_limits) != 2:
        raise ValueError("shade_limits should be a tuple of two values.")
    if len(wind_limits) != 2:
        raise ValueError("wind_limits should be a tuple of two values.")
    if len(moisture_limits) != 2:
        raise ValueError("moisture_limits should be a tuple of two values.")

    # run calculation of thermal comfort indices (timeseries)
    df = thermal_comfort_datas(
        epws=[epw],
        thermal_comfort_indices=[thermal_comfort_index],
        shade_proportions=shade_limits,
        wind_multipliers=wind_limits,
        additional_air_moistures=moisture_limits,
        met_rates=[met_rate],
        clo_values=[clo_value],
        run_parallel=False,
    )

    if show_all:
        return df

    # get the min and max values for each row
    min_values = df.min(axis=1).rename(thermal_comfort_index.name)
    max_values = df.max(axis=1).rename(thermal_comfort_index.name)
    df = pd.concat([min_values, max_values], axis=1)
    df.columns = pd.MultiIndex.from_tuples(
        [(thermal_comfort_index.name, "Min"), (thermal_comfort_index.name, "Max")],
        names=["Thermal Comfort Index", "Bounds"],
    )

    return df


def thermal_comfort_summary(
    epw: EPW,
    thermal_comfort_index: ThermalComfortIndex,
    comfort_limits: tuple[float] = None,
    hour_limits: tuple[int] = (0, 23),
    shade_limits: tuple[float] = (0, 1),
    wind_limits: tuple[float] = (0, 1.1),
    moisture_limits: tuple[float] = (0, 0.7),
    met_rate: float = None,
    clo_value: float = None,
    formatted: bool = False,
) -> pd.DataFrame:
    """Return the proportion of hours within the specified range for each month.

    ARGS:
        epw (EPW):
            An EPW object with the weather data.
        thermal_comfort_index (ThermalComfortIndex):
            The thermal comfort index to calculate.
        comfort_limits (tuple[float], optional):
            A tuple of two numbers that represents the range of comfort for the
            thermal comfort index. Default is None, which would use the default
            comfort limits for the provided comfort index.
        hour_limits (tuple[int], optional):
            A tuple of two numbers that represents the range of hours to consider.
            Default is (0, 23) which would consider all hours.
        shade_limits (tuple[float], optional):
            A tuple of two numbers between 0 and 1 that represents the range of
            shade given to an abstract point. Default is (0, 1).
        wind_limits (tuple[float], optional):
            A tuple of two numbers between 0 and 1.1 that represents the range of
            wind speed multipliers. Default is (0, 1.1).
        moisture_limits (tuple[float], optional):
            A tuple of two numbers between 0 and 0.7 that represents the range of
            effectiveness of evaporative cooling on the air. Default is (0, 0.7).
        met_rate (float, optional):
            The metabolic rate of the person in met. Default is None, which would
            use the default value for the provided comfort index.
        clo_value (float, optional):
            The clothing insulation value in clo. Default is None, which would
            use the default value for the provided comfort index.
        formatted (bool, optional):
            Set to True to return a formatted DataFrame. Default is False.

    Returns:
        pd.DataFrame | pd.io.formats.style.Style:
            A pandas DataFrame with the proportion of time within the
            comfort range for each month -OR- a formatted DataFrame.
    """

    if comfort_limits is None:
        comfort_limits = thermal_comfort_index.default_comfort_limits
    if len(comfort_limits) != 2:
        raise ValueError("comfort_limits should be a tuple of two values.")
    if comfort_limits[0] > comfort_limits[1]:
        raise ValueError("comfort_limits should be in ascending order.")

    if len(hour_limits) != 2:
        raise ValueError("hour_limits should be a tuple of two values.")
    for hour in hour_limits:
        if hour < 0 or hour > 23:
            raise ValueError("hour_limits should be between 0 and 23.")

    if met_rate is None:
        met_rate = thermal_comfort_index.default_met_rate
    if clo_value is None:
        clo_value = thermal_comfort_index.default_clo_value

    # run calculation
    df = thermal_comfort_bounds(
        epw=epw,
        thermal_comfort_index=thermal_comfort_index,
        shade_limits=shade_limits,
        wind_limits=wind_limits,
        moisture_limits=moisture_limits,
        met_rate=met_rate,
        clo_value=clo_value,
        show_all=False,
    )

    # create filter/mask
    if hour_limits[0] < hour_limits[1]:
        mask = (df.index.hour >= hour_limits[0]) & (df.index.hour <= hour_limits[1])
    else:
        mask = (df.index.hour >= hour_limits[0]) | (df.index.hour <= hour_limits[1])

    # filter dataset
    df = df[mask]

    # count the proportion of hours within the comfort limits
    threshold_datasets = {
        f"Too cold (<{min(comfort_limits)}°C)": df < min(comfort_limits),
        f"Comfortable ({min(comfort_limits)}°C to {max(comfort_limits)}°C)": (
            df >= min(comfort_limits)
        )
        & (df <= max(comfort_limits)),
        f"Too hot (>{max(comfort_limits)}°C)": df > max(comfort_limits),
    }
    new_df = []
    for k, v in threshold_datasets.items():
        # groupby month and get proportion meeting targets
        _sums = v.groupby(v.index.month).sum()
        _counts = v.groupby(v.index.month).count()
        _temp = _sums / _counts

        # sort each row by size to get low/high proportion
        _data = []
        for _, row in _temp.iterrows():
            _data.append([min(row), max(row)])
        _temp = pd.DataFrame(data=_data, index=_temp.index)
        _temp.columns = pd.MultiIndex.from_tuples(
            [
                (thermal_comfort_index.name, k, "Lowest proportion"),
                (thermal_comfort_index.name, k, "Highest proportion"),
            ],
            names=["Index", "Condition", "Bound"],
        )
        _temp.index = [month_abbr[i] for i in _temp.index]
        new_df.append(_temp)

    new_df = pd.concat(new_df, axis=1)

    if formatted:
        caption = f"Feasible proportion of time achieving target comfort {thermal_comfort_index.name} conditions ({min(comfort_limits)}°C to {max(comfort_limits)}°C) using {epw}"
        caption += f" from {hour_limits[0]:02d}:00 to {hour_limits[1]:02d}:59"
        caption += f". Including effects from shade ({min(shade_limits):0.0%} to {max(shade_limits):0.0%}), wind ({min(wind_limits):0.0%} to {max(wind_limits):0.0%}), and air moisture ({min(moisture_limits):0.0%} to {max(moisture_limits):0.0%})."
        if (met_rate is not None) and (clo_value is not None):
            caption = caption[:-1]
            caption += (
                f", and using a MET rate of {met_rate:0.1f} and CLO value of {clo_value:0.1f}."
            )
        return (
            new_df.style.set_caption(caption)
            .background_gradient(cmap="Blues", subset=new_df.columns[0:2], low=0, high=1, axis=None)
            .background_gradient(
                cmap="Greens", subset=new_df.columns[2:4], low=0, high=1, axis=None
            )
            .background_gradient(cmap="Reds", subset=new_df.columns[4:6], low=0, high=1, axis=None)
            .format("{:.1%}")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("color", "#555555"),
                            ("font-size", "x-small"),
                            ("caption-side", "bottom"),
                            ("font-style", "bold"),
                            ("text-align", "left"),
                        ],
                    },
                    {
                        "selector": "td:hover",
                        "props": [
                            ("background-color", "#ffffb3"),
                            ("color", "black"),
                        ],
                    },
                    {
                        "selector": ".index_name",
                        "props": [("color", "#555555"), ("font-weight", "normal")],
                    },
                    {
                        "selector": "th:not(.index_name)",
                        "props": [
                            ("background-color", "white"),
                            ("color", "black"),
                        ],
                    },
                ]
            )
        )

    return new_df
