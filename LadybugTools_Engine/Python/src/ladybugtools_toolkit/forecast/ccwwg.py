import warnings
from pathlib import Path
from typing import Dict, List

import ladybugtools_toolkit
import numpy as np
import pandas as pd
from ladybug.datatype.energyflux import (
    DiffuseHorizontalRadiation,
    DirectNormalRadiation,
    GlobalHorizontalRadiation,
)
from ladybug.datatype.fraction import OpaqueSkyCover, RelativeHumidity, TotalSkyCover
from ladybug.datatype.illuminance import (
    DiffuseHorizontalIlluminance,
    DirectNormalIlluminance,
    GlobalHorizontalIlluminance,
)
from ladybug.datatype.luminance import ZenithLuminance
from ladybug.datatype.pressure import AtmosphericStationPressure
from ladybug.datatype.speed import WindSpeed
from ladybug.datatype.temperature import DryBulbTemperature
from ladybug.epw import EPW, HourlyContinuousCollection, Location, MonthlyCollection
from ladybug.psychrometrics import dew_point_from_db_rh
from ladybug.skymodel import calc_horizontal_infrared
from ladybugtools_toolkit.ladybug_extension.datacollection import from_series, to_series
from scipy import spatial

# load the requisite dataset into memory for querying
# https://burohappold.sharepoint.com/:u:/r/sites/BHoM/Datasets/CCWorldWeatherGen/HadCM3_A2.npz
_DATASET_PATH = (
    Path(ladybugtools_toolkit.__file__).parent.parent / "data" / "HadCM3_A2.npz"
)
_DATASET_PATH = Path(r"C:\ccwwg\datasets\HadCM3_A2.npz")
if not _DATASET_PATH.exists():
    raise FileNotFoundError(
        f"{_DATASET_PATH} does not exist. Forecasting using the HadCM3 datasets will not be possible without this file!"
    )


def _construct_forecast_key(
    emissions_scenario: str, forecast_year: int, variable: str
) -> str:
    """Construct a lookup/key for a HadCM3 precompiled dataset of forecast morph factors."""
    forecast_year_options = [2020, 2050, 2080]
    emissions_scenario_options = ["A2a", "A2b", "A2c"]
    variable_options = [
        "DSWF",
        "MSLP",
        "PREC",
        "RHUM",
        "TCLW",
        "TEMP",
        "TMIN",
        "TMAX",
        "WIND",
    ]

    if forecast_year not in forecast_year_options:
        raise ValueError(
            f"{forecast_year} ({type(forecast_year)}) is not a possible option for this forecast model. Please use one of {forecast_year_options}."
        )

    if emissions_scenario not in emissions_scenario_options:
        raise ValueError(
            f"{emissions_scenario} ({type(emissions_scenario)}) is not a possible option for this forecast model. Please use one of {emissions_scenario_options}."
        )

    if variable not in variable_options:
        raise ValueError(
            f"{variable}  ({type(variable)}) is not a possible option for this forecast model. Please use one of {variable_options}."
        )

    return f"{emissions_scenario}_{variable}_{forecast_year}"


def _query_factors(
    location: Location,
    emissions_scenario: str,
    forecast_year: int,
    variable: str,
) -> List[List[float]]:
    """Query the forecast model for morph factors nearby the target site and weighted interpolate between these."""

    forecast_key = _construct_forecast_key(emissions_scenario, forecast_year, variable)

    with np.load(_DATASET_PATH) as dataset:
        data = dataset[forecast_key]
        grid = dataset["grid"] if variable != "WIND" else dataset["wind_grid"]

    # get the nearest location indices based on the input location
    n_nearest = 4
    distance, nearest_point_indices = spatial.KDTree(grid).query(
        [location.latitude, location.longitude], k=n_nearest
    )

    # construct weights for each index
    weights = 1 - (distance / sum(distance))

    return np.average(data[:, nearest_point_indices], axis=1, weights=weights)


def _factors_to_annual(factors: List[float]) -> List[float]:
    """Cast monthly morphing factors to annual hourly ones."""
    if len(factors) != 12:
        raise ValueError(f"This method won't work ({len(factors)} != 12).")
    year_idx = pd.date_range("2017-01-01 00:00:00", freq="60T", periods=8760)
    month_idx = pd.date_range("2017-01-01 00:00:00", freq="MS", periods=12)

    # expand values across an entire year, filling NaNs where unavailable, and bookend
    annual_values_nans = (
        pd.Series(data=factors, index=month_idx).reindex(year_idx, method=None).values
    )
    annual_values_nans[-1] = annual_values_nans[0]

    # interpolate between NaNs
    return pd.Series(annual_values_nans).interpolate().values


def _forecast_dry_bulb_temperature(
    location: Location,
    dbt_collection: HourlyContinuousCollection,
    emissions_scenario: str,
    forecast_year: int,
) -> HourlyContinuousCollection:
    """Forecast dry bulb temperature using IPCC HadCM3 forecast model."""

    if not isinstance(dbt_collection.header.data_type, DryBulbTemperature):
        raise ValueError(
            f"This method can only forecast for dtype of {DryBulbTemperature}"
        )

    # test for data validity and return the original collection is all "invalid"
    if all(i == 99.9 for i in dbt_collection):
        warnings.warn(
            "The original dry bulb temperature values are all missing. The original data will be returned instead."
        )
        return dbt_collection

    # attempt to transform the input data
    series = to_series(dbt_collection)
    tmin = _factors_to_annual(
        _query_factors(location, emissions_scenario, forecast_year, "TMIN")
    )
    temp = _factors_to_annual(
        _query_factors(location, emissions_scenario, forecast_year, "TEMP")
    )
    tmax = _factors_to_annual(
        _query_factors(location, emissions_scenario, forecast_year, "TMAX")
    )

    dbt_0_monthly_average_daily_max = (
        series.resample("1D")
        .max()
        .resample("MS")
        .mean()
        .reindex(series.index, method="ffill")
    )
    dbt_0_monthly_average_daily_mean = (
        series.resample("MS").mean().reindex(series.index, method="ffill")
    )
    dbt_0_monthly_average_daily_min = (
        series.resample("1D")
        .min()
        .resample("MS")
        .mean()
        .reindex(series.index, method="ffill")
    )
    adbt_m = (tmax - tmin) / (
        dbt_0_monthly_average_daily_max - dbt_0_monthly_average_daily_min
    )
    dbt_new = series + temp + adbt_m * (series - dbt_0_monthly_average_daily_mean)

    # last check to ensure results arent weird
    avg_diff_limit = 20
    if not np.allclose(series, dbt_new, atol=avg_diff_limit):
        warnings.warn(
            "Forecast for dry-bulb temperature returns values beyond feasible range of transformation. The original data will be returned instead."
        )
        return dbt_collection

    return from_series(dbt_new)


def _forecast_relative_humidity(
    location: Location,
    rh_collection: HourlyContinuousCollection,
    emissions_scenario: str,
    forecast_year: int,
) -> HourlyContinuousCollection:
    """Forecast relative humidity using IPCC HadCM3 forecast model."""

    if not isinstance(rh_collection.header.data_type, RelativeHumidity):
        raise ValueError(
            f"This method can only forecast for dtype of {RelativeHumidity}"
        )

    # test for data validity and return the original collection is all "invalid"
    if all(i == 999 for i in rh_collection):
        warnings.warn(
            "The original relative humidity values are all missing. The original data will be returned instead."
        )
        return rh_collection

    # attempt to transform the input data
    series = to_series(rh_collection)
    rhum = _factors_to_annual(
        _query_factors(location, emissions_scenario, forecast_year, "RHUM")
    )

    rh_new = (series + rhum).clip(0, 110)

    # last check to ensure results arent weird
    avg_diff_limit = 10
    if not np.allclose(series, rh_new, atol=avg_diff_limit):
        warnings.warn(
            "Forecast for relative humidity returns values beyond feasible range of transformation. The original data will be returned instead."
        )
        return rh_collection

    return from_series(rh_new)


def _forecast_atmospheric_pressure(
    location: Location,
    ap_collection: HourlyContinuousCollection,
    emissions_scenario: str,
    forecast_year: int,
) -> HourlyContinuousCollection:
    """Forecast atmospheric pressure using IPCC HadCM3 forecast model."""

    if not isinstance(ap_collection.header.data_type, AtmosphericStationPressure):
        raise ValueError(
            f"This method can only forecast for dtype of {AtmosphericStationPressure}"
        )

    # test for data validity and return the original collection is all "invalid"
    if all(i == 999999 for i in ap_collection):
        warnings.warn(
            "The original atmospheric pressure values are all missing. The original data will be returned instead."
        )
        return ap_collection

    # attempt to transform the input data
    series = to_series(ap_collection)
    mslp = _factors_to_annual(
        _query_factors(location, emissions_scenario, forecast_year, "MSLP")
    )

    ap_new = series + mslp

    # last check to ensure results arent weird
    avg_diff_limit = 100
    if not np.allclose(series, ap_new, atol=avg_diff_limit):
        warnings.warn(
            "Forecast for atmospheric pressure returns values beyond feasible range of transformation. The original data will be returned instead."
        )
        return ap_collection

    return from_series(ap_new)


def _calculate_dew_point_temperature(
    dbt_collection: HourlyContinuousCollection,
    rh_collection: HourlyContinuousCollection,
) -> HourlyContinuousCollection:
    """Calculate DPT from composite variables."""

    if all(i == 99.9 for i in dbt_collection) or all(i == 999 for i in rh_collection):
        warnings.warn(
            "The original dry bulb temperature or relative humidity values are all missing. Dew point temperature will be constructed from the default value for missing values."
        )
        return EPW.from_missing_values().dew_point_temperature

    _dbt = to_series(dbt_collection)
    _rh = to_series(rh_collection)

    dpt = []
    for dbt, rh in list(zip(*[_dbt, _rh])):
        dpt.append(dew_point_from_db_rh(dbt, rh))
    return from_series(
        pd.Series(dpt, index=_dbt.index, name="Dew Point Temperature (C)")
    )


def _forecast_wind_speed(
    location: Location,
    ws_collection: HourlyContinuousCollection,
    emissions_scenario: str,
    forecast_year: int,
) -> HourlyContinuousCollection:
    """Forecast wind speed using IPCC HadCM3 forecast model."""

    if not isinstance(ws_collection.header.data_type, WindSpeed):
        raise ValueError(f"This method can only forecast for dtype of {WindSpeed}")

    # test for data validity and return the original collection is all "invalid"
    if all(i == 999 for i in ws_collection):
        warnings.warn(
            "The original wind speed values are all missing. The original data will be returned instead."
        )
        return ws_collection

    # attempt to transform the input data
    series = to_series(ws_collection)
    wind = _factors_to_annual(
        _query_factors(location, emissions_scenario, forecast_year, "WIND")
    )

    ws_new = (1 + wind / 100) * series  # * 0.514444

    # last check to ensure results arent weird
    avg_diff_limit = 8
    if not np.allclose(series, ws_new, atol=avg_diff_limit):
        warnings.warn(
            "Forecast for wind speed returns values beyond feasible range of transformation. The original data will be returned instead."
        )
        return ws_collection

    return from_series(ws_new)


def _forecast_sky_cover(
    location: Location,
    sc_collection: HourlyContinuousCollection,
    emissions_scenario: str,
    forecast_year: int,
) -> HourlyContinuousCollection:
    """Forecast sky cover using IPCC HadCM3 forecast model."""

    if not isinstance(sc_collection.header.data_type, (TotalSkyCover, OpaqueSkyCover)):
        raise ValueError(
            f"This method can only forecast for dtypes of {TotalSkyCover, OpaqueSkyCover}"
        )

    # test for data validity and return the original collection is all "invalid"
    if all(i == 99 for i in sc_collection):
        warnings.warn(
            "The original sky cover values are all missing. The original data will be returned instead."
        )
        return sc_collection

    # attempt to transform the input data
    series = to_series(sc_collection)
    ccov = _factors_to_annual(
        _query_factors(location, emissions_scenario, forecast_year, "TCLW")
    )

    sc_new = (series + (ccov / 10)).clip(0, 10)

    # last check to ensure results arent weird
    avg_diff_limit = 10
    if not np.allclose(series, sc_new, atol=avg_diff_limit):
        warnings.warn(
            "Forecast for sky cover returns values beyond feasible range of transformation. The original data will be returned instead."
        )
        return sc_collection

    return from_series(sc_new)


def _calculate_horizontal_infrared_radiation_intensity(
    osc_collection: HourlyContinuousCollection,
    dbt_collection: HourlyContinuousCollection,
    dpt_collection: HourlyContinuousCollection,
) -> HourlyContinuousCollection:
    """Calculate HIR from composite variables."""

    if (
        all(i == 99.9 for i in dbt_collection)
        or all(i == 99 for i in osc_collection)
        or all(i == 99.9 for i in dpt_collection)
    ):
        warnings.warn(
            "The original OSC, DBT or DPT values are all missing. HIR will be constructed from the default value for missing values."
        )
        return EPW.from_missing_values().horizontal_infrared_radiation_intensity

    _osc = to_series(osc_collection)
    _dbt = to_series(dbt_collection)
    _dpt = to_series(dpt_collection)

    hir = []
    for osc, dbt, dpt in list(zip(*[_osc, _dbt, _dpt])):
        hir.append(calc_horizontal_infrared(osc, dbt, dpt))

    return from_series(
        pd.Series(
            hir, index=_dbt.index, name="Horizontal Infrared Radiation Intensity (W/m2)"
        )
    )


def _forecast_solar(
    location: Location,
    solar_collection: HourlyContinuousCollection,
    emissions_scenario: str,
    forecast_year: int,
) -> HourlyContinuousCollection:
    """Forecast solar variables using IPCC HadCM3 forecast model."""

    if not isinstance(
        solar_collection.header.data_type,
        (
            GlobalHorizontalRadiation,
            GlobalHorizontalIlluminance,
            DirectNormalRadiation,
            DirectNormalIlluminance,
            DiffuseHorizontalRadiation,
            DiffuseHorizontalIlluminance,
            ZenithLuminance,
        ),
    ):
        raise ValueError(
            f"This method can only forecast for dtypes of {GlobalHorizontalRadiation, GlobalHorizontalIlluminance, DirectNormalRadiation, DirectNormalIlluminance, DiffuseHorizontalRadiation, DiffuseHorizontalIlluminance, ZenithLuminance}, not {type(solar_collection.header.data_type)}"
        )

    # test for data validity and return the original collection is all "invalid"
    if isinstance(
        solar_collection.header.data_type,
        (
            GlobalHorizontalRadiation,
            DirectNormalRadiation,
            DiffuseHorizontalRadiation,
            ZenithLuminance,
        ),
    ):
        missing_val = 9999
    else:
        missing_val = 999999
    if all(i == missing_val for i in solar_collection):
        warnings.warn(
            "The original solar values are all missing. The original data will be returned instead."
        )
        return solar_collection

    # attempt to transform the input data
    series = to_series(solar_collection)
    monthly_avg = series.resample("MS").mean()
    temp = 1 + (
        _query_factors(location, emissions_scenario, forecast_year, "DSWF")
        / monthly_avg
    ).reindex(series.index)
    temp[-1] = temp[0]
    sc_new = (series * temp.interpolate()).clip(lower=0)

    # last check to ensure results arent weird
    avg_diff_limit = 100
    if not np.allclose(series, sc_new, atol=avg_diff_limit):
        warnings.warn(
            "Forecast for solar values returns values beyond feasible range of transformation. The original data will be returned instead."
        )
        return solar_collection

    return from_series(sc_new)


def _modify_ground_temperature(
    original_epw: EPW, new_epw: EPW
) -> Dict[str, MonthlyCollection]:
    """Based on changes in DBT from a source and target EPW file, adjust the source monthly ground temperatures accordingly.
    Args:
        original_epw (EPW):
            The source EPW file.
        new_epw (EPW):
            The target EPW file.
    Returns:
        Dict[str, MonthlyCollection]:
            A set of Monthly ground temperature data collections.
    """
    factors = (
        to_series(new_epw.dry_bulb_temperature).resample("MS").mean()
        / to_series(original_epw.dry_bulb_temperature).resample("MS").mean()
    ).values
    new_ground_temperatures = {}
    for depth, collection in original_epw.monthly_ground_temperature.items():
        new_ground_temperatures[depth] = MonthlyCollection(
            header=collection.header,
            values=factors * collection.values,
            datetimes=collection.datetimes,
        )
    return new_ground_temperatures


def forecast_epw(epw: EPW, emissions_scenario: str, forecast_year: int) -> EPW:
    """Forecast an EPW using the methodology provided by
    Climate Change Weather File Generators
    Technical reference manual for the CCWeatherGen and CCWorldWeatherGen tools
    Version 1.2
    Mark F. Jentsch
    Sustainable Energy Research Group
    University of Southampton
    November 2012
    Args:
        epw (EPW):
            The EPW file to transform.
        emissions_scenario (str):
            An emissions scenario to forecast with.
        forecast_year (int):
            The year to forecast.
    Returns:
        EPW:
            A "forecast" EPW file.
    """

    # CONSOLE_LOGGER(
    #     f"Forecasting {Path(epw.file_path)} using IPCC HadCM3 model, {emissions_scenario} emissions scenario for {forecast_year}"
    # )

    # create an "empty" epw object eready to populate
    new_epw = EPW.from_missing_values(epw.is_leap_year)
    new_epw.location = Location(
        latitude=epw.location.latitude,
        longitude=epw.location.longitude,
        source=f"{epw.location.source} {emissions_scenario}-{forecast_year}",
        city=epw.location.city,
        country=epw.location.country,
        elevation=epw.location.elevation,
        state=epw.location.state,
        station_id=epw.location.station_id,
    )
    new_epw.comments_1 = f"{epw.comments_1}. Forecast using transformation factors from the IPCC HadCM3 {emissions_scenario} emissions scenario for {forecast_year} according to the methodology from Jentsch M.F., James P.A.B., Bourikas L. and Bahaj A.S. (2013) Transforming existing weather data for worldwide locations to enable energy and building performance simulation under future climates, Renewable Energy, Volume 55, pp 514-524."
    new_epw.comments_2 = epw.comments_2
    new_epw._file_path = (
        Path(epw.file_path).parent
        / f"{Path(epw.file_path).stem}__HadCM3_{emissions_scenario}_{forecast_year}.epw"
    ).as_posix()

    # copy over variables that aren't going to change
    new_epw.years.values = epw.years.values
    new_epw.wind_direction.values = epw.wind_direction.values
    new_epw.present_weather_observation.values = epw.present_weather_observation.values
    new_epw.present_weather_codes.values = epw.present_weather_codes.values
    new_epw.aerosol_optical_depth.values = epw.aerosol_optical_depth.values
    new_epw.snow_depth.values = epw.snow_depth.values
    new_epw.days_since_last_snowfall.values = epw.days_since_last_snowfall.values
    new_epw.albedo.values = epw.albedo.values
    new_epw.liquid_precipitation_depth.values = epw.liquid_precipitation_depth.values
    new_epw.liquid_precipitation_quantity.values = (
        epw.liquid_precipitation_quantity.values
    )
    new_epw.precipitable_water.values = epw.precipitable_water.values

    # forecast variables
    new_epw.dry_bulb_temperature.values = _forecast_dry_bulb_temperature(
        epw.location, epw.dry_bulb_temperature, emissions_scenario, forecast_year
    ).values
    new_epw.relative_humidity.values = _forecast_relative_humidity(
        epw.location, epw.relative_humidity, emissions_scenario, forecast_year
    ).values
    new_epw.atmospheric_station_pressure.values = _forecast_atmospheric_pressure(
        epw.location,
        epw.atmospheric_station_pressure,
        emissions_scenario,
        forecast_year,
    )
    new_epw.dew_point_temperature.values = _calculate_dew_point_temperature(
        new_epw.dry_bulb_temperature, new_epw.relative_humidity
    ).values
    new_epw.wind_speed.values = _forecast_wind_speed(
        epw.location, epw.wind_speed, emissions_scenario, forecast_year
    ).values
    new_epw.total_sky_cover.values = _forecast_sky_cover(
        epw.location, epw.total_sky_cover, emissions_scenario, forecast_year
    ).values
    new_epw.opaque_sky_cover.values = _forecast_sky_cover(
        epw.location, epw.opaque_sky_cover, emissions_scenario, forecast_year
    ).values
    new_epw.horizontal_infrared_radiation_intensity.values = (
        _calculate_horizontal_infrared_radiation_intensity(
            new_epw.opaque_sky_cover,
            new_epw.dry_bulb_temperature,
            new_epw.dew_point_temperature,
        ).values
    )
    new_epw.global_horizontal_radiation.values = _forecast_solar(
        epw.location, epw.global_horizontal_radiation, emissions_scenario, forecast_year
    ).values
    new_epw.direct_normal_radiation.values = _forecast_solar(
        epw.location, epw.direct_normal_radiation, emissions_scenario, forecast_year
    ).values
    new_epw.diffuse_horizontal_radiation.values = _forecast_solar(
        epw.location,
        epw.diffuse_horizontal_radiation,
        emissions_scenario,
        forecast_year,
    ).values
    new_epw.global_horizontal_illuminance.values = _forecast_solar(
        epw.location,
        epw.global_horizontal_illuminance,
        emissions_scenario,
        forecast_year,
    ).values
    new_epw.direct_normal_illuminance.values = _forecast_solar(
        epw.location, epw.direct_normal_illuminance, emissions_scenario, forecast_year
    ).values
    new_epw.diffuse_horizontal_illuminance.values = _forecast_solar(
        epw.location,
        epw.diffuse_horizontal_illuminance,
        emissions_scenario,
        forecast_year,
    ).values
    new_epw.zenith_luminance.values = _forecast_solar(
        epw.location,
        epw.zenith_luminance,
        emissions_scenario,
        forecast_year,
    ).values

    # modify ground temperatures based on differences in EPW DBT
    new_epw._monthly_ground_temps = _modify_ground_temperature(epw, new_epw)

    return new_epw
