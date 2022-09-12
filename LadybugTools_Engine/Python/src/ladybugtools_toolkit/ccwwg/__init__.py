from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import List, Union

import fortranformat as ff
import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection, MonthlyCollection
from ladybug.epw import EPW
from ladybug.location import Location
from ladybug.psychrometrics import dew_point_from_db_rh
from ladybug.skymodel import (
    calc_horizontal_infrared,
    estimate_illuminance_from_irradiance,
    zhang_huang_solar_split,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.ladybug_extension.epw.solar_altitude import (
    solar_altitude as get_solar_altitude,
)
from scipy import spatial


class ForecastYear(Enum):
    _2020 = 2020
    _2050 = 2050
    _2080 = 2080

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{__class__.__name__}[{self}]"

    @classmethod
    def from_str(cls, str: str) -> ForecastYear:
        return getattr(ForecastYear, f"_{str}")

    @classmethod
    def from_int(cls, int: int) -> ForecastYear:
        return getattr(ForecastYear, f"_{int}")


class EmissionsScenario(Enum):
    A2a = "A2a"
    A2b = "A2b"
    A2c = "A2c"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{__class__.__name__}[{self}]"

    @classmethod
    def from_str(cls, str: str) -> EmissionsScenario:
        return getattr(EmissionsScenario, f"{str}")


def load_points_kml(kml_path: Path) -> List[List[float]]:
    points = []
    with open(kml_path, "r") as f:
        for line in f:
            if line.startswith("<coordinates>"):
                line = (
                    line.replace("<coordinates>", "")
                    .replace("</coordinates>", "")
                    .strip()
                )
                lat, long = line.split(",")
                points.append([float(lat), float(long)])
    return points


def load_variable_dif(file_path: Path) -> List[List[float]]:
    with open(file_path, "r") as fp:
        data = fp.readlines()
    starts = [n for n, i in enumerate(data) if i.startswith("IPCC")]
    starts += [len(data)]
    header_rows = 6
    indices = list(zip(starts, starts[1:]))

    config_row = data[indices[0][0] : indices[0][1] + header_rows]
    n_values = int(config_row[-1].split()[0])
    format = config_row[-1].split()[3]

    reader = ff.FortranRecordReader(format)

    values = []
    for x, y in indices:
        temp = []
        for row in data[x:y][header_rows:]:
            temp.extend(reader.read(row))
        values.append(temp[:n_values])

    return values


def nearest_n_point_indices(
    points: List[List[float]], location: Location, n: int
) -> List[int]:
    _, nearest_point_indices = spatial.KDTree(points).query(
        [location.latitude, location.longitude], k=n
    )
    return nearest_point_indices


def construct_file_path(
    root_directory: Path,
    climate_variable: str,
    emissions_scenario: EmissionsScenario,
    forecast_year: ForecastYear,
) -> Path:
    file_path = (
        root_directory
        / f"HADCM3_{emissions_scenario.value}_{climate_variable}_{forecast_year.value}.dif"
    )

    if file_path.exists():
        return file_path
    else:
        raise FileNotFoundError(
            f"It doesn't seem as though a dataset is available for {file_path.name}."
        )


class TranslationFactors:
    def __init__(
        self,
        location: Location,
        emissions_scenario: EmissionsScenario,
        forecast_year: ForecastYear,
        DSWF: pd.Series = None,
        MSLP: pd.Series = None,
        PREC: pd.Series = None,
        RHUM: pd.Series = None,
        TCLW: pd.Series = None,
        TEMP: pd.Series = None,
        TMAX: pd.Series = None,
        TMIN: pd.Series = None,
        WIND: pd.Series = None,
    ):
        self.location = location
        self.emissions_scenario = emissions_scenario
        self.forecast_year = forecast_year
        self.DSWF = DSWF
        self.MSLP = MSLP
        self.PREC = PREC
        self.RHUM = RHUM
        self.TCLW = TCLW
        self.TEMP = TEMP
        self.TMAX = TMAX
        self.TMIN = TMIN
        self.WIND = WIND

    def __str__(self) -> str:
        return f"{self.location.city}-{self.emissions_scenario}-{self.forecast_year}"

    def __repr__(self) -> str:
        return f"{__class__.__name__}[{self}]"


class ForecastScenario:
    def __init__(
        self, 
        emissions_scenario: EmissionsScenario, 
        forecast_year: ForecastYear, 
        dataset_dir: Union[Path, str] = r"C:\ccwwg\datasets"
    ):

        self.emissions_scenario = emissions_scenario
        self.forecast_year = forecast_year

        self._root_directory = Path(dataset_dir)
        self._month_idx = pd.date_range("2021-01-01", freq="MS", periods=12)
        self._year_idx = pd.date_range("2021-01-01 00:30:00", freq="60T", periods=8760)

        def __setter(
            obj: object, var: str, dir: Path, es: EmissionsScenario, fy: ForecastYear
        ):
            setattr(obj, var, load_variable_dif(construct_file_path(dir, var, es, fy)))

        results = []
        print(f"Loading {self} datasets")
        with ThreadPoolExecutor() as executor:
            for var in [
                "DSWF",
                "MSLP",
                "PREC",
                "RHUM",
                "TCLW",
                "TEMP",
                "TMIN",
                "TMAX",
                "WIND",
            ]:
                results.append(
                    executor.submit(
                        __setter,
                        self,
                        var,
                        self._root_directory,
                        self.emissions_scenario,
                        self.forecast_year,
                    )
                )

        self._points = load_points_kml(self._root_directory / "HADCM3_grid_centre.kml")
        self._wind_points = load_points_kml(
            self._root_directory / "HADCM3_grid_WIND_centre.kml"
        )

    def get_translation_factors(self, location: Location) -> TranslationFactors:
        """
        Get the translation factors for a given location.
        """
        nearest_point_indices = nearest_n_point_indices(self._points, location, n=4)
        nearest_wind_point_indices = nearest_n_point_indices(
            self._wind_points, location, n=4
        )

        def __mp(fs, obj: TranslationFactors, var: str) -> None:
            vals = (
                pd.DataFrame(getattr(fs, var), index=fs._month_idx)
                .reindex(fs._year_idx, method="ffill")
                .iloc[
                    :,
                    nearest_wind_point_indices
                    if var == "WIND"
                    else nearest_point_indices,
                ]
                .mean(axis=1)
            )
            setattr(obj, var, vals)

        translations = TranslationFactors(
            location, self.emissions_scenario, self.forecast_year
        )
        results = []
        with ThreadPoolExecutor() as executor:
            for var in [
                "DSWF",
                "MSLP",
                "PREC",
                "RHUM",
                "TCLW",
                "TEMP",
                "TMIN",
                "TMAX",
                "WIND",
            ]:
                results.append(
                    executor.submit(
                        __mp,
                        self,
                        translations,
                        var,
                    )
                )

        return translations

    def __str__(self) -> str:
        return f"{self.emissions_scenario}-{self.forecast_year} forecast"

    def __repr__(self) -> str:
        return f"{__class__.__name__}[{self}]"


def check_vars(epw_obj: EPW, vars: List[str] = None) -> None:

    if vars is None:
        vars = [
            "dry_bulb_temperature",
            "relative_humidity",
            "dew_point_temperature",
            "opaque_sky_cover",
        ]

    for var in vars:
        try:
            col = getattr(epw_obj, var)
            print(f"{var} [{col.min}, {col.average}, {col.max}]")
        except:
            pass


def forecast_epw(
    epw: EPW, emissions_scenario: EmissionsScenario, forecast_year: ForecastYear
) -> EPW:
    """Forecast an EPW file using the CCWWG methodology."""

    warnings.warn(
        f"\n\nThis forecast method does not transform the following attributes of the EPW object:\n"
        "- Extraterrestrial Horizontal Radiation\n"
        "- Extraterrestrial Direct Normal Radiation\n"
        "- Wind Direction\n"
        "- Visibility\n"
        "- Ceiling Height\n"
        "- Present Weather Observation\n"
        "- Present Weather Codes\n"
        "- Aerosol Optical Depth\n"
        "- Snow Depth\n"
        "- Days Since Last Snowfall\n"
        "- Albedo\n\n"
        "Header information is modified where feasible, but heating and cooling design days are not yet modified so please use this file with caution when using this file in EnergyPlus for sizing load calculations.\n",
        stacklevel=2,
    )

    # load data
    forecast_scenario = ForecastScenario(emissions_scenario, forecast_year)

    # get forecast data
    translation_factors = forecast_scenario.get_translation_factors(epw.location)

    # create new epw object for modification
    new_epw = EPW.from_missing_values(epw.is_leap_year)
    new_epw.location = epw.location

    # YEAR
    new_epw.years.values = epw.years.values

    # WIND DIRECTION
    new_epw.wind_direction.values = epw.wind_direction.values

    # DRY BULB TEMPERATURE
    dbt_0 = to_series(epw.dry_bulb_temperature)
    idx_hourly = dbt_0.index
    dbt_0_monthly_average_daily_max = (
        dbt_0.resample("1D")
        .max()
        .resample("MS")
        .mean()
        .reindex(idx_hourly, method="ffill")
    )
    dbt_0_monthly_average_daily_mean = (
        dbt_0.resample("MS").mean().reindex(idx_hourly, method="ffill")
    )
    dbt_0_monthly_average_daily_min = (
        dbt_0.resample("1D")
        .min()
        .resample("MS")
        .mean()
        .reindex(idx_hourly, method="ffill")
    )
    adbt_m = (translation_factors.TMAX.values - translation_factors.TMIN.values) / (
        dbt_0_monthly_average_daily_max - dbt_0_monthly_average_daily_min
    )
    dbt_new = (
        dbt_0
        + translation_factors.TEMP.values
        + adbt_m * (dbt_0 - dbt_0_monthly_average_daily_mean)
    ).values
    dbt_new = np.where(dbt_0.values == 99.9, dbt_0.values, dbt_new)
    new_epw.dry_bulb_temperature.values = dbt_new

    # check_vars(new_epw, ["dry_bulb_temperature"])

    # RELATIVE HUMIDITY
    rh_0 = to_series(epw.relative_humidity)
    rh_new = (rh_0 + translation_factors.RHUM.values).clip(0, 110).values
    rh_new = np.where(rh_0.values == 999, rh_0.values, rh_new)
    new_epw.relative_humidity.values = rh_new

    # check_vars(new_epw, ["relative_humidity"])

    # DEW POINT TEMPERATURE
    dpt_new = HourlyContinuousCollection.compute_function_aligned(
        dew_point_from_db_rh,
        [
            new_epw.dry_bulb_temperature,
            new_epw.relative_humidity,
        ],
        epw.dew_point_temperature.header.data_type,
        "C",
    ).values
    dpt_new = np.where(
        epw.dew_point_temperature.values == 99.9,
        epw.dew_point_temperature.values,
        dpt_new,
    )
    new_epw.dew_point_temperature.values = dpt_new

    # check_vars(new_epw, ["dew_point_temperature"])

    # ATMOSPHERIC STATION PRESSURE
    asp_0 = to_series(epw.atmospheric_station_pressure)
    asp_new = (asp_0 + (translation_factors.MSLP.values * 100)).values
    asp_new = np.where(asp_0.values == 999999, asp_0.values, asp_new)
    new_epw.atmospheric_station_pressure.values = asp_new

    # TOTAL SKY COVER
    try:
        tsc_0 = to_series(epw.total_sky_cover)
        tsc_new = (tsc_0 + (translation_factors.TCLW.values / 10)).clip(0, 10).values
        tsc_new = np.where(tsc_0.values == 99, tsc_0.values, tsc_new)
        new_epw.total_sky_cover.values = tsc_new
    except ValueError as exc:
        warnings.warn(
            "Total sky cover not forecast.",
            stacklevel=2,
        )

    # OPAQUE SKY COVER
    try:
        osc_0 = to_series(epw.opaque_sky_cover)
        osc_new = (tsc_0 + (translation_factors.TCLW.values / 10)).clip(0, 10).values
        osc_new = np.where(osc_0.values == 99, osc_0.values, osc_new)
        new_epw.opaque_sky_cover.values = osc_new
    except ValueError as exc:
        warnings.warn(
            "Opaque sky cover not forecast.",
            stacklevel=2,
        )

    try:
        # LIQUID PRECIPITATION DEPTH
        lpd_0 = to_series(epw.liquid_precipitation_depth)
        lpd_new = ((1 + (translation_factors.PREC.values / 100)) * lpd_0).values
        lpd_new = np.where(lpd_0.values == 999, lpd_0.values, lpd_new)
        new_epw.liquid_precipitation_depth.values = lpd_new
    except ValueError as e:
        warnings.warn(
            "The input weatherfile does not contain any data for liquid_precipitation_depth, and therefore this variable cannot be forecast.",
            stacklevel=2,
        )

    try:
        # LIQUID PRECIPITATION QUANTITY
        lpq_0 = to_series(epw.liquid_precipitation_quantity)
        lpq_new = ((1 + (translation_factors.PREC.values / 100)) * lpq_0).values
        lpq_new = np.where(lpq_0.values == 99, lpq_0.values, lpq_new)
        new_epw.liquid_precipitation_quantity.values = lpq_new
    except ValueError as e:
        warnings.warn(
            "The input weatherfile does not contain any data for liquid_precipitation_quantity, and therefore this variable cannot be forecast.",
            stacklevel=2,
        )

    # PRECIPITABLE WATER
    pw_0 = to_series(epw.precipitable_water)
    pw_new = ((1 + (translation_factors.PREC.values / 100)) * pw_0).values
    pw_new = np.where(pw_0.values == 999, pw_0.values, pw_new)
    new_epw.precipitable_water.values = pw_new

    # WIND SPEED
    ws_0 = to_series(epw.wind_speed)
    ws_new = (1 + translation_factors.WIND.values / 100) * ws_0 * 0.514444
    ws_new = np.where(ws_0.values == 999, ws_0.values, ws_new)
    new_epw.wind_speed.values = ws_new

    # GLOBAL HORIZONTAL RADIATION
    ghr_0 = to_series(epw.global_horizontal_radiation)
    ghr_scale_factor = 1 + (translation_factors.DSWF.values / ghr_0)
    ghr_scale_factor[ghr_scale_factor < 0] = 1
    ghr_new = (ghr_0 * ghr_scale_factor).clip(lower=0).values
    ghr_new = np.nan_to_num(np.where(ghr_0.values == 9999, ghr_0.values, ghr_new))
    new_epw.global_horizontal_radiation.values = ghr_new

    # DIRECT NORMAL RADIATION
    dnr_0 = to_series(new_epw.direct_normal_radiation)
    dnr_scale_factor = 1 + (translation_factors.DSWF.values / dnr_0)
    dnr_scale_factor[dnr_scale_factor < 0] = 1
    dnr_new = (dnr_0 * dnr_scale_factor).clip(lower=0).values
    dnr_new = np.nan_to_num(np.where(dnr_0.values == 9999, dnr_0.values, dnr_new))
    new_epw.direct_normal_radiation.values = dnr_new
    if (new_epw.direct_normal_radiation.average == 9999) & (
        new_epw.global_horizontal_radiation.average != 9999
    ):
        pass

    # DIFFUSE HORIZONTAL RADIATION
    dhr_0 = to_series(epw.diffuse_horizontal_radiation)
    dhr_scale_factor = 1 + (translation_factors.DSWF.values / dhr_0)
    dhr_scale_factor[dhr_scale_factor < 0] = 1
    dhr_new = (dhr_0 * dhr_scale_factor).clip(lower=0).values
    dhr_new = np.nan_to_num(np.where(dhr_0.values == 9999, dhr_0.values, dhr_new))
    new_epw.diffuse_horizontal_radiation.values = dhr_new
    if (new_epw.diffuse_horizontal_radiation.average == 9999) & (
        new_epw.global_horizontal_radiation.average != 9999
    ):
        pass

    # GLOBAL HORIZONTAL ILLUMINANCE
    ghi_0 = to_series(epw.global_horizontal_illuminance)
    ghi_scale_factor = 1 + (translation_factors.DSWF.values / ghi_0)
    ghi_scale_factor[ghi_scale_factor < 0] = 1
    ghi_new = (ghi_0 * ghi_scale_factor).clip(lower=0).values
    ghi_new = np.nan_to_num(np.where(ghi_0.values >= 999900, ghi_0.values, ghi_new))
    new_epw.global_horizontal_illuminance.values = ghi_new

    # DIRECT NORMAL ILLUMINANCE
    dni_0 = to_series(epw.direct_normal_illuminance)
    dni_scale_factor = 1 + (translation_factors.DSWF.values / dni_0)
    dni_scale_factor[dni_scale_factor < 0] = 1
    dni_new = (dni_0 * dni_scale_factor).clip(lower=0).values
    dni_new = np.nan_to_num(np.where(dni_0.values >= 999900, dni_0.values, dni_new))
    new_epw.direct_normal_illuminance.values = dni_new

    # DIFFUSE HORIZONTAL ILLUMINANCE
    dhi_0 = to_series(epw.diffuse_horizontal_illuminance)
    dhi_scale_factor = 1 + (translation_factors.DSWF.values / dhi_0)
    dhi_scale_factor[dhi_scale_factor < 0] = 1
    dhi_new = (dhi_0 * dhi_scale_factor).clip(lower=0).values
    dhi_new = np.nan_to_num(np.where(dhi_0.values >= 999900, dhi_0.values, dhi_new))
    new_epw.diffuse_horizontal_illuminance.values = dhi_new

    # HORIZONTAL INFRARED RADIATION
    # check_vars(new_epw, ["opaque_sky_cover"])
    try:
        hir_new = HourlyContinuousCollection.compute_function_aligned(
            calc_horizontal_infrared,
            [
                new_epw.opaque_sky_cover,
                new_epw.dry_bulb_temperature,
                new_epw.dew_point_temperature,
            ],
            new_epw.horizontal_infrared_radiation_intensity.header.data_type,
            new_epw.horizontal_infrared_radiation_intensity.header.unit,
        ).values

        hir_new = np.nan_to_num(
            np.where(
                epw.horizontal_infrared_radiation_intensity.values == 9999,
                epw.horizontal_infrared_radiation_intensity.values,
                hir_new,
            )
        )
        new_epw.horizontal_infrared_radiation_intensity.values = hir_new
    except ValueError as exc:
        warnings.warn(
            "horizontal_infrared_radiation_intensity not forecast.",
            stacklevel=2,
        )

    # ZENITH LUMINANCE
    solar_altitude = get_solar_altitude(epw)
    zl_new = [
        i[-1]
        for i in HourlyContinuousCollection.compute_function_aligned(
            estimate_illuminance_from_irradiance,
            [
                solar_altitude,
                new_epw.global_horizontal_illuminance,
                new_epw.direct_normal_illuminance,
                new_epw.diffuse_horizontal_illuminance,
                new_epw.dew_point_temperature,
            ],
            new_epw.zenith_luminance.header.data_type,
            new_epw.zenith_luminance.header.unit,
        )
    ]

    zl_new = np.nan_to_num(
        np.where(
            to_series(epw.zenith_luminance) >= 9999,
            to_series(epw.zenith_luminance).values,
            zl_new,
        )
    )
    new_epw.zenith_luminance.values = zl_new

    # GROUND TEMPERATURES
    factors = (
        to_series(new_epw.dry_bulb_temperature).resample("MS").mean()
        / to_series(epw.dry_bulb_temperature).resample("MS").mean()
    ).values
    new_ground_temperatures = {}
    for depth, collection in epw.monthly_ground_temperature.items():
        new_ground_temperatures[depth] = MonthlyCollection(
            header=collection.header,
            values=factors * collection.values,
            datetimes=collection.datetimes,
        )
    new_epw._monthly_ground_temps = new_ground_temperatures

    # Fix the radiation values if they are all 9999
    dir_norm_rad, dif_horiz_rad = zhang_huang_solar_split(
        solar_altitude.values,
        dbt_0.index.dayofyear,
        epw.total_sky_cover.values,
        new_epw.relative_humidity.values,
        new_epw.dry_bulb_temperature.values,
        np.roll(new_epw.dry_bulb_temperature.values, -3),
        new_epw.wind_speed.values,
        new_epw.atmospheric_station_pressure.values,
        use_disc=False,
    )
    if new_epw.direct_normal_radiation.average == 9999:
        new_epw.direct_normal_radiation.values = dir_norm_rad
    if new_epw.diffuse_horizontal_radiation.average == 9999:
        new_epw.diffuse_horizontal_radiation.values = dif_horiz_rad

    # Modify the EPW to state that it is a forecast file
    new_epw.comments_1 = f"{epw.comments_1}. Forecast using transformation factors from HadCM3 {emissions_scenario.value} emissions scenario for {forecast_year.value}."
    new_epw.comments_2 = epw.comments_2

    new_epw = fix_dodgy_month(new_epw)

    return new_epw


def fix_dodgy_month(epw: EPW) -> EPW:

    epw_str = epw.to_file_string()
    temp_df = pd.DataFrame([i.split(",") for i in epw_str.split("\n")[8:-1]])

    if (
        (temp_df.iloc[743][1] == "2")
        & (temp_df.iloc[743][2] == "31")
        & (temp_df.iloc[743][3] == "24")
    ):
        print("Theres a problem - trying to fix it")
        temp_df[1] = np.roll(temp_df[1].values, 1)
    else:
        print("no problem - returning the original epw")
        return epw

    fixed_epw_str = (
        "\n".join(epw_str.split("\n")[:8])
        + "\n"
        + "\n".join([",".join(i) for i in temp_df.values])
        + "\n"
    )
    return EPW.from_file_string(fixed_epw_str)


if __name__ == "__main__":

    INPUT_EPW = "<PATH_TO_EPW>"
    OUTPUT_DIR = "<PATH_TO_OUTPUT_DIR>"

    epws = [
        Path(
            INPUT_EPW
        ),
    ]
    ess = ["A2a", "A2b", "A2c"]
    fys = [2020, 2050, 2080]

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True, parents=True)

    n = 0
    nn = len(epws) * len(ess) * len(fys)
    for epw_file in epws:
        for es in ess[:]:
            for fy in fys[:]:
                print(f"\n## [{n:02d}/{nn:02d} {n/nn:0.2%}] {epw_file.stem} ##")

                epw = EPW(epw_file)
                emissionsScenario = EmissionsScenario.from_str(es)
                forecastYear = ForecastYear.from_int(fy)

                new_epw = forecast_epw(epw, emissionsScenario, forecastYear)
                new_epw_fixed = fix_dodgy_month(new_epw)
                sp = (
                    out_dir
                    / Path(epw_file)
                    .with_suffix(f".{emissionsScenario}_{forecastYear}.epw")
                    .name
                ).as_posix()
                print(f"Saving to {sp}")
                new_epw_fixed.save(sp)
                n += 1
