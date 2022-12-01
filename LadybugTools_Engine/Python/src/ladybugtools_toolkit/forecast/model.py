from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Union

import fortranformat as ff
import numpy as np
import pandas as pd
from ladybug.datatype.temperature import DewPointTemperature
from ladybug.epw import (
    EPW,
    AnalysisPeriod,
    HourlyContinuousCollection,
    Location,
    MonthlyCollection,
)
from ladybug.header import Header
from ladybug.psychrometrics import dew_point_from_db_rh
from ladybug.skymodel import (
    calc_horizontal_infrared,
    estimate_illuminance_from_irradiance,
    zhang_huang_solar_split,
)
from ladybugtools_toolkit.ladybug_extension.epw import solar_altitude
from scipy import spatial
from tqdm import tqdm

from ..ladybug_extension.analysis_period import to_datetimes
from ..ladybug_extension.datacollection import to_series


@dataclass(init=True, repr=True, eq=True)
class ForecastModel:
    model_type: str = field(init=False, repr=True, compare=True)


@dataclass(init=True, repr=True, eq=True)
class HadCM3(ForecastModel):
    forecast_year: int = field(init=True, repr=True)
    emissions_scenario: str = field(init=True, repr=True)
    model_type: str = field(init=False, repr=True, default="IPCC HadCM3 A2")

    def __post_init__(self):
        forecast_year_options = [2020, 2050, 2080]
        emissions_scenario_options = ["A2a", "A2b", "A2c"]

        self.dataset_directory: Path = Path("C:/ccwwg/datasets")

        if self.forecast_year not in forecast_year_options:
            raise ValueError(
                f"{self.forecast_year} is not possible for this forecast model. Please use one of {forecast_year_options}."
            )

        if self.emissions_scenario not in emissions_scenario_options:
            raise ValueError(
                f"{self.emissions_scenario} is not possible for this forecast model. Please use one of {emissions_scenario_options}."
            )

        if not self.dataset_directory.exists():
            raise ValueError(
                f"{self._dataset_directory} cannot be found, and therefore no data can be loaded for this forecast model."
            )

        if not len(list(self.dataset_directory.glob("*.*"))) == 83:
            raise ValueError(
                f"The number of files present in {self.dataset_directory} () suggest that the forecast datasets are not available for this model."
            )

        self._points = self._load_points_kml(
            self.dataset_directory / "HadCM3_grid_centre.kml"
        )
        self._wind_points = self._load_points_kml(
            self.dataset_directory / "HadCM3_grid_WIND_centre.kml"
        )
        self._year_idx = to_datetimes(AnalysisPeriod())

    @staticmethod
    def _load_points_kml(kml_path: Path) -> List[List[float]]:
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

    @staticmethod
    def _load_variable_dif(file_path: Path) -> List[List[float]]:
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

    def _construct_file_path(
        self,
        climate_variable: str,
    ) -> Path:
        file_path = (
            self.dataset_directory
            / f"HADCM3_{self.emissions_scenario}_{climate_variable}_{self.forecast_year}.dif"
        )
        if file_path.exists():
            return file_path
        else:
            raise FileNotFoundError(
                f"It doesn't seem as though a dataset is available for {file_path.name}."
            )

    @staticmethod
    def _nearest_n_point_indices(
        points: List[List[float]], location: Location, n: int
    ) -> Union[List[float], List[int]]:
        distance, nearest_point_indices = spatial.KDTree(points).query(
            [location.latitude, location.longitude], k=n
        )
        return distance, nearest_point_indices

    def _get_temporal_factor(self, location: Location, variable: str) -> pd.Series:

        if variable not in [
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
            raise ValueError(
                f"{variable} is not a morphing variable that can be loaded for {__class__.__name__}."
            )

        # load values from file
        vals = pd.DataFrame(
            self._load_variable_dif(self._construct_file_path(variable))
        )

        # get nearest n-points to qudrangulate betwen them
        if variable == "WIND":
            distances, point_indices = self._nearest_n_point_indices(
                self._wind_points, location, n=4
            )
        else:
            distances, point_indices = self._nearest_n_point_indices(
                self._points, location, n=4
            )

        # convert distances to "weights", with highest weighting for closest locations
        weights = 1 - distances / sum(distances)

        return pd.Series(
            np.average(vals.iloc[:, point_indices], axis=1, weights=weights),
            index=pd.Series(index=self._year_idx, dtype=object)
            .resample("MS")
            .mean()
            .index,
        ).reindex(self._year_idx, method="ffill")

    def forecast(self, epw: EPW) -> EPW:

        # load datasets
        d = {}
        for var in tqdm(
            [
                "DSWF",
                "MSLP",
                "PREC",
                "RHUM",
                "TCLW",
                "TEMP",
                "TMIN",
                "TMAX",
                "WIND",
            ],
            desc="Loading forecast model datasets",
        ):
            d[var] = self._get_temporal_factor(epw.location, var)

        # create an "empty" epw object eready to populate
        new_epw = EPW.from_missing_values(epw.is_leap_year)
        new_epw.location = epw.location
        new_epw.comments_1 = f"{epw.comments_1}. Forecast using transformation factors from the IPCC HadCM3 {self.emissions_scenario} emissions scenario for {self.forecast_year}."
        new_epw.comments_2 = epw.comments_2

        # copy over variables that aren't going to change
        new_epw.years.values = epw.years.values
        new_epw.wind_direction.values = epw.wind_direction.values

        # attempt to morph each variable in the input EPW
        new_epw.dry_bulb_temperature.values = self._morph_dbt(
            to_series(epw.dry_bulb_temperature), d["TMIN"], d["TEMP"], d["TMAX"]
        ).values

        new_epw.relative_humidity.values = self._morph_rh(
            to_series(epw.relative_humidity), d["RHUM"]
        ).values

        new_epw.dew_point_temperature.values = self._get_dpt(new_epw).values

        new_epw.atmospheric_station_pressure.values = self._morph_atm(
            to_series(epw.atmospheric_station_pressure), d["MSLP"]
        ).values

        new_epw.extraterrestrial_horizontal_radiation.values = (
            epw.extraterrestrial_horizontal_radiation.values
        )
        new_epw.extraterrestrial_direct_normal_radiation.values = (
            epw.extraterrestrial_direct_normal_radiation.values
        )

        new_epw.direct_normal_radiation.values = self._morph_sol(
            to_series(epw.direct_normal_radiation),
            d["DSWF"],
            missing=new_epw.direct_normal_radiation.values[0],
        ).values
        new_epw.diffuse_horizontal_radiation.values = self._morph_sol(
            to_series(epw.diffuse_horizontal_radiation),
            d["DSWF"],
            missing=new_epw.diffuse_horizontal_radiation.values[0],
        ).values
        new_epw.global_horizontal_radiation.values = self._morph_sol(
            to_series(epw.global_horizontal_radiation),
            d["DSWF"],
            missing=new_epw.global_horizontal_radiation.values[0],
        ).values

        new_epw.total_sky_cover.values = self._morph_cc(
            to_series(epw.total_sky_cover), d["TCLW"]
        ).values
        new_epw.opaque_sky_cover.values = self._morph_cc(
            to_series(epw.opaque_sky_cover), d["TCLW"]
        ).values

        new_epw.horizontal_infrared_radiation_intensity.values = self._get_hir(
            new_epw
        ).values

        new_epw.direct_normal_illuminance.values = self._morph_sol(
            to_series(epw.direct_normal_illuminance),
            d["DSWF"],
            missing=new_epw.direct_normal_illuminance.values[0],
        ).values
        new_epw.diffuse_horizontal_illuminance.values = self._morph_sol(
            to_series(epw.diffuse_horizontal_illuminance),
            d["DSWF"],
            missing=new_epw.diffuse_horizontal_illuminance.values[0],
        ).values
        new_epw.global_horizontal_illuminance.values = self._morph_sol(
            to_series(epw.global_horizontal_illuminance),
            d["DSWF"],
            missing=new_epw.global_horizontal_illuminance.values[0],
        ).values

        new_epw.zenith_luminance.values = self._get_zl(new_epw).values

        new_epw.wind_speed.values = self._morph_ws(
            to_series(epw.wind_speed), d["WIND"]
        ).values

        new_epw.wind_direction.values = epw.wind_direction.values
        new_epw.visibility.values = epw.visibility.values
        new_epw.present_weather_observation.values = (
            epw.present_weather_observation.values
        )
        new_epw.present_weather_codes.values = epw.present_weather_codes.values

        new_epw.precipitable_water.values = self._morph_prec(
            to_series(epw.precipitable_water),
            d["PREC"],
            missing=new_epw.precipitable_water.values[0],
        ).values

        new_epw.aerosol_optical_depth.values = epw.aerosol_optical_depth.values
        new_epw.snow_depth.values = epw.snow_depth.values
        new_epw.days_since_last_snowfall.values = epw.days_since_last_snowfall.values
        new_epw.albedo.values = epw.albedo.values

        new_epw.liquid_precipitation_depth.values = self._morph_prec(
            to_series(epw.liquid_precipitation_depth),
            d["PREC"],
            missing=new_epw.liquid_precipitation_depth.values[0],
        ).values

        new_epw.liquid_precipitation_quantity.values = self._morph_prec(
            to_series(epw.liquid_precipitation_quantity),
            d["PREC"],
            missing=new_epw.liquid_precipitation_quantity.values[0],
        ).values

        new_epw._monthly_ground_temps = self._get_gnd_temp(epw, new_epw)

        return new_epw

    @staticmethod
    def _morph_dbt(
        series: pd.Series, tmin: pd.Series, temp: pd.Series, tmax: pd.Series
    ) -> pd.Series:
        if len(series) != len(tmin) != len(temp) != len(tmax):
            raise ValueError(
                f"The shape of the DBT series and morph factors do not match ({len(series)} != {len(tmin)} != {len(temp)} != {len(tmax)})."
            )

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(
                "The index for the series to morph is not datetime-indexed."
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
        adbt_m = (tmax.values - tmin.values) / (
            dbt_0_monthly_average_daily_max - dbt_0_monthly_average_daily_min
        )
        dbt_new = (
            series + temp.values + adbt_m * (series - dbt_0_monthly_average_daily_mean)
        ).values
        dbt_new = np.where(series.values == 99.9, series.values, dbt_new)
        return pd.Series(dbt_new, index=series.index, name=series.name)

    @staticmethod
    def _morph_rh(series: pd.Series, rhum: pd.Series) -> pd.Series:
        rh_new = (series + rhum.values).clip(0, 110).values
        rh_new = np.where(series.values == 999, series.values, rh_new)
        return pd.Series(rh_new, index=series.index, name=series.name)

    @staticmethod
    def _morph_atm(series: pd.Series, mslp: pd.Series) -> pd.Series:
        asp_new = (series + mslp.values * 100).values
        asp_new = np.where(series.values == 999999, series.values, asp_new)
        return pd.Series(asp_new, index=series.index, name=series.name)

    @staticmethod
    def _morph_cc(series: pd.Series, ccov: pd.Series) -> pd.Series:
        osc_new = (series + (ccov.values / 10)).clip(0, 10).values
        osc_new = np.where(series.values == 99, series.values, osc_new)
        return pd.Series(osc_new, index=series.index, name=series.name)

    @staticmethod
    def _morph_prec(series: pd.Series, prec: pd.Series, missing: int) -> pd.Series:
        prec_new = (1 + (prec.values / 100)) * series.values
        prec_new = np.where(series.values == missing, series.values, prec_new)
        return pd.Series(prec_new, index=series.index, name=series.name)

    @staticmethod
    def _morph_ws(series: pd.Series, wind: pd.Series) -> pd.Series:
        ws_new = (1 + wind.values / 100) * series * 0.514444
        ws_new = np.where(series.values == 999, series.values, ws_new)
        return pd.Series(ws_new, index=series.index, name=series.name)

    @staticmethod
    def _morph_sol(series: pd.Series, dswf: pd.Series, missing: int) -> pd.Series:
        scale_factor = 1 + (dswf.values / series)
        scale_factor[scale_factor < 0] = 1
        sol_new = (series * scale_factor).clip(lower=0).values
        sol_new = np.nan_to_num(
            np.where(series.values == missing, series.values, sol_new)
        )
        return pd.Series(sol_new, index=series.index, name=series.name)

    @staticmethod
    def _get_dpt(epw: EPW) -> pd.Series:
        _dbt = to_series(epw.dry_bulb_temperature)
        _rh = to_series(epw.relative_humidity)
        dpt = []
        for dbt, rh in list(zip(*[_dbt, _rh])):
            dpt.append(dew_point_from_db_rh(dbt, rh))
        return pd.Series(
            np.where(
                epw.dew_point_temperature.values == 99.9,
                epw.dew_point_temperature.values,
                dpt,
            ),
            index=_dbt.index,
        )

    @staticmethod
    def _get_hir(epw: EPW) -> pd.Series:
        _osc = to_series(epw.opaque_sky_cover)
        _dbt = to_series(epw.dry_bulb_temperature)
        _dpt = to_series(epw.dew_point_temperature)
        hir = []
        for osc, dbt, dpt in list(zip(*[_osc, _dbt, _dpt])):
            hir.append(calc_horizontal_infrared(osc, dbt, dpt))
        return pd.Series(
            data=hir,
            index=_dbt.index,
        )

    @staticmethod
    def _get_zl(epw: EPW) -> pd.Series:
        s_alts = solar_altitude(epw)
        _ghi = to_series(epw.global_horizontal_illuminance)
        _dni = to_series(epw.direct_normal_illuminance)
        _dhi = to_series(epw.diffuse_horizontal_illuminance)
        _dpt = to_series(epw.dew_point_temperature)
        zl = []
        for s_alt, ghi, dni, dhi, dpt in list(zip(*[s_alts, _ghi, _dni, _dhi, _dpt])):
            zl.append(
                estimate_illuminance_from_irradiance(s_alt, ghi, dni, dhi, dpt)[-1]
            )
        return pd.Series(
            data=zl,
            index=_dpt.index,
        )

    @staticmethod
    def _get_gnd_temp(original_epw: EPW, new_epw: EPW) -> Dict[str, MonthlyCollection]:
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
