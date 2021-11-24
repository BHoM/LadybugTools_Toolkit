from __future__ import annotations

import copy
import json
import warnings
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype.angle import Angle
from ladybug.datatype.fraction import Fraction, HumidityRatio
from ladybug.datatype.generic import GenericType
from ladybug.datatype.specificenergy import Enthalpy
from ladybug.datatype.temperature import SkyTemperature, WetBulbTemperature
from ladybug.datatype.time import Time
from ladybug.epw import EPW
from ladybug.header import Header
from ladybug.psychrometrics import (
    dew_point_from_db_rh,
    enthalpy_from_db_hr,
    humid_ratio_from_db_rh,
    wet_bulb_from_db_rh,
)
from ladybug.skymodel import (
    calc_horizontal_infrared,
    calc_sky_temperature,
    clearness_index,
    estimate_illuminance_from_irradiance,
)
from ladybug.sunpath import Sunpath

from datacollection import BH_HourlyContinuousCollection, BH_MonthlyCollection
from enums import EmissionsScenario, ForecastYear
from forecast_scenario import ForecastScenario


class BH_EPW(EPW):
    def __init__(self, file_path):
        super().__init__(file_path)
        __slots__ = super().__slots__

    def __repr__(self):
        """EPW representation."""
        return "EPW file Data for [%s]" % self.location.city

    def to_dataframe(self, include_location: bool = False) -> pd.DataFrame:
        """Create a Pandas DataFrame from the EPW object.

        Args:
            include_location (bool, optional): Include the EPW location as an additional column index level. Defaults to False.

        Returns:
            pd.DataFrame: A Pandas DataFrame.
        """

        all_series = []
        for p in dir(self):
            try:
                all_series.append(getattr(self, p).series)
            except (AttributeError, TypeError) as e:
                pass

        df = pd.concat(all_series, axis=1)

        if not include_location:
            df.columns = df.columns.droplevel(0)

        return df

    def to_csv(self, file_path: str) -> str:
        """Save the EPW contents (plus solar position and psychrometric values) to a CSV file.

        Args:
            file_path (str): The CSV file into which the EPW will be written.

        Returns:
            str: The path to teh resultant CSV file.
        """
        self.to_dataframe().to_csv(file_path)
        return file_path

    def to_json(self) -> str:
        """Convert an EPW into a JSON string representation version, according to the Ladybug EPW schema.

        Returns:
            str: A JSON string, with "Infinity" values replaced with 0's.
        """
        return json.dumps(self.to_dict()).replace("Infinity", "0")

    def forecast(
        self,
        emissions_scenario: EmissionsScenario,
        forecast_year: ForecastYear,
        save: bool = False,
    ) -> BH_EPW:

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
            "The header information is modified where feasible, but heating and cooling design days are not yet modified so please use this file with caution when using this file in EnergyPlus for sizing load calculations.\n",
            stacklevel=2,
        )

        # load data
        forecast_scenario = ForecastScenario(emissions_scenario, forecast_year)

        # get forecast data
        translation_factors = forecast_scenario.get_translation_factors(self.location)

        #print(f"Forecasting climate data for {self}")

        # copy the input epw for modification
        new_epw = copy.deepcopy(self)

        # DRY BULB TEMPERATURE
        dbt_0 = new_epw.dry_bulb_temperature.series
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
        adbt_m = (translation_factors.TMAX - translation_factors.TMIN) / (
            dbt_0_monthly_average_daily_max - dbt_0_monthly_average_daily_min
        )
        dbt_new = (
            dbt_0
            + translation_factors.TEMP
            + adbt_m * (dbt_0 - dbt_0_monthly_average_daily_mean)
        ).values
        dbt_new = np.where(
            dbt_0.values == 99.9, dbt_0.values, dbt_new
        )  # missing value handler
        new_epw._data[6] = BH_HourlyContinuousCollection(
            new_epw.dry_bulb_temperature.header, dbt_new
        )

        # RELATIVE HUMIDITY
        rh_0 = new_epw.relative_humidity.series
        rh_new = (rh_0 + translation_factors.RHUM).clip(0, 110).values
        rh_new = np.where(
            rh_0.values == 999, rh_0.values, rh_new
        )  # missing value handler
        new_epw._data[8] = BH_HourlyContinuousCollection(
            new_epw.relative_humidity.header, rh_new
        )

        # DEW POINT TEMPERATURE
        dpt_new = BH_HourlyContinuousCollection.compute_function_aligned(
            dew_point_from_db_rh,
            [
                new_epw.dry_bulb_temperature,
                new_epw.relative_humidity,
            ],
            self.dew_point_temperature.header.data_type,
            self.dew_point_temperature.header.unit,
        ).values
        dpt_new = np.where(
            self.dew_point_temperature.values == 99.9,
            self.dew_point_temperature.values,
            dpt_new,
        )  # missing value handler
        new_epw._data[7] = BH_HourlyContinuousCollection(
            self.dew_point_temperature.header, dpt_new
        )

        # ATMOSPHERIC STATION PRESSURE
        asp_0 = new_epw.atmospheric_station_pressure.series
        asp_new = (asp_0 + (translation_factors.MSLP * 100)).values
        asp_new = np.where(
            asp_0.values == 999999, asp_0.values, asp_new
        )  # missing value handler
        new_epw._data[9] = BH_HourlyContinuousCollection(
            new_epw.atmospheric_station_pressure.header, asp_new
        )

        # TOTAL SKY COVER
        tsc_0 = new_epw.total_sky_cover.series
        tsc_new = (tsc_0 + (translation_factors.TCLW / 10)).clip(0, 10).values
        tsc_new = np.where(
            tsc_0.values == 99, tsc_0.values, tsc_new
        )  # missing value handler
        new_epw._data[22] = BH_HourlyContinuousCollection(
            new_epw.total_sky_cover.header, tsc_new
        )

        # OPAQUE SKY COVER
        osc_0 = new_epw.opaque_sky_cover.series
        osc_new = (tsc_0 + (translation_factors.TCLW / 10)).clip(0, 10).values
        osc_new = np.where(
            osc_0.values == 99, osc_0.values, osc_new
        )  # missing value handler
        new_epw._data[23] = BH_HourlyContinuousCollection(
            new_epw.opaque_sky_cover.header, osc_new
        )

        try:
            # LIQUID PRECIPITATION DEPTH
            lpd_0 = new_epw.liquid_precipitation_depth.series
            lpd_new = ((1 + (translation_factors.PREC / 100)) * lpd_0).values
            lpd_new = np.where(
                lpd_0.values == 999, lpd_0.values, lpd_new
            )  # missing value handler
            new_epw._data[33] = BH_HourlyContinuousCollection(
                new_epw.liquid_precipitation_depth.header, lpd_new
            )
        except ValueError as e:
            warnings.warn("The input weatherfile does not contain any data for liquid_precipitation_depth, and therefore this variable cannot be forecasted.", stacklevel=2)

        try:
            # LIQUID PRECIPITATION QUANTITY
            lpq_0 = new_epw.liquid_precipitation_quantity.series
            lpq_new = ((1 + (translation_factors.PREC / 100)) * lpq_0).values
            lpq_new = np.where(
                lpq_0.values == 99, lpq_0.values, lpq_new
            )  # missing value handler
            new_epw._data[34] = BH_HourlyContinuousCollection(
                new_epw.liquid_precipitation_quantity.header, lpq_new
            )
        except ValueError as e:
            warnings.warn("The input weatherfile does not contain any data for liquid_precipitation_quantity, and therefore this variable cannot be forecasted.", stacklevel=2)

        # PRECIPITABLE WATER
        pw_0 = new_epw.precipitable_water.series
        pw_new = ((1 + (translation_factors.PREC / 100)) * pw_0).values
        pw_new = np.where(
            pw_0.values == 999, pw_0.values, pw_new
        )  # missing value handler
        new_epw._data[28] = BH_HourlyContinuousCollection(
            new_epw.precipitable_water.header, pw_new
        )

        # WIND SPEED
        ws_0 = new_epw.wind_speed.series
        ws_new = (1 + translation_factors.WIND / 100) * ws_0 * 0.514444
        ws_new = np.where(
            ws_0.values == 999, ws_0.values, ws_new
        )  # missing value handler
        new_epw._data[21] = BH_HourlyContinuousCollection(
            new_epw.wind_speed.header, ws_new
        )

        # GLOBAL HORIZONTAL RADIATION
        ghr_0 = new_epw.global_horizontal_radiation.series
        ghr_scale_factor = 1 + (translation_factors.DSWF / ghr_0)
        ghr_scale_factor[ghr_scale_factor < 0] = 1
        ghr_new = (ghr_0 * ghr_scale_factor).clip(lower=0).values
        ghr_new = np.nan_to_num(
            np.where(ghr_0.values == 9999, ghr_0.values, ghr_new)
        )  # missing value handler
        new_epw._data[13] = BH_HourlyContinuousCollection(
            new_epw.global_horizontal_radiation.header, ghr_new
        )

        # DIRECT NORMAL RADIATION
        dnr_0 = new_epw.direct_normal_radiation.series
        dnr_scale_factor = 1 + (translation_factors.DSWF / dnr_0)
        dnr_scale_factor[dnr_scale_factor < 0] = 1
        dnr_new = (dnr_0 * dnr_scale_factor).clip(lower=0).values
        dnr_new = np.nan_to_num(
            np.where(dnr_0.values == 9999, dnr_0.values, dnr_new)
        )  # missing value handler
        new_epw._data[14] = BH_HourlyContinuousCollection(
            new_epw.direct_normal_radiation.header, dnr_new
        )

        # DIFFUSE HORIZONTAL RADIATION
        dhr_0 = new_epw.diffuse_horizontal_radiation.series
        dhr_scale_factor = 1 + (translation_factors.DSWF / dhr_0)
        dhr_scale_factor[dhr_scale_factor < 0] = 1
        dhr_new = (dhr_0 * dhr_scale_factor).clip(lower=0).values
        dhr_new = np.nan_to_num(
            np.where(dhr_0.values == 9999, dhr_0.values, dhr_new)
        )  # missing value handler
        new_epw._data[15] = BH_HourlyContinuousCollection(
            new_epw.diffuse_horizontal_radiation.header, dhr_new
        )

        # GLOBAL HORIZONTAL ILLUMINANCE
        ghi_0 = new_epw.global_horizontal_illuminance.series
        ghi_scale_factor = 1 + (translation_factors.DSWF / ghi_0)
        ghi_scale_factor[ghi_scale_factor < 0] = 1
        ghi_new = (ghi_0 * ghi_scale_factor).clip(lower=0).values
        ghi_new = np.nan_to_num(
            np.where(ghi_0.values >= 999900, ghi_0.values, ghi_new)
        )  # missing value handler
        new_epw._data[16] = BH_HourlyContinuousCollection(
            new_epw.global_horizontal_illuminance.header, ghi_new
        )

        # DIRECT NORMAL ILLUMINANCE
        dni_0 = new_epw.direct_normal_illuminance.series
        dni_scale_factor = 1 + (translation_factors.DSWF / dni_0)
        dni_scale_factor[dni_scale_factor < 0] = 1
        dni_new = (dni_0 * dni_scale_factor).clip(lower=0).values
        dni_new = np.nan_to_num(
            np.where(dni_0.values >= 999900, dni_0.values, dni_new)
        )  # missing value handler
        new_epw._data[17] = BH_HourlyContinuousCollection(
            new_epw.direct_normal_illuminance.header, dni_new
        )

        # DIFFUSE HORIZONTAL ILLUMINANCE
        dhi_0 = new_epw.diffuse_horizontal_illuminance.series
        dhi_scale_factor = 1 + (translation_factors.DSWF / dhi_0)
        dhi_scale_factor[dhi_scale_factor < 0] = 1
        dhi_new = (dhi_0 * dhi_scale_factor).clip(lower=0).values
        dhi_new = np.nan_to_num(
            np.where(dhi_0.values >= 999900, dhi_0.values, dhi_new)
        )  # missing value handler
        new_epw._data[18] = BH_HourlyContinuousCollection(
            new_epw.diffuse_horizontal_illuminance.header, dhi_new
        )

        # HORIZONTAL INFRARED RADIATION
        hir_new = np.array(
            BH_HourlyContinuousCollection.compute_function_aligned(
                calc_horizontal_infrared,
                [
                    new_epw.opaque_sky_cover,
                    new_epw.dry_bulb_temperature,
                    new_epw.dew_point_temperature,
                ],
                self.horizontal_infrared_radiation_intensity.header.data_type,
                self.horizontal_infrared_radiation_intensity.header.unit,
            ).values
        )
        hir_new = np.nan_to_num(
            np.where(
                self.horizontal_infrared_radiation_intensity.values == 9999,
                self.horizontal_infrared_radiation_intensity.values,
                hir_new,
            )
        )  # missing value handler
        new_epw._data[12] = BH_HourlyContinuousCollection(
            self.horizontal_infrared_radiation_intensity.header, hir_new
        )

        # ZENITH LUMINANCE
        solar_altitude = self.solar_altitude
        zl_new = np.array(
            [
                i[-1]
                for i in BH_HourlyContinuousCollection.compute_function_aligned(
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
        )

        zl_new = np.nan_to_num(
            np.where(
                self.zenith_luminance.series >= 9999,
                self.zenith_luminance.series.values,
                zl_new,
            )
        )  # missing value handler
        new_epw._data[19] = BH_HourlyContinuousCollection(
            self.zenith_luminance.header, zl_new
        )

        # GROUND TEMPERATURES
        factors = (
            new_epw.dry_bulb_temperature.series.resample("MS").mean()
            / self.dry_bulb_temperature.series.resample("MS").mean()
        ).values
        new_ground_temperatures = {}
        for depth, collection in new_epw.monthly_ground_temperature.items():
            new_ground_temperatures[depth] = BH_MonthlyCollection(
                header=collection.header,
                values=factors * collection.values,
                datetimes=collection.datetimes,
            )
        new_epw._monthly_ground_temps = new_ground_temperatures

        # Modify the EPW to state that it is a forecast file
        new_epw.location.city = f"{translation_factors}"
        new_epw.comments_1 = f"{new_epw.comments_1}. Forecast using transformation factors from HadCM3 {emissions_scenario.value} emissions scenario for {forecast_year.value}."

        if save:
            transformed_file = (
                Path(self.file_path).parent
                / f"{Path(self.file_path).stem}_{emissions_scenario.value}_{forecast_year.value}.epw"
            )
            new_epw._file_path = transformed_file.as_posix()
            with open(transformed_file, "w") as fp:
                fp.write(new_epw.to_file_string())

        return new_epw

    @property
    def sun_position(self) -> BH_HourlyContinuousCollection:
        """Calculate a set of Sun positions for each hour of the year

        Returns:
            BH_HourlyContinuousCollection: A data collection containing sun positions for each hour of the year.
        """
        sunpath = Sunpath.from_location(self.location)

        return BH_HourlyContinuousCollection(
            Header(
                data_type=GenericType(name="Sun Position", unit="sun_position"),
                unit="sun_position",
                analysis_period=AnalysisPeriod(),
                metadata=self.dry_bulb_temperature.header.metadata,
            ),
            [sunpath.calculate_sun_from_hoy(i) for i in range(8760)],
        )

    @property
    def datetime(self) -> BH_HourlyContinuousCollection:
        """Get a list of datetimes for each hour of the year.

        Returns:
            BH_HourlyContinuousCollection: A collection of datetimes for each hour of the year.
        """
        n_hours = 8784 if self.is_leap_year else 8760
        year = 2020 if self.is_leap_year else 2021
        return BH_HourlyContinuousCollection(
            Header(
                data_type=GenericType(name="Datetime", unit="datetime"),
                unit="datetime",
                analysis_period=AnalysisPeriod(),
                metadata=self.dry_bulb_temperature.header.metadata,
            ),
            list(pd.date_range(f"{year}-01-01 00:30:00", freq="60T", periods=n_hours)),
        )

    @property
    def solar_time(self) -> BH_HourlyContinuousCollection:
        """Calculate solar time for each hour of the year.

        Returns:
            BH_HourlyContinuousCollection: A collection of solar time datetimes for each hour of the year.
        """

        st = []
        for dt in self.datetime:
            gamma = (
                2
                * np.pi
                / 365
                * (dt.timetuple().tm_yday - 1 + float(dt.hour - 12) / 24)
            )
            equation_of_time = 229.18 * (
                0.000075
                + 0.001868 * np.cos(gamma)
                - 0.032077 * np.sin(gamma)
                - 0.014615 * np.cos(2 * gamma)
                - 0.040849 * np.sin(2 * gamma)
            )
            time_offset = equation_of_time + 4 * self.location.longitude
            tst = dt.hour * 60 + dt.minute + dt.second / 60 + time_offset
            st.append(datetime.combine(dt.date(), time(0)) + timedelta(minutes=tst))

        return BH_HourlyContinuousCollection(
            Header(
                data_type=GenericType(name="Solar Time", unit="datetime"),
                unit="datetime",
                analysis_period=AnalysisPeriod(),
                metadata=self.dry_bulb_temperature.header.metadata,
            ),
            st,
        )

    @property
    def solar_time_in_hours(self) -> BH_HourlyContinuousCollection:
        """Calculate solar time for each hour of the year

        Returns:
            BH_HourlyContinuousCollection: A collection of solar time hours for each hour of the year.
        """
        solar_time = self.solar_time

        return BH_HourlyContinuousCollection(
            Header(
                data_type=Time(
                    name="Solar Time",
                ),
                unit="hr",
                analysis_period=AnalysisPeriod(),
                metadata=solar_time.header.metadata,
            ),
            [dt.hour + (dt.minute / 60) for dt in solar_time.values],
        )

    @property
    def solar_azimuth(self) -> BH_HourlyContinuousCollection:
        """Calculate annual hourly solar azimuth angles.

        Returns:
            BH_HourlyContinuousCollection: Annual hourly solar azimuth positions.
        """
        return BH_HourlyContinuousCollection(
            Header(
                data_type=Angle(
                    name="Solar Azimuth",
                ),
                unit="degrees",
                analysis_period=AnalysisPeriod(),
                metadata=self.dry_bulb_temperature.header.metadata,
            ),
            [i.azimuth_in_radians for i in self.sun_position.values],
        )

    @property
    def solar_azimuth_in_radians(self) -> BH_HourlyContinuousCollection:
        """Calculate annual hourly solar azimuth angles.

        Returns:
            BH_HourlyContinuousCollection: Annual hourly solar azimuth positions.
        """
        return BH_HourlyContinuousCollection(
            Header(
                data_type=Angle(
                    name="Solar Azimuth",
                ),
                unit="radians",
                analysis_period=AnalysisPeriod(),
                metadata=self.dry_bulb_temperature.header.metadata,
            ),
            [i.azimuth_in_radians for i in self.sun_position.values],
        )

    @property
    def solar_altitude(self) -> BH_HourlyContinuousCollection:
        """Calculate annual hourly solar altitude angles.

        Returns:
            BH_HourlyContinuousCollection: Annual hourly solar altitude positions.
        """
        return BH_HourlyContinuousCollection(
            Header(
                data_type=Angle(
                    name="Solar Altitude",
                ),
                unit="degrees",
                analysis_period=AnalysisPeriod(),
                metadata=self.dry_bulb_temperature.header.metadata,
            ),
            [i.altitude for i in self.sun_position.values],
        )

    @property
    def solar_altitude_in_radians(self) -> BH_HourlyContinuousCollection:
        """Calculate annual hourly solar altitude angles.

        Returns:
            BH_HourlyContinuousCollection: Annual hourly solar altitude positions.
        """
        return BH_HourlyContinuousCollection(
            Header(
                data_type=Angle(
                    name="Solar Altitude",
                ),
                unit="radians",
                analysis_period=AnalysisPeriod(),
                metadata=self.dry_bulb_temperature.header.metadata,
            ),
            [i.altitude_in_radians for i in self.sun_position.values],
        )

    @property
    def apparent_solar_zenith(self) -> BH_HourlyContinuousCollection:
        """Calculate annual hourly apparent solar zenith angles.

        Returns:
            BH_HourlyContinuousCollection: Annual hourly apparent solar zenith angles.
        """
        return BH_HourlyContinuousCollection(
            Header(
                data_type=GenericType(name="Apparent Solar Zenith", unit="degrees"),
                unit="degrees",
                analysis_period=AnalysisPeriod(),
                metadata=self.dry_bulb_temperature.header.metadata,
            ),
            [np.pi / 2 - i for i in self.solar_altitude.values],
        )

    @property
    def wet_bulb_temperature(self) -> BH_HourlyContinuousCollection:
        """Calculate an annual hourly wet bulb temperature collection for a given EPW.

        Returns:
            BH_HourlyContinuousCollection: A Wet Bulb Temperature data collection.
        """
        return BH_HourlyContinuousCollection.compute_function_aligned(
            wet_bulb_from_db_rh,
            [
                self.dry_bulb_temperature,
                self.relative_humidity,
                self.atmospheric_station_pressure,
            ],
            WetBulbTemperature(),
            "C",
        )

    @property
    def humidity_ratio(self) -> BH_HourlyContinuousCollection:
        """Calculate an annual hourly humidity ratio collection for a given EPW.

        Returns:
            BH_HourlyContinuousCollection: A Humidity Ratio data collection.
        """
        return BH_HourlyContinuousCollection.compute_function_aligned(
            humid_ratio_from_db_rh,
            [
                self.dry_bulb_temperature,
                self.relative_humidity,
                self.atmospheric_station_pressure,
            ],
            HumidityRatio(),
            "fraction",
        )

    @property
    def enthalpy(self) -> BH_HourlyContinuousCollection:
        """Calculate an annual hourly enthalpy collection.

        Returns:
            BH_HourlyContinuousCollection: A Enthalpy data collection.
        """
        return BH_HourlyContinuousCollection.compute_function_aligned(
            enthalpy_from_db_hr,
            [
                self.dry_bulb_temperature,
                self.humidity_ratio,
            ],
            Enthalpy(),
            "kJ/kg",
        )

    @property
    def clearness_index(self) -> BH_HourlyContinuousCollection:
        """Return the clearness index value for each hour of the year.

        Returns:
            BH_HourlyContinuousCollection: A collection of clearness_index vlaues
        """
        return BH_HourlyContinuousCollection.compute_function_aligned(
            clearness_index,
            [
                self.global_horizontal_radiation,
                self.solar_altitude,
                self.extraterrestrial_direct_normal_radiation,
            ],
            Fraction(
                name="Clearness Index",
            ),
            "fraction",
        )

    @property
    def years(self) -> BH_HourlyContinuousCollection:
        """Return years as a Ladybug Data Collection."""
        _ = self._get_data_by_field(0)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def dry_bulb_temperature(self) -> BH_HourlyContinuousCollection:
        """Return annual Dry Bulb Temperature as a Ladybug Data Collection.

        This is the dry bulb temperature in C at the time indicated. Note that
        this is a full numeric field (i.e. 23.6) and not an integer representation
        with tenths. Valid values range from -70C to 70 C. Missing value for this
        field is 99.9.
        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(6)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def dew_point_temperature(self) -> BH_HourlyContinuousCollection:
        """Return annual Dew Point Temperature as a Ladybug Data Collection.

        This is the dew point temperature in C at the time indicated. Note that this is
        a full numeric field (i.e. 23.6) and not an integer representation with tenths.
        Valid values range from -70 C to 70 C. Missing value for this field is 99.9
        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(7)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def relative_humidity(self) -> BH_HourlyContinuousCollection:
        """Return annual Relative Humidity as a Ladybug Data Collection.

        This is the Relative Humidity in percent at the time indicated. Valid values
        range from 0% to 110%. Missing value for this field is 999.
        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(8)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def atmospheric_station_pressure(self) -> BH_HourlyContinuousCollection:
        """Return annual Atmospheric Station Pressure as a Ladybug Data Collection.

        This is the station pressure in Pa at the time indicated. Valid values range
        from 31,000 to 120,000. (These values were chosen from the standard barometric
        pressure for all elevations of the World). Missing value for this field is 999999
        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(9)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def extraterrestrial_horizontal_radiation(self) -> BH_HourlyContinuousCollection:
        """Return annual Extraterrestrial Horizontal Radiation as a Ladybug Data Collection.

        This is the Extraterrestrial Horizontal Radiation in Wh/m2. It is not currently
        used in EnergyPlus calculations. It should have a minimum value of 0; missing
        value for this field is 9999.
        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(10)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def extraterrestrial_direct_normal_radiation(self) -> BH_HourlyContinuousCollection:
        """Return annual Extraterrestrial Direct Normal Radiation as a Ladybug Data Collection.

        This is the Extraterrestrial Direct Normal Radiation in Wh/m2. (Amount of solar
        radiation in Wh/m2 received on a surface normal to the rays of the sun at the top
        of the atmosphere during the number of minutes preceding the time indicated).
        It is not currently used in EnergyPlus calculations. It should have a minimum
        value of 0; missing value for this field is 9999.
        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(11)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def horizontal_infrared_radiation_intensity(self) -> BH_HourlyContinuousCollection:
        """Return annual Horizontal Infrared Radiation Intensity as a Ladybug Data Collection.

        This is the Horizontal Infrared Radiation Intensity in W/m2. If it is missing,
        it is calculated from the Opaque Sky Cover field as shown in the following
        explanation. It should have a minimum value of 0; missing value for this field
        is 9999.
        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(12)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def global_horizontal_radiation(self) -> BH_HourlyContinuousCollection:
        """Return annual Global Horizontal Radiation as a Ladybug Data Collection.

        This is the Global Horizontal Radiation in Wh/m2. (Total amount of direct and
        diffuse solar radiation in Wh/m2 received on a horizontal surface during the
        number of minutes preceding the time indicated.) It is not currently used in
        EnergyPlus calculations. It should have a minimum value of 0; missing value
        for this field is 9999.
        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(13)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def direct_normal_radiation(self) -> BH_HourlyContinuousCollection:
        """Return annual Direct Normal Radiation as a Ladybug Data Collection.

        This is the Direct Normal Radiation in Wh/m2. (Amount of solar radiation in
        Wh/m2 received directly from the solar disk on a surface perpendicular to the
        sun's rays, during the number of minutes preceding the time indicated.) If the
        field is missing ( >= 9999) or invalid ( < 0), it is set to 0. Counts of such
        missing values are totaled and presented at the end of the runperiod.
        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(14)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def diffuse_horizontal_radiation(self) -> BH_HourlyContinuousCollection:
        """Return annual Diffuse Horizontal Radiation as a Ladybug Data Collection.

        This is the Diffuse Horizontal Radiation in Wh/m2. (Amount of solar radiation in
        Wh/m2 received from the sky (excluding the solar disk) on a horizontal surface
        during the number of minutes preceding the time indicated.) If the field is
        missing ( >= 9999) or invalid ( < 0), it is set to 0. Counts of such missing
        values are totaled and presented at the end of the runperiod
        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(15)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def global_horizontal_illuminance(self) -> BH_HourlyContinuousCollection:
        """Return annual Global Horizontal Illuminance as a Ladybug Data Collection.

        This is the Global Horizontal Illuminance in lux. (Average total amount of
        direct and diffuse illuminance in hundreds of lux received on a horizontal
        surface during the number of minutes preceding the time indicated.) It is not
        currently used in EnergyPlus calculations. It should have a minimum value of 0;
        missing value for this field is 999999 and will be considered missing if greater
        than or equal to 999900.
        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(16)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def direct_normal_illuminance(self) -> BH_HourlyContinuousCollection:
        """Return annual Direct Normal Illuminance as a Ladybug Data Collection.

        This is the Direct Normal Illuminance in lux. (Average amount of illuminance in
        hundreds of lux received directly from the solar disk on a surface perpendicular
        to the sun's rays, during the number of minutes preceding the time indicated.)
        It is not currently used in EnergyPlus calculations. It should have a minimum
        value of 0; missing value for this field is 999999 and will be considered missing
        if greater than or equal to 999900.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(17)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def diffuse_horizontal_illuminance(self) -> BH_HourlyContinuousCollection:
        """Return annual Diffuse Horizontal Illuminance as a Ladybug Data Collection.

        This is the Diffuse Horizontal Illuminance in lux. (Average amount of illuminance
        in hundreds of lux received from the sky (excluding the solar disk) on a
        horizontal surface during the number of minutes preceding the time indicated.)
        It is not currently used in EnergyPlus calculations. It should have a minimum
        value of 0; missing value for this field is 999999 and will be considered missing
        if greater than or equal to 999900.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(18)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def zenith_luminance(self) -> BH_HourlyContinuousCollection:
        """Return annual Zenith Luminance as a Ladybug Data Collection.

        This is the Zenith Illuminance in Cd/m2. (Average amount of luminance at
        the sky's zenith in tens of Cd/m2 during the number of minutes preceding
        the time indicated.) It is not currently used in EnergyPlus calculations.
        It should have a minimum value of 0; missing value for this field is 9999.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs\/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(19)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def wind_direction(self) -> BH_HourlyContinuousCollection:
        """Return annual Wind Direction as a Ladybug Data Collection.

        This is the Wind Direction in degrees where the convention is that North=0.0,
        East=90.0, South=180.0, West=270.0. (Wind direction in degrees at the time
        indicated. If calm, direction equals zero.) Values can range from 0 to 360.
        Missing value is 999.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(20)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def wind_speed(self) -> BH_HourlyContinuousCollection:
        """Return annual Wind Speed as a Ladybug Data Collection.

        This is the wind speed in m/sec. (Wind speed at time indicated.) Values can
        range from 0 to 40. Missing value is 999.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(21)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def total_sky_cover(self) -> BH_HourlyContinuousCollection:
        """Return annual Total Sky Cover as a Ladybug Data Collection.

        This is the value for total sky cover (tenths of coverage). (i.e. 1 is 1/10
        covered. 10 is total coverage). (Amount of sky dome in tenths covered by clouds
        or obscuring phenomena at the hour indicated at the time indicated.) Minimum
        value is 0; maximum value is 10; missing value is 99.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(22)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def opaque_sky_cover(self) -> BH_HourlyContinuousCollection:
        """Return annual Opaque Sky Cover as a Ladybug Data Collection.

        This is the value for opaque sky cover (tenths of coverage). (i.e. 1 is 1/10
        covered. 10 is total coverage). (Amount of sky dome in tenths covered by
        clouds or obscuring phenomena that prevent observing the sky or higher cloud
        layers at the time indicated.) This is not used unless the field for Horizontal
        Infrared Radiation Intensity is missing and then it is used to calculate
        Horizontal Infrared Radiation Intensity. Minimum value is 0; maximum value is
        10; missing value is 99.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(23)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def visibility(self) -> BH_HourlyContinuousCollection:
        """Return annual Visibility as a Ladybug Data Collection.

        This is the value for visibility in km. (Horizontal visibility at the time
        indicated.) It is not currently used in EnergyPlus calculations. Missing
        value is 9999.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(24)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def ceiling_height(self) -> BH_HourlyContinuousCollection:
        """Return annual Ceiling Height as a Ladybug Data Collection.

        This is the value for ceiling height in m. (77777 is unlimited ceiling height.
        88888 is cirroform ceiling.) It is not currently used in EnergyPlus calculations.
        Missing value is 99999

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(25)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def present_weather_observation(self) -> BH_HourlyContinuousCollection:
        """Return annual Present Weather Observation as a Ladybug Data Collection.

        If the value of the field is 0, then the observed weather codes are taken from
        the following field. If the value of the field is 9, then "missing" weather is
        assumed. Since the primary use of these fields (Present Weather Observation and
        Present Weather Codes) is for rain/wet surfaces, a missing observation field or
        a missing weather code implies no rain.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(26)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def present_weather_codes(self) -> BH_HourlyContinuousCollection:
        """Return annual Present Weather Codes as a Ladybug Data Collection.

        The present weather codes field is assumed to follow the TMY2 conventions for
        this field. Note that though this field may be represented as numeric (e.g. in
        the CSV format), it is really a text field of 9 single digits. This convention
        along with values for each "column" (left to right) is presented in Table 16.
        Note that some formats (e.g. TMY) does not follow this convention - as much as
        possible, the present weather codes are converted to this convention during
        WeatherConverter processing. Also note that the most important fields are those
        representing liquid precipitation - where the surfaces of the building would be
        wet. EnergyPlus uses "Snow Depth" to determine if snow is on the ground.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(27)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def precipitable_water(self) -> BH_HourlyContinuousCollection:
        """Return annual Precipitable Water as a Ladybug Data Collection.

        This is the value for Precipitable Water in mm. (This is not rain - rain is
        inferred from the PresWeathObs field but a better result is from the Liquid
        Precipitation Depth field). It is not currently used in EnergyPlus calculations
        (primarily due to the unreliability of the reporting of this value). Missing
        value is 999.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(28)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def aerosol_optical_depth(self) -> BH_HourlyContinuousCollection:
        """Return annual Aerosol Optical Depth as a Ladybug Data Collection.

        This is the value for Aerosol Optical Depth in thousandths. It is not currently
        used in EnergyPlus calculations. Missing value is .999.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(29)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def snow_depth(self) -> BH_HourlyContinuousCollection:
        """Return annual Snow Depth as a Ladybug Data Collection.

        This is the value for Snow Depth in cm. This field is used to tell when snow
        is on the ground and, thus, the ground reflectance may change. Missing value
        is 999.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(30)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def days_since_last_snowfall(self) -> BH_HourlyContinuousCollection:
        """Return annual Days Since Last Snow Fall as a Ladybug Data Collection.

        This is the value for Days Since Last Snowfall. It is not currently used in
        EnergyPlus calculations. Missing value is 99.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(31)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def albedo(self) -> BH_HourlyContinuousCollection:
        """Return annual Albedo values as a Ladybug Data Collection.

        The ratio (unitless) of reflected solar irradiance to global horizontal
        irradiance. It is not currently used in EnergyPlus.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(32)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def liquid_precipitation_depth(self) -> BH_HourlyContinuousCollection:
        """Return annual liquid precipitation depth as a Ladybug Data Collection.

        The amount of liquid precipitation (mm) observed at the indicated time for the
        period indicated in the liquid precipitation quantity field. If this value is
        not missing, then it is used and overrides the "precipitation" flag as rainfall.
        Conversely, if the precipitation flag shows rain and this field is missing or
        zero, it is set to 1.5 (mm).

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(33)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def liquid_precipitation_quantity(self) -> BH_HourlyContinuousCollection:
        """Return annual Liquid Precipitation Quantity as a Ladybug Data Collection.

        The period of accumulation (hr) for the liquid precipitation depth field.
        It is not currently used in EnergyPlus.

        Read more at: https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v8.4.0/AuxiliaryPrograms.pdf (Chapter 2.9.1)
        """
        _ = self._get_data_by_field(34)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def sky_temperature(self) -> BH_HourlyContinuousCollection:
        """Return annual Sky Temperature as a Ladybug Data Collection.

        This value in degrees Celsius is derived from the Horizontal Infrared
        Radiation Intensity in Wh/m2. It represents the long wave radiant
        temperature of the sky
        Read more at: https://bigladdersoftware.com/epx/docs/8-9/engineering-reference/climate-calculations.html#energyplus-sky-temperature-calculation
        """
        # create sky temperature header
        sky_temp_header = Header(
            data_type=SkyTemperature(),
            unit="C",
            analysis_period=AnalysisPeriod(),
            metadata=self._metadata,
        )

        # calculate sky temperature for each hour
        horiz_ir = self._get_data_by_field(12).values
        sky_temp_data = [calc_sky_temperature(hir) for hir in horiz_ir]
        return BH_HourlyContinuousCollection(sky_temp_header, sky_temp_data)

    @property
    def monthly_ground_temperature(self) -> Dict[float, BH_MonthlyCollection]:
        """Return a dictionary of Monthly Data collections.
        The keys of this dictionary are the depths at which each set
        of temperatures occurs."""
        self._load_header_check()

        modified_dict = {}
        for depth, collection in super().monthly_ground_temperature.items():
            modified_dict[depth] = BH_MonthlyCollection(
                collection.header, collection.values, collection.datetimes
            )
        return modified_dict
