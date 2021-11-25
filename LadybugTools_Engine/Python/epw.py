from __future__ import annotations

import json
from datetime import datetime, time, timedelta
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
    enthalpy_from_db_hr,
    humid_ratio_from_db_rh,
    wet_bulb_from_db_rh,
)
from ladybug.skymodel import calc_sky_temperature, clearness_index
from ladybug.sunpath import Sunpath

from datacollection import BH_HourlyContinuousCollection, BH_MonthlyCollection


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

    @property
    def sun_position(self) -> BH_HourlyContinuousCollection:
        """Calculate a set of Sun positions for each hour of the year."""
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
        """Get a list of datetimes for each hour of the year."""
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
        """Calculate solar time for each hour of the year."""

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
        """Calculate solar time for each hour of the year."""
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
        """Calculate annual hourly solar azimuth angle."""
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
        """Calculate annual hourly solar azimuth angle in radians."""
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
        """Calculate annual hourly solar altitude angle."""
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
        """Calculate annual hourly solar altitude angle in radians."""
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
        """Calculate annual hourly apparent solar zenith angles."""
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
        """Calculate an annual hourly wet bulb temperature collection for a given EPW."""
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
        """Calculate an annual hourly humidity ratio collection for a given EPW."""
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
        """Calculate an annual hourly enthalpy collection."""
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
        """Return the clearness index value for each hour of the year."""
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
        """Return annual Dry Bulb Temperature as a Ladybug Data Collection."""
        _ = self._get_data_by_field(6)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def dew_point_temperature(self) -> BH_HourlyContinuousCollection:
        """Return annual Dew Point Temperature as a Ladybug Data Collection."""
        _ = self._get_data_by_field(7)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def relative_humidity(self) -> BH_HourlyContinuousCollection:
        """Return annual Relative Humidity as a Ladybug Data Collection."""
        _ = self._get_data_by_field(8)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def atmospheric_station_pressure(self) -> BH_HourlyContinuousCollection:
        """Return annual Atmospheric Station Pressure as a Ladybug Data Collection."""
        _ = self._get_data_by_field(9)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def extraterrestrial_horizontal_radiation(self) -> BH_HourlyContinuousCollection:
        """Return annual Extraterrestrial Horizontal Radiation as a Ladybug Data Collection."""
        _ = self._get_data_by_field(10)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def extraterrestrial_direct_normal_radiation(self) -> BH_HourlyContinuousCollection:
        """Return annual Extraterrestrial Direct Normal Radiation as a Ladybug Data Collection."""
        _ = self._get_data_by_field(11)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def horizontal_infrared_radiation_intensity(self) -> BH_HourlyContinuousCollection:
        """Return annual Horizontal Infrared Radiation Intensity as a Ladybug Data Collection."""
        _ = self._get_data_by_field(12)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def global_horizontal_radiation(self) -> BH_HourlyContinuousCollection:
        """Return annual Global Horizontal Radiation as a Ladybug Data Collection."""
        _ = self._get_data_by_field(13)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def direct_normal_radiation(self) -> BH_HourlyContinuousCollection:
        """Return annual Direct Normal Radiation as a Ladybug Data Collection."""
        _ = self._get_data_by_field(14)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def diffuse_horizontal_radiation(self) -> BH_HourlyContinuousCollection:
        """Return annual Diffuse Horizontal Radiation as a Ladybug Data Collection."""
        _ = self._get_data_by_field(15)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def global_horizontal_illuminance(self) -> BH_HourlyContinuousCollection:
        """Return annual Global Horizontal Illuminance as a Ladybug Data Collection."""
        _ = self._get_data_by_field(16)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def direct_normal_illuminance(self) -> BH_HourlyContinuousCollection:
        """Return annual Direct Normal Illuminance as a Ladybug Data Collection."""
        _ = self._get_data_by_field(17)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def diffuse_horizontal_illuminance(self) -> BH_HourlyContinuousCollection:
        """Return annual Diffuse Horizontal Illuminance as a Ladybug Data Collection."""
        _ = self._get_data_by_field(18)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def zenith_luminance(self) -> BH_HourlyContinuousCollection:
        """Return annual Zenith Luminance as a Ladybug Data Collection."""
        _ = self._get_data_by_field(19)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def wind_direction(self) -> BH_HourlyContinuousCollection:
        """Return annual Wind Direction as a Ladybug Data Collection."""
        _ = self._get_data_by_field(20)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def wind_speed(self) -> BH_HourlyContinuousCollection:
        """Return annual Wind Speed as a Ladybug Data Collection."""
        _ = self._get_data_by_field(21)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def total_sky_cover(self) -> BH_HourlyContinuousCollection:
        """Return annual Total Sky Cover as a Ladybug Data Collection."""
        _ = self._get_data_by_field(22)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def opaque_sky_cover(self) -> BH_HourlyContinuousCollection:
        """Return annual Opaque Sky Cover as a Ladybug Data Collection."""
        _ = self._get_data_by_field(23)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def visibility(self) -> BH_HourlyContinuousCollection:
        """Return annual Visibility as a Ladybug Data Collection."""
        _ = self._get_data_by_field(24)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def ceiling_height(self) -> BH_HourlyContinuousCollection:
        """Return annual Ceiling Height as a Ladybug Data Collection."""
        _ = self._get_data_by_field(25)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def present_weather_observation(self) -> BH_HourlyContinuousCollection:
        """Return annual Present Weather Observation as a Ladybug Data Collection."""
        _ = self._get_data_by_field(26)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def present_weather_codes(self) -> BH_HourlyContinuousCollection:
        """Return annual Present Weather Codes as a Ladybug Data Collection."""
        _ = self._get_data_by_field(27)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def precipitable_water(self) -> BH_HourlyContinuousCollection:
        """Return annual Precipitable Water as a Ladybug Data Collection."""
        _ = self._get_data_by_field(28)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def aerosol_optical_depth(self) -> BH_HourlyContinuousCollection:
        """Return annual Aerosol Optical Depth as a Ladybug Data Collection."""
        _ = self._get_data_by_field(29)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def snow_depth(self) -> BH_HourlyContinuousCollection:
        """Return annual Snow Depth as a Ladybug Data Collection."""
        _ = self._get_data_by_field(30)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def days_since_last_snowfall(self) -> BH_HourlyContinuousCollection:
        """Return annual Days Since Last Snow Fall as a Ladybug Data Collection."""
        _ = self._get_data_by_field(31)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def albedo(self) -> BH_HourlyContinuousCollection:
        """Return annual Albedo values as a Ladybug Data Collection."""
        _ = self._get_data_by_field(32)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def liquid_precipitation_depth(self) -> BH_HourlyContinuousCollection:
        """Return annual liquid precipitation depth as a Ladybug Data Collection."""
        _ = self._get_data_by_field(33)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def liquid_precipitation_quantity(self) -> BH_HourlyContinuousCollection:
        """Return annual Liquid Precipitation Quantity as a Ladybug Data Collection."""
        _ = self._get_data_by_field(34)
        return BH_HourlyContinuousCollection(_.header, _.values)

    @property
    def sky_temperature(self) -> BH_HourlyContinuousCollection:
        """Return annual Sky Temperature as a Ladybug Data Collection."""
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
        """Return a dictionary of Monthly Data collections."""
        self._load_header_check()

        modified_dict = {}
        for depth, collection in super().monthly_ground_temperature.items():
            modified_dict[depth] = BH_MonthlyCollection(
                collection.header, collection.values, collection.datetimes
            )
        return modified_dict
