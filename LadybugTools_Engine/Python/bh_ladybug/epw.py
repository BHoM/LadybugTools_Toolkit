from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from typing import Dict

import numpy as np
from ladybug.datatype import temperature
from ladybug.designday import DesignDay
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

from .analysisperiod import BH_AnalysisPeriod

from .header import BH_Header

from .datacollection import BH_HourlyContinuousCollection, BH_MonthlyCollection
from .location import BH_Location


class BH_EPW(EPW):
    def __init__(self, file_path) -> BH_EPW:
        super().__init__(file_path)
        __slots__ = super().__slots__

    # def __repr__(self):
    #     """EPW representation."""
    #     return "EPW file Data for [%s]" % self.location.city

    def _type(self) -> str:
        return self.__class__.__name__

    def to_dataframe(self) -> pd.DataFrame:
        """Create a Pandas DataFrame from the EPW object."""

        all_series = []
        for p in dir(self):
            try:
                all_series.append(getattr(self, p).to_series())
            except (AttributeError, TypeError, ZeroDivisionError) as e:
                pass

        for k, v in self.monthly_ground_temperature.items():
            hourly_collection = v.to_hourly()
            hourly_series = hourly_collection.to_series()
            hourly_series.name = f"{hourly_series.name} at {k}m"
            all_series.append(hourly_series)

        df = pd.concat(all_series, axis=1).sort_index(axis=1)

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

    @property
    def location(self) -> BH_Location:
        """Return location data."""
        _ = self._location
        return BH_Location(
            _.city,
            _.state,
            _.country,
            _.latitude,
            _.longitude,
            _.time_zone,
            _.elevation,
            _.station_id,
            _.source,
        )

    def _import_header(self, header_lines):
        """Set EPW design days, typical weeks, and ground temperatures from header lines.
        Modified to enable parsing of EPWs within missing header data and incorrect number of fields.
        """
        # parse the heating, cooling and extreme design conditions.
        dday_data = header_lines[1].strip().split(",")
        if len(dday_data) >= 2 and int(dday_data[1]) == 1:
            if dday_data[4] == "Heating":
                for key, val in zip(DesignDay.HEATING_KEYS, dday_data[5:20]):
                    self._heating_dict[key] = val
            if dday_data[20] == "Cooling":
                for key, val in zip(DesignDay.COOLING_KEYS, dday_data[21:53]):
                    self._cooling_dict[key] = val
            if dday_data[53] == "Extremes":
                for key, val in zip(DesignDay.EXTREME_KEYS, dday_data[54:70]):
                    self._extremes_dict[key] = val

        # parse typical and extreme periods into analysis periods.
        week_data = header_lines[2].split(",")
        try:
            num_weeks = int(week_data[1]) if len(week_data) >= 2 else 0
        except ValueError:
            num_weeks = 0
        st_ind = 2
        for _ in range(num_weeks):
            week_dat = week_data[st_ind : st_ind + 4]
            st_ind += 4
            st = [int(num) for num in week_dat[2].split("/")]
            end = [int(num) for num in week_dat[3].split("/")]
            if len(st) == 3:
                a_per = BH_AnalysisPeriod(st[1], st[2], 0, end[1], end[2], 23)
            elif len(st) == 2:
                a_per = BH_AnalysisPeriod(st[0], st[1], 0, end[0], end[1], 23)
            if "Max" in week_dat[0] and week_dat[1] == "Extreme":
                self._extreme_hot_weeks[week_dat[0]] = a_per
            elif "Min" in week_dat[0] and week_dat[1] == "Extreme":
                self._extreme_cold_weeks[week_dat[0]] = a_per
            elif week_dat[1] == "Typical":
                self._typical_weeks[week_dat[0]] = a_per

        # parse the monthly ground temperatures in the header.
        grnd_data = header_lines[3].strip().split(",")
        try:
            num_depths = int(grnd_data[1]) if len(grnd_data) >= 2 else 0
        except ValueError:
            num_depths = 0
        st_ind = 2
        for _ in range(num_depths):
            header_meta = dict(self._metadata)  # copying the metadata dictionary
            header_meta["depth"] = float(grnd_data[st_ind])
            header_meta["soil conductivity"] = grnd_data[st_ind + 1]
            header_meta["soil density"] = grnd_data[st_ind + 2]
            header_meta["soil specific heat"] = grnd_data[st_ind + 3]
            grnd_header = BH_Header(
                temperature.GroundTemperature(), "C", AnalysisPeriod(), header_meta
            )
            grnd_vals = [float(x) for x in grnd_data[st_ind + 4 : st_ind + 16]]
            self._monthly_ground_temps[float(grnd_data[st_ind])] = BH_MonthlyCollection(
                grnd_header, grnd_vals, list(range(12))
            )
            st_ind += 16

        # parse leap year, daylight savings and comments.
        leap_dl_sav = header_lines[4].strip().split(",")
        self._is_leap_year = True if leap_dl_sav[1] == "Yes" else False
        self.daylight_savings_start = leap_dl_sav[2]
        self.daylight_savings_end = leap_dl_sav[3]
        comments_1 = header_lines[5].strip().split(",")
        if len(comments_1) > 0:
            self.comments_1 = ",".join(comments_1[1:])
        comments_2 = header_lines[6].strip().split(",")
        if len(comments_2) > 0:
            self.comments_2 = ",".join(comments_2[1:])

        self._is_header_loaded = True
