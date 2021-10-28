import json
from typing import List
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype.fraction import HumidityRatio
from ladybug.datatype.specificenergy import Enthalpy
from ladybug.datatype.temperature import SkyTemperature, WetBulbTemperature
from ladybug.epw import EPW
from ladybug.skymodel import calc_sky_temperature
from ladybug.header import Header
from ladybug.psychrometrics import (
    enthalpy_from_db_hr,
    humid_ratio_from_db_rh,
    wet_bulb_from_db_rh,
)
from ladybug.sunpath import Sunpath, Sun
from ladybug.datatype.angle import Angle
import pandas as pd
import numpy as np

from .datacollection import BH_HourlyContinuousCollection


class BH_EPW(EPW):
    def __init__(self, file_path):
        super().__init__(file_path)

    def datetime_index(self) -> pd.DatetimeIndex:
        """Generate a pandas DatetimeIndex for the current epw.

        Returns:
            DatetimeIndex: A pandas DatetimeIndex object.
        """
        n_hours = 8784 if self.is_leap_year else 8760
        year = 2020 if self.is_leap_year else 2021
        return pd.date_range(
            f"{year}-01-01 00:30:00", freq="60T", periods=n_hours, name="timestamp"
        )

    def sun_positions(self) -> List[Sun]:
        """Calculate a set of Sun positions for each hour of the year

        Returns:
            List[Sun]: Annual hourly sun positions
        """
        sunpath = Sunpath.from_location(self.location)
        return [sunpath.calculate_sun_from_hoy(i) for i in range(8760)]

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
                all_series.append(getattr(self, p).to_series())
            except AttributeError as e:
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
    def solar_azimuth(self) -> BH_HourlyContinuousCollection:
        """Calculate annual hourly solar azimuth positions.

        Returns:
            BH_HourlyContinuousCollection: Annual hourly solar azimuth positions.
        """
        _solar_azimuth = BH_HourlyContinuousCollection(
            Header(
                data_type=Angle(),
                unit="radians",
                analysis_period=AnalysisPeriod(),
                metadata={
                    **self.dry_bulb_temperature.header.metadata,
                    **{"description": "Solar Azimuth"},
                },
            ),
            [i.azimuth_in_radians for i in self.sun_positions()],
        )

        return BH_HourlyContinuousCollection(
            _solar_azimuth.header, _solar_azimuth.values
        )

    @property
    def solar_altitude(self) -> BH_HourlyContinuousCollection:
        """Calculate annual hourly apparent solar altitude angles.

        Returns:
            BH_HourlyContinuousCollection: Annual hourly apparent solar altitude angles.
        """
        _solar_altitude = BH_HourlyContinuousCollection(
            Header(
                data_type=Angle(),
                unit="radians",
                analysis_period=AnalysisPeriod(),
                metadata={
                    **self.dry_bulb_temperature.header.metadata,
                    **{"description": "Solar Altitude"},
                },
            ),
            [i.altitude_in_radians for i in self.sun_positions()],
        )

        return BH_HourlyContinuousCollection(
            _solar_altitude.header, _solar_altitude.values
        )

    @property
    def apparent_solar_zenith(self) -> BH_HourlyContinuousCollection:
        """Calculate annual hourly apparent solar zenith angles.

        Returns:
            BH_HourlyContinuousCollection: Annual hourly apparent solar zenith angles.
        """
        _apparent_solar_zenith = BH_HourlyContinuousCollection(
            Header(
                data_type=Angle(),
                unit="radians",
                analysis_period=AnalysisPeriod(),
                metadata={
                    **self.dry_bulb_temperature.header.metadata,
                    **{"description": "Apparent Solar Zenith"},
                },
            ),
            [np.pi / 2 - i for i in self.solar_altitude.values],
        )
        return BH_HourlyContinuousCollection(
            _apparent_solar_zenith.header, _apparent_solar_zenith.values
        )

    @property
    def wet_bulb_temperature(self) -> BH_HourlyContinuousCollection:
        """Calculate an annual hourly wet bulb temperature collection for a given EPW.

        Returns:
            BH_HourlyContinuousCollection: A Wet Bulb Temperature data collection.
        """
        _ = BH_HourlyContinuousCollection.compute_function_aligned(
            wet_bulb_from_db_rh,
            [
                self.dry_bulb_temperature,
                self.relative_humidity,
                self.atmospheric_station_pressure,
            ],
            WetBulbTemperature(),
            "C",
        )
        wet_bulb_temperature = BH_HourlyContinuousCollection(_.header, _.values)
        return wet_bulb_temperature

    @property
    def humidity_ratio(self) -> BH_HourlyContinuousCollection:
        """Calculate an annual hourly humidity ratio collection for a given EPW.

        Returns:
            BH_HourlyContinuousCollection: A Humidity Ratio data collection.
        """
        _ = BH_HourlyContinuousCollection.compute_function_aligned(
            humid_ratio_from_db_rh,
            [
                self.dry_bulb_temperature,
                self.relative_humidity,
                self.atmospheric_station_pressure,
            ],
            HumidityRatio(),
            "fraction",
        )
        humidity_ratio = BH_HourlyContinuousCollection(_.header, _.values)
        return humidity_ratio

    @property
    def enthalpy(self) -> BH_HourlyContinuousCollection:
        """Calculate an annual hourly enthalpy collection.

        Returns:
            BH_HourlyContinuousCollection: A Enthalpy data collection.
        """
        _ = BH_HourlyContinuousCollection.compute_function_aligned(
            enthalpy_from_db_hr,
            [
                self.dry_bulb_temperature,
                self.humidity_ratio,
            ],
            Enthalpy(),
            "kJ/kg",
        )
        enthalpy = BH_HourlyContinuousCollection(_.header, _.values)
        return enthalpy

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

        # calculate sy temperature for each hour
        horiz_ir = self._get_data_by_field(12).values
        sky_temp_data = [calc_sky_temperature(hir) for hir in horiz_ir]
        return BH_HourlyContinuousCollection(sky_temp_header, sky_temp_data)
