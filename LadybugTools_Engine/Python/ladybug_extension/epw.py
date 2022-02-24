import sys
sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import numpy as np
import pandas as pd
from ladybug import datatype
from ladybug.epw import EPW, AnalysisPeriod, Header, HourlyContinuousCollection
from ladybug.location import Location
from ladybug.psychrometrics import (
    enthalpy_from_db_hr,
    humid_ratio_from_db_rh,
    rel_humid_from_db_dpt,
    wet_bulb_from_db_rh,
)
from ladybug.skymodel import clearness_index
from ladybug.sunpath import Sun, Sunpath
from ladybug.wea import Wea

from ladybug_extension.datacollection import to_datetimes, to_hourly, to_series


def from_dataframe(dataframe: pd.DataFrame, location: Location = None) -> EPW:
    """Create an EPW object from a Pandas DataFrame with named columns.

    Args:
        dataframe (pd.DataFrame): A Pandas DataFrame with named columns.
        location (Location, optional): A ladybug Location object. Defaults to None.

    Returns:
        EPW: An EPW object.
    """

    # Check dataframe shape for leapedness and length
    if dataframe.index.is_leap_year.any():
        leap_yr = True
        assert (
            len(dataframe.index) == 8784
        ), "The dataframe must have 8784 rows for leap years."
    else:
        leap_yr = False
        assert (
            len(dataframe.index) == 8760
        ), "The dataframe must have 8760 rows for non-leap years."

    if location is None:
        location = Location()
    try:
        location.source += "[Custom EPW from Pandas DataFrame]"
    except TypeError:
        location.source = "[Custom EPW from Pandas DataFrame]"

    epw_obj = EPW.from_missing_values(is_leap_year=leap_yr)
    epw_obj.location = location

    # Assign data to the EPW
    try:
        epw_obj.aerosol_optical_depth.values = dataframe[
            "Aerosol Optical Depth (fraction)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.albedo.values = dataframe["Albedo (fraction)"].values
    except KeyError:
        pass

    try:
        epw_obj.atmospheric_station_pressure.values = dataframe[
            "Atmospheric Station Pressure (Pa)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.ceiling_height.values = dataframe["Ceiling Height (m)"].values
    except KeyError:
        pass

    try:
        epw_obj.extraterrestrial_direct_normal_radiation.values = dataframe[
            "Extraterrestrial Direct Normal Radiation (Wh/m2)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.extraterrestrial_horizontal_radiation.values = dataframe[
            "Extraterrestrial Horizontal Radiation (Wh/m2)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.liquid_precipitation_depth.values = dataframe[
            "Liquid Precipitation Depth (mm)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.liquid_precipitation_quantity.values = dataframe[
            "Liquid Precipitation Quantity (fraction)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.days_since_last_snowfall.values = dataframe[
            "Days Since Last Snowfall (day)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.dry_bulb_temperature.values = dataframe[
            "Dry Bulb Temperature (C)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.dew_point_temperature.values = dataframe[
            "Dew Point Temperature (C)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.wind_speed.values = dataframe["Wind Speed (m/s)"].values
    except KeyError:
        pass

    try:
        epw_obj.wind_direction.values = dataframe["Wind Direction (degrees)"].values
    except KeyError:
        pass

    try:
        epw_obj.direct_normal_radiation.values = dataframe[
            "Direct Normal Radiation (Wh/m2)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.snow_depth.values = dataframe["Snow Depth (cm)"].values
    except KeyError:
        pass

    try:
        epw_obj.diffuse_horizontal_radiation.values = dataframe[
            "Diffuse Horizontal Radiation (Wh/m2)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.horizontal_infrared_radiation_intensity.values = dataframe[
            "Horizontal Infrared Radiation Intensity (W/m2)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.direct_normal_illuminance.values = dataframe[
            "Direct Normal Illuminance (lux)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.diffuse_horizontal_illuminance.values = dataframe[
            "Diffuse Horizontal Illuminance (lux)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.precipitable_water.values = dataframe["Precipitable Water (mm)"].values
    except KeyError:
        pass

    try:
        epw_obj.present_weather_codes.values = dataframe[
            "Present Weather Codes (codes)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.present_weather_observation.values = dataframe[
            "Present Weather Observation (observation)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.total_sky_cover.values = dataframe["Total Sky Cover (tenths)"].values
    except KeyError:
        pass

    try:
        epw_obj.opaque_sky_cover.values = dataframe["Opqaue Sky Cover (tenths)"].values
    except KeyError:
        pass

    try:
        epw_obj.visibility.values = dataframe["Visibility (km)"].values
    except KeyError:
        pass

    try:
        epw_obj.zenith_luminance.values = dataframe["Zenith Luminance (cd/m2)"].values
    except KeyError:
        pass

    try:
        epw_obj.relative_humidity.values = dataframe["Relative Humidity (%)"].values
    except KeyError:
        epw_obj.relative_humidity.values = (
            HourlyContinuousCollection.compute_function_aligned(
                rel_humid_from_db_dpt,
                [epw_obj.dry_bulb_temperature, epw_obj.dew_point_temperature],
                datatype.fraction.RelativeHumidity(),
                "%",
            ).values
        )

    try:
        epw_obj.global_horizontal_radiation.values = dataframe[
            "Global Horizontal Radiation (Wh/m2)"
        ].values
    except KeyError:
        wea = Wea(
            location,
            epw_obj.direct_normal_radiation,
            epw_obj.diffuse_horizontal_radiation,
        )
        epw_obj.global_horizontal_radiation.values = (
            wea.global_horizontal_irradiance.values
        )

    try:
        epw_obj.global_horizontal_illuminance.values = dataframe[
            "Global Horizontal Illuminance (lux)"
        ].values
    except KeyError:
        glob_horiz = []
        sp = Sunpath.from_location(location)
        sp.is_leap_year = leap_yr
        for dt, dni, dhi in zip(
            epw_obj.direct_normal_illuminance.datetimes,
            epw_obj.direct_normal_illuminance,
            epw_obj.diffuse_horizontal_illuminance,
        ):
            sun = sp.calculate_sun_from_date_time(dt)
            glob_horiz.append(dhi + dni * np.sin(np.radians(sun.altitude)))
        epw_obj.global_horizontal_illuminance.values = glob_horiz

    return epw_obj


def _epw_equality(epw0: EPW, epw1: EPW, include_header: bool = False) -> bool:
    """Check for equality between two EPW objects, with regards to the data contained within.

    Args:
        epw0 (EPW): A ladybug EPW object.
        epw1 (EPW): A ladybug EPW object.
        include_header (bool, optional): Include the str repsresentation of the EPW files header in the comparison. Defaults to False.

    Returns:
        bool: True if the two EPW objects are equal, False otherwise.
    """
    if not isinstance(epw0, EPW) or not isinstance(epw1, EPW):
        raise TypeError("Both inputs must be of type EPW.")

    if include_header:
        if epw0.header != epw1.header:
            return False

    # Check key metrics
    dbt_match = epw0.dry_bulb_temperature == epw1.dry_bulb_temperature
    rh_match = epw0.relative_humidity == epw1.relative_humidity
    dpt_match = epw0.dew_point_temperature == epw1.dew_point_temperature
    ws_match = epw0.wind_speed == epw1.wind_speed
    wd_match = epw0.wind_direction == epw1.wind_direction
    ghr_match = epw0.global_horizontal_radiation == epw1.global_horizontal_radiation
    dnr_match = epw0.direct_normal_radiation == epw1.direct_normal_radiation
    dhr_match = epw0.diffuse_horizontal_radiation == epw1.diffuse_horizontal_radiation
    atm_match = epw0.atmospheric_station_pressure == epw1.atmospheric_station_pressure

    return all(
        [
            dbt_match,
            rh_match,
            dpt_match,
            ws_match,
            wd_match,
            ghr_match,
            dnr_match,
            dhr_match,
            atm_match,
        ]
    )


def radiation_tilt_orientation_matrix(epw: EPW) -> pd.DataFrame:
    """Compute the annual cumulative radiation matrix per surface tilt and orientation, for a given EPW object.

    Args:
        epw (EPW): The EPW object for which this calculation is made.

    Returns:
        pd.DataFrame: _description_
    """    
    
    wea = Wea.from_annual_values(epw.location, epw.direct_normal_radiation.values, epw.diffuse_horizontal_radiation.values, is_leap_year=epw.is_leap_year)
    
    # I implement a bit of a hack here, to calculate only the Eastern insolation - then mirror it about the North-South axis to get the whole matrix
    altitudes = np.linspace(0, 90, 10)
    azimuths = np.linspace(0, 180, 19)
    combinations = np.array(list(itertools.product(altitudes, azimuths)))

    return HourlyContinuousCollection(
        header=Header(
            data_type=datatype.fraction.Fraction(name="Clearness Index"),
            unit="fraction",
            analysis_period=AnalysisPeriod(),
        ),
        values=ci,
    )

if __name__ == "__main__":
    epw = EPW(r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\test\GBR_London.Gatwick.037760_IWEC.epw")
    print(to_dataframe(epw))
