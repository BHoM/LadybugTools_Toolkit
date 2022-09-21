import math
from datetime import datetime

import pandas as pd
from ladybug.epw import Location
from ladybug.skymodel import (
    calc_horizontal_infrared,
    calc_sky_temperature,
    estimate_illuminance_from_irradiance,
    get_extra_radiation,
    zhang_huang_solar,
    zhang_huang_solar_split,
)
from ladybug.sunpath import Sunpath


from ladybugtools_toolkit import analytics


@analytics
def scrape_weather(
    station: str,
    start_date: str = "1970-01-01",
    end_date: str = None,
    interpolate: bool = False,
    resample: bool = False,
) -> pd.DataFrame:
    """Scrape historic data from global airport weather stations using their ICAO codes
        (https://en.wikipedia.org/wiki/List_of_airports_by_IATA_and_ICAO_code)

    Args:
        station (str):
            Airport ICAO code.
        start_date (str, optional):
            Date from which records will be searched. Defaults to "1970-01-01".
        end_date (str, optional):
            Date until which records will be searched. Defaults to None.
        interpolate (bool, optional):
            Set to True to interpolate gaps smaller than 2-hours. Defaults to False.
        resample (bool, optional):
            Set to True to resample the data to 0 and 30 minutes past the hour. Defaults to False.

    Returns:
        pd.DataFrame:
            A Pandas DataFrame containing time-indexed weather data.
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")

    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Scrape data from source website (https://mesonet.agron.iastate.edu/request/download.phtml)
    uri = f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={station}&year1={start_date.year}&month1={start_date.month}&day1={start_date.day}&year2={end_date.year}&month2={end_date.month}&day2={end_date.day}&tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=null&trace=null&direct=no&data=tmpc&data=dwpc&data=relh&data=drct&data=sknt&data=alti&data=p01m&data=vsby&data=skyc1&data=skyc2&data=skyc3"
    df = pd.read_csv(
        uri,
        header=0,
        index_col="valid",
        parse_dates=True,
        na_values=["M", "null"],
        low_memory=False,
    )

    # Post-process data into right units
    df["sknt"] = df.sknt / 1.94384  # convert knots to m/s
    # convert inches of mercury (Hg) to Pa
    df["alti"] = df.alti * 3386.38866667
    df["vsby"] = df["vsby"] * 1.60934  # convert miles to kilometres

    # Get sky clearness
    rplc = {
        "   ": 0,
        "CLR": 0,
        "NCD": 0,
        "NSC": 0,
        "SKC": 0,
        "///": 0,
        "FEW": 1.5,
        "SCT": 3.5,
        "BKN": 6,
        "OVC": 8,
        "VV ": 8,
        "VV": 8,
    }

    for i in ["skyc1", "skyc2", "skyc3"]:
        df[i] = df[i].fillna("NSC").replace(rplc) / 8 * 10
    df["opaque_sky_cover"] = df[["skyc1", "skyc2", "skyc3"]].mean(axis=1)
    df.drop(["skyc1", "skyc2", "skyc3"], axis=1, inplace=True)

    # Rename headers
    renamer = {
        "lon": "longitude",
        "lat": "latitude",
        "elevation": "elevation",
        "tmpc": "dry_bulb_temperature",
        "dwpc": "dew_point_temperature",
        "relh": "relative_humidity",
        "drct": "wind_direction",
        "sknt": "wind_speed",
        "alti": "atmospheric_station_pressure",
        "p01m": "liquid_precipitation_depth",
        "vsby": "visibility",
    }
    df.rename(columns=renamer, inplace=True)
    df.index.name = None

    # Calculate HIR and sky temperature
    df["horizontal_infrared_radiation_intensity"] = [
        calc_horizontal_infrared(
            row.opaque_sky_cover, row.dry_bulb_temperature, row.dew_point_temperature
        )
        for row in df.itertuples()
    ]
    df["sky_temperature"] = [
        calc_sky_temperature(
            row.horizontal_infrared_radiation_intensity, source_emissivity=1
        )
        for row in df.itertuples()
    ]

    # Calculate sun locations
    loc = Location(
        latitude=df.latitude.values[0],
        longitude=df.longitude.values[0],
        elevation=df.elevation.values[0],
        city=f"ICAO-{df.station.values[0]}",
    )
    altitude_in_radians = []
    azimuth_in_radians = []
    for i in df.index:
        sunpath = Sunpath.from_location(loc).calculate_sun_from_date_time(i)
        altitude_in_radians.append(sunpath.altitude_in_radians)
        azimuth_in_radians.append(sunpath.azimuth_in_radians)
    df["solar_altitude"] = altitude_in_radians
    df["solar_azimuth"] = azimuth_in_radians

    # Calculate irradiance and illuminance
    df["temp_offset_3"] = df.dry_bulb_temperature.shift(3)
    dir_norm, dif_horiz = zhang_huang_solar_split(
        df.solar_altitude * 180 / math.pi,
        df.index.day_of_year,
        df.opaque_sky_cover,
        df.relative_humidity,
        df.dry_bulb_temperature,
        df.temp_offset_3,
        df.wind_speed,
        df.atmospheric_station_pressure,
    )
    df["direct_normal_radiation"] = dir_norm
    df["diffuse_horizontal_radiation"] = dif_horiz
    df["global_horizontal_radiation"] = [
        zhang_huang_solar(
            row.solar_altitude * 180 / math.pi,
            row.opaque_sky_cover,
            row.relative_humidity,
            row.dry_bulb_temperature,
            row.temp_offset_3,
            row.wind_speed,
            irr_0=1355,
        )
        for row in df.itertuples()
    ]
    df["extraterrestrial_horizontal_radiation"] = [
        get_extra_radiation(i) for i in df.index.day_of_year
    ]
    df["extraterrestrial_horizontal_radiation"] = df[
        "extraterrestrial_horizontal_radiation"
    ].where(df.global_horizontal_radiation != 0, 0)
    df["direct_normal_radiation"].fillna(0, inplace=True)
    df["diffuse_horizontal_radiation"].fillna(0, inplace=True)
    df["global_horizontal_radiation"].fillna(0, inplace=True)

    vals = []
    for _, row in df.iterrows():
        vals.append(
            estimate_illuminance_from_irradiance(
                row.solar_altitude * 180 / math.pi,
                row.global_horizontal_radiation,
                row.direct_normal_radiation,
                row.diffuse_horizontal_radiation,
                row.dew_point_temperature,
            )
        )
    gh_ill, dn_ill, dh_ill, z_lum = list(zip(*vals))
    df["direct_normal_illuminance"] = dn_ill
    df["diffuse_horizontal_illuminance"] = dh_ill
    df["global_horizontal_illuminance"] = gh_ill
    df["zenith_luminance"] = z_lum
    df.drop(["temp_offset_3"], axis=1, inplace=True)

    if interpolate:
        df.interpolate(limit=4, inplace=True)

    if resample:
        df = df.resample("30T").mean()

    return df
