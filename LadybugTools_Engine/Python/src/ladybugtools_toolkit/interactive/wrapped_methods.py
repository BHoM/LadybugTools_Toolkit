"""Wrapped methods for safer handling of Ladybug Comfort methods."""

import functools
import warnings
from threading import Thread

import numpy as np
from ladybug.psychrometrics import dew_point_from_db_rh_fast
from ladybug_comfort.asv import actual_sensation_vote
from ladybug_comfort.at import apparent_temperature
from ladybug_comfort.di import discomfort_index
from ladybug_comfort.hi import heat_index
from ladybug_comfort.humidex import humidex
from ladybug_comfort.pet import physiologic_equivalent_temperature
from ladybug_comfort.pmv import pierce_set
from ladybug_comfort.ts import thermal_sensation
from ladybug_comfort.utci import universal_thermal_climate_index
from ladybug_comfort.wbgt import wet_bulb_globe_temperature
from ladybug_comfort.wc import windchill_temp

from . import (
    ATMOSPHERIC_PRESSURE,
    PERSON_AGE,
    PERSON_HEIGHT,
    PERSON_MASS,
    PERSON_POSITION,
    PERSON_SEX,
    TERRAIN_ROUGHNESS_LENGTH,
)


class TimeoutException(Exception):
    """Custom exception for function timeout."""


def timeout(seconds_before_timeout: int) -> None:
    """Add a timeout to a function."""

    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [
                TimeoutException(
                    f"{func.__name__} timed-out - {seconds_before_timeout} seconds exceeded"
                )
            ]

            def new_function():
                """Wrapped function to run in a thread."""
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:  # pylint: disable=broad-except
                    res[0] = e

            t = Thread(target=new_function)
            t.daemon = True
            try:
                t.start()
                t.join(seconds_before_timeout)
            except Exception as e:
                print("error starting thread")
                raise e
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret

        return wrapper

    return deco


def _physiologic_equivalent_temperature(
    air_temperature: float,
    mean_radiant_temperature: float,
    air_velocity: float,
    relative_humidity: float,
    metabolic_rate: float,
    clo_value: float,
) -> float:
    if relative_humidity == 0:
        relative_humidity = 0.001
    if clo_value == 0:
        clo_value = 0.001

    # create time-wrapped func, and ignore warnings
    @timeout(seconds_before_timeout=1)
    def temp_func():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.float16(
                physiologic_equivalent_temperature(
                    ta=air_temperature,
                    tr=mean_radiant_temperature,
                    vel=air_velocity,
                    rh=relative_humidity,
                    met=metabolic_rate,
                    clo=clo_value,
                    age=PERSON_AGE,
                    sex=PERSON_SEX,
                    ht=PERSON_HEIGHT,
                    m_body=PERSON_MASS,
                    pos=PERSON_POSITION,
                    b_press=ATMOSPHERIC_PRESSURE,
                )["pet"]
            )

    return temp_func()


def _universal_thermal_climate_index(
    air_temperature: float,
    mean_radiant_temperature: float,
    air_velocity: float,
    relative_humidity: float,
) -> float:
    """Wrapped version of UTCI.

    Notes:
        This assumes vel is at person height, and translates it to 10m.
    """
    vel_10 = air_velocity * (
        np.log(10 / TERRAIN_ROUGHNESS_LENGTH) / np.log(PERSON_HEIGHT / TERRAIN_ROUGHNESS_LENGTH)
    )
    utci = universal_thermal_climate_index(
        ta=air_temperature, tr=mean_radiant_temperature, vel=vel_10, rh=relative_humidity
    )

    return np.clip(utci, -60, 60).astype(np.float16)


def _wet_bulb_globe_temperature(
    air_temperature: float,
    mean_radiant_temperature: float,
    air_velocity: float,
    relative_humidity: float,
) -> float:
    """Wrapped version of WBGT."""
    return np.float16(
        wet_bulb_globe_temperature(
            ta=air_temperature, mrt=mean_radiant_temperature, ws=air_velocity, rh=relative_humidity
        )
    )


def _humidex(air_temperature: float, relative_humidity: float) -> float:
    """Wrapped version of Humidex."""
    if relative_humidity == 0:
        relative_humidity = 0.001
    tdp = dew_point_from_db_rh_fast(db_temp=air_temperature, rel_humid=relative_humidity)
    return np.float16(humidex(ta=air_temperature, tdp=tdp))


def _actual_sensation_vote(
    air_temperature: float,
    solar_radiation: float,
    air_velocity: float,
    relative_humidity: float,
) -> float:
    """Wrapped version of Actual Sensation Vote."""
    return np.float16(
        actual_sensation_vote(
            ta=air_temperature, ws=air_velocity, rh=relative_humidity, sr=solar_radiation
        )
    )


def _apparent_temperature(
    air_temperature: float,
    air_velocity: float,
    relative_humidity: float,
) -> float:
    """Wrapped version of Apparent Temperature."""
    return np.float16(
        apparent_temperature(ta=air_temperature, rh=relative_humidity, ws=air_velocity * 3.6)
    )


def _discomfort_index(
    air_temperature: float,
    relative_humidity: float,
) -> float:
    """Wrapped version of Discomfort Index."""
    return np.float16(discomfort_index(ta=air_temperature, rh=relative_humidity))


def _heat_index(
    air_temperature: float,
    relative_humidity: float,
) -> float:
    """Wrapped version of Heat Index."""
    return np.float16(heat_index(ta=air_temperature, rh=relative_humidity))


def _standard_effective_temperature(
    air_temperature: float,
    mean_radiant_temperature: float,
    air_velocity: float,
    relative_humidity: float,
    metabolic_rate: float,
    clo_value: float,
    **kwargs,
) -> float:
    """Wrapped version of Standard Effective Temperature."""
    return np.float16(
        pierce_set(
            ta=air_temperature,
            tr=mean_radiant_temperature,
            vel=air_velocity,
            rh=relative_humidity,
            met=metabolic_rate,
            clo=clo_value,
            **kwargs,
        )
    )


def _thermal_sensation(
    air_temperature: float,
    air_velocity: float,
    relative_humidity: float,
    solar_radiation: float,
) -> float:
    """Wrapped version of Thermal Sensation.

    Notes:
        This method includes an approximation of ground temperature based on
        solar radiation and air temperature. It's not very good, but it's fast.

    """
    # TODO - create better way of determining ground surface temperature here ... but FAST!
    # for now it just assumes its linearly correlated with solar radiation to a max of 70C
    ground_temperature = float(np.interp(solar_radiation, [0, 1400], [air_temperature, 70]))

    return np.float16(
        thermal_sensation(
            ta=air_temperature,
            ws=air_velocity,
            rh=relative_humidity,
            sr=solar_radiation,
            tground=ground_temperature,
        )
    )


def _windchill_temp(
    air_temperature: float,
    air_velocity: float,
) -> float:
    """Wrapped version of Windchill Temperature."""
    return np.float16(windchill_temp(ta=air_temperature, ws=air_velocity))
