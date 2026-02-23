import warnings
from datetime import date, timedelta
from typing import Optional, Union

import numpy as np
import pandas as pd
from ladybug.location import Location
from pvlib.irradiance import campbell_norman, get_extra_radiation
from pvlib.location import Location as pvlib_location

from ..ladybug_extension.location import (
    location_to_pytz_fixed_offset,
)


def radiation_from_location(
    location: Location,
    start_date: Union[str, date] = "2017-01-01",
    end_date: Union[str, date] = "2017-12-31",
    total_sky_cover: Optional[Union[float, list[float]]] = None,
) -> pd.DataFrame:
    """Estimate solar radiation for a given location and datetimes using PVLib.

    Args:
        location (Location):
            A ladybug Location object.
        start_date (Union[str, date]):
            The start date of the analysis period.
            Defaults to "2017-01-01".
        end_date (Union[str, date]):
            The end date of the analysis period.
            Defaults to "2017-12-31".
        total_sky_cover (Union[float, list[float]] , optional):
            Cloud cover percentage (0-1) or a list of percentages for each datetime.
            Defaults to None, which assumes no cloud cover (0%).

    Returns:
        pd.DataFrame:
            A DataFrame containing the estimated solar radiation values.
            The DataFrame will have the following columns:
                - Direct Normal Irradiance (W/m2).
                - Diffuse Horizontal Irradiance (W/m2).
                - Global Horizontal Irradiance (W/m2).

    """
    # checks
    if not isinstance(location, Location):
        raise TypeError("location must be a ladybug Location object.")
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).to_pydatetime().date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).to_pydatetime().date()
    if not isinstance(start_date, date):
        raise TypeError("start_date must be a date object.")
    if not isinstance(end_date, date):
        raise TypeError("end_date must be a date object.")
    if start_date > end_date:
        raise ValueError("start_date must be less than end_date.")

    # create the list of datetimes being queried
    datetimes = pd.date_range(
        start=start_date,
        end=end_date + timedelta(days=1),
        freq="h",
        inclusive="both",
        tz=location_to_pytz_fixed_offset(location),
    )[:-1]

    # process cloudcover data
    if total_sky_cover is None:
        warnings.warn(
            "cloud_cover is None. This will be set to 0% for all datetimes. This can result in higher than expected radiation values. Try to estimate a rough cloud-cover for the location."
        )
        total_sky_cover = [0] * len(datetimes)
    if isinstance(total_sky_cover, (float, int)):
        total_sky_cover = [total_sky_cover] * len(datetimes)
    if isinstance(total_sky_cover, (list, tuple)):
        if any([i < 0 or i > 1 for i in total_sky_cover]):
            raise ValueError("cloud_cover must be between 0 and 1.")
    if len(total_sky_cover) != len(datetimes):
        raise ValueError(
            f"cloud_cover must be the same length as the date range ({len(total_sky_cover)} != {len(datetimes)})."
        )
    total_sky_cover = np.array(total_sky_cover) * 100.0  # convert to percentage

    # modify the location so that its source is pvlib
    location = location.duplicate()
    location.source = "pvlib"
    if sum(total_sky_cover) == 0:
        location.source += " (0% cloud cover)"
    elif len(set(total_sky_cover)) == 1:
        location.source += f" ({total_sky_cover[0] / 100:.0%} constant cloud cover)"
    else:
        location.source += f" ({total_sky_cover.mean() / 100:.0%} average cloud cover)"

    # create pvlib location
    pv_location = pvlib_location(
        latitude=location.latitude,
        longitude=location.longitude,
        tz=location.time_zone,
        altitude=location.elevation,
        name=location.source,
    )

    # calculate the solar radiation
    solar_position = pv_location.get_solarposition(datetimes)
    dni_extra = get_extra_radiation(datetimes)
    transmittance = (
        ((100.0 - total_sky_cover) / 100.0) * 0.75
    )  # estimate for transmittance of clouds. Assuming 25% reduction if passing through clouds.
    irrad_s = campbell_norman(
        solar_position["apparent_zenith"], transmittance, dni_extra=dni_extra
    )
    irrad_s = irrad_s.fillna(0)

    # construct dataframe
    metadata_str = " | ".join(
        [
            f"{k}: {v}"
            for k, v in {
                "time-zone": location.time_zone,
                "latitude": location.latitude,
                "longitude": location.longitude,
                "elevation": location.elevation,
                "source": location.source,
            }.items()
        ]
    )
    df = pd.DataFrame(
        {
            ("Direct Normal Radiation", "W/m2", metadata_str): irrad_s["dni"],
            ("Diffuse Horizontal Radiation", "W/m2", metadata_str): irrad_s["dhi"],
            ("Global Horizontal Radiation", "W/m2", metadata_str): irrad_s["ghi"],
            ("Total Sky Cover", "fraction", metadata_str): total_sky_cover,
        }
    )

    return df


def radiation_at_height(
    reference_value: float,
    target_height: float,
    reference_height: float,
    lapse_rate: float = 0.08,
) -> float:
    """Calculate the radiation at a given height, given a reference
    radiation and height.

    References:
        Armel Oumbe, Lucien Wald. A parameterisation of vertical profile of
        solar irradiance for correcting solar fluxes for changes in terrain
        elevation. Earth Observation and Water Cycle Science Conference, Nov
        2009, Frascati, Italy. pp.S05.

    Args:
        reference_value (float):
            The radiation at the reference height.
        target_height (float):
            The height at which the radiation is required, in m.
        reference_height (float, optional):
            The height at which the reference radiation was measured.
        lapse_rate (float, optional):
            The lapse rate of the atmosphere. Defaults to 0.08.

    Returns:
        float:
            The radiation at the given height.

    """
    # todo - add sensible limits in here for lapse rate, reference height and target height

    lapse_rate_per_m = lapse_rate * reference_value / 1000
    increase = lapse_rate_per_m * (target_height - reference_height)
    return reference_value + increase


def estimate_net_zero_carbon_pv_area_requirement(
    building_footprint_area: float, pv_yield_per_m2: float
) -> float:
    """Estimate the area of PV required according to the Net Zero Carbon Building Standard.
    The target states that 40kWh/m2 footprint area of PV is required.

    Args:
        building_footprint_area (float):
            The area of the building footprint in m2.
        pv_yield_per_m2 (float):
            The yield of the PV system in kWh/m2/year.

    Returns:
        float:
            The area of PV required in m2.

    """
    # todo - add sensible limits in here for pv yield per m2
    # todo - add sensible limits in here for building footprint area

    # calculate the area of PV required
    pv_area = (building_footprint_area * 40) / pv_yield_per_m2

    return pv_area
