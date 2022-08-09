import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.ladybug_extension.datacollection.from_series import (
    from_series,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.monthlycollection.to_hourly import (
    to_hourly,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series


def ground_temperature_at_depth(
    epw: EPW, depth: float, soil_diffusivity: float = 0.31e-6
) -> HourlyContinuousCollection:
    """Construct annual hourly ground temperature at given depth.

    Args:
        epw (EPW):
            A ladybug EPW object.
        depth (float):
            The depth at which the ground temperature will be calculated.
        soil_diffusivity (float, optional):
            Soil diffusivity value. Common values available from the table below.
            Defaults to 0.31e-6.

    Example:
        Below is a table from https://dx.doi.org/10.3390%2Fs16030306 detailing suitable diffusivity
        values for different soil types:
        +---------------------------+-----------------+-------------+--------------------+
        | RockType                  |     Thermal     | Volumetric  |      Thermal       |
        |                           |   Conductivity  |    Heat     |     Diffusivity    |
        |                           |      (W/mK)     |  Capacity   |    (*10^6 m2/s)    |
        |                           | Min | Typ | Max |  (MJ/m3K)   | Min  | Typ  | Max  |
        +---------------------------+-----+-----+-----|-------------+---+----------------+
        | Basalt                    | 1.3 | 1.7 | 2.3 |     2.6     | 0.5  | 0.65 | 0.88 |
        | Greenstone                | 2.0 | 2.6 | 2.9 |     2.9     | 0.69 | 0.90 | 1.00 |
        | Gabbro                    | 1.7 | 1.9 | 2.5 |     2.6     | 0.65 | 0.73 | 0.96 |
        | Granite                   | 2.1 | 3.4 | 4.1 |     3.0     | 0.70 | 1.13 | 1.37 |
        | Peridotite                | 3.8 | 4,0 | 5.3 |     2.7     | 1.41 | 1.48 | 1.96 |
        | Gneiss                    | 1.9 | 2.9 | 4.0 |     2.4     | 0.79 | 1.21 | 1.67 |
        | Marble                    | 1.3 | 2.1 | 3.1 |     2.0     | 0.65 | 1.05 | 1.55 |
        | Mica schist               | 1.5 | 2,0 | 3.1 |     2.2     | 0.68 | 0.91 | 1.41 |
        | Shale sedimentary         | 1.5 | 2.1 | 2.1 |     2.5     | 0.60 | 0.84 | 0.84 |
        | Limestone                 | 2.5 | 2.8 | 4.0 |     2.4     | 1.04 | 1.17 | 1.67 |
        | Loam                      | 1.5 | 2.1 | 3.5 |     2.3     | 0.65 | 0.91 | 1.52 |
        | Quartzite                 | 3.6 | 6,0 | 6.6 |     2.2     | 1.64 | 2.73 | 3.00 |
        | Salt                      | 5.3 | 5.4 | 6.4 |     1.2     | 4.42 | 4.50 | 5.33 |
        | Sandstone                 | 1.3 | 2.3 | 5.1 |     2.8     | 0.46 | 0.82 | 1.82 |
        | Siltstones and argillites | 1.1 | 2.2 | 3.5 |     2.4     | 0.46 | 0.92 | 1.46 |
        | Dry gravel                | 0.4 | 0.4 | 0.5 |     1.6     | 0.25 | 0.25 | 0.31 |
        | Water saturated gravel    | 1.8 | 1.8 | 1.8 |     2.4     | 0.75 | 0.75 | 0.75 |
        | Dry sand                  | 0.3 | 0.4 | 0.55|     1.6     | 0.19 | 0.25 | 0.34 |
        | Water saturated sand      | 1.7 | 2.4 | 5.0 |     2.9     | 0.59 | 0.83 | 1.72 |
        | Dry clay/silt             | 0.4 | 0.5 | 1.0 |     1.6     | 0.25 | 0.31 | 0.62 |
        | Water saturated clay/silt | 0.9 | 1.7 | 2.3 |     3.4     | 0.26 | 0.5  | 0.68 |
        | Peat                      | 0.2 | 0.4 | 0.7 |     3.8     | 0.05 | 0.10 | 0.18 |
        +---------------------------+-----+-----+-----+-------------+------+------+------+

    Returns:
        HourlyContinuousCollection:
            A data collection containing ground temperature values.
    """

    try:
        return to_hourly(epw.monthly_ground_temperature[depth])
    except (KeyError, ValueError):
        dbt = to_series(epw.dry_bulb_temperature)

        dbt_range = dbt.max() - dbt.min()
        dbt_mean = dbt.mean()
        day_count = len(dbt.index.dayofyear.unique())

        coldest_day = dbt.resample("D").min().values.argmin()
        days_since_coldest_day = []
        for i in range(day_count):
            if i <= coldest_day:
                days_since_coldest_day.append(day_count - (coldest_day - i))
            else:
                days_since_coldest_day.append(i - coldest_day)

        annual_profile = 2 * np.pi / day_count
        annual_profile_factored = np.sqrt(2 * soil_diffusivity * 86400 / annual_profile)
        gnd_temp_daily = (
            pd.Series(
                [
                    dbt_mean
                    - (dbt_range / 2)
                    * np.exp(-depth / annual_profile_factored)
                    * np.cos((annual_profile * i) - (depth / annual_profile_factored))
                    for i in days_since_coldest_day
                ],
                index=dbt.resample("D").min().index,
            )
            .resample("60T")
            .mean()
            .interpolate()
            .values
        )
        gnd_temp_hourly = list(gnd_temp_daily) + list(gnd_temp_daily[0:23])
        gnd_temp_hourly = pd.Series(
            gnd_temp_hourly, index=dbt.index, name="Ground Temperature (C)"
        )

        return from_series(gnd_temp_hourly)
