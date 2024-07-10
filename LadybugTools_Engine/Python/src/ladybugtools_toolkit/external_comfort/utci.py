"""Methods for calculating UTCI."""
# pylint: disable=C0302

# pylint: disable=E0401
from datetime import datetime
import warnings
from calendar import month_abbr
from concurrent.futures import ProcessPoolExecutor

# pylint: enable=E0401

import numpy as np
import numpy.typing as npt
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import (
    UniversalThermalClimateIndex as LB_UniversalThermalClimateIndex,
)
from ladybug.epw import EPW
from ladybug_comfort.collection.solarcal import OutdoorSolarCal
from ladybug_comfort.collection.utci import UTCI
from scipy.interpolate import interp1d, interp2d
from tqdm import tqdm

from ..bhom.analytics import bhom_analytics
from ..categorical.categories import UTCI_DEFAULT_CATEGORIES, CategoricalComfort
from ..helpers import evaporative_cooling_effect, month_hour_binned_series
from ..ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)


def _saturated_vapor_pressure_hpa(dry_bulb_temperature: np.ndarray):
    """Calculate saturated vapor pressure (hPa) at temperature (C).

    Args:
        dry_bulb_temperature (np.ndarray):
            Dry bulb temperature [C]

    Returns:
        np.ndarray:
            Saturated vapor pressure [hPa]

    Note:
        This equation of saturation vapor pressure is specific to the UTCI model.

    """
    dry_bulb_temperature = np.atleast_1d(dry_bulb_temperature)
    g = (
        -2836.5744,
        -6028.076559,
        19.54263612,
        -0.02737830188,
        0.000016261698,
        7.0229056e-10,
        -1.8680009e-13,
    )
    tk = dry_bulb_temperature + 273.15  # air temp in K
    es = 2.7150305 * np.log(tk)
    for i, x in enumerate(g):
        es += x * (tk ** (i - 2))

    return np.exp(es) * 0.01


def _utci_ndarray(
    air_temperature: np.ndarray,
    mean_radiant_temperature: np.ndarray,
    wind_speed: np.ndarray,
    relative_humidity: np.ndarray,
) -> np.ndarray:
    """This method is a vectorised version of the universal_thermal_climate_index method defined in ladybug-tools
    https://github.com/ladybug-tools/ladybug-comfort/blob/master/ladybug_comfort/utci.py

    Args:
        air_temperature (np.ndarray):
            Air temperature [C]
        mean_radiant_temperature (np.ndarray):
            Mean radiant temperature [C]
        wind_speed (np.ndarray):
            Wind speed 10 m above ground level [m/s]
        relative_humidity (np.ndarray):
            Relative humidity [%]

    Returns:
        np.ndarray:
            The Universal Thermal Climate Index (UTCI) for the input conditions as approximated by a 4-D polynomial
    """

    g = (
        -2836.5744,
        -6028.076559,
        19.54263612,
        -0.02737830188,
        0.000016261698,
        7.0229056e-10,
        -1.8680009e-13,
    )
    tk = air_temperature + 273.15  # air temp in K
    es = 2.7150305 * np.log(tk)
    for i, x in enumerate(g):
        es = es + (x * (tk ** (i - 2)))
    es = np.exp(es) * 0.01
    eh_pa = es * (relative_humidity / 100.0)  # partial vapor pressure
    pa_pr = eh_pa / 10.0  # convert vapour pressure to kPa
    d_tr = (
        mean_radiant_temperature - air_temperature
    )  # difference between radiant and air temperature

    utci_approx = (
        air_temperature
        + 0.607562052
        + -0.0227712343 * air_temperature
        + 8.06470249e-4 * air_temperature * air_temperature
        + -1.54271372e-4 * air_temperature * air_temperature * air_temperature
        + -3.24651735e-6
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        + 7.32602852e-8
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        + 1.35959073e-9
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        + -2.25836520 * wind_speed
        + 0.0880326035 * air_temperature * wind_speed
        + 0.00216844454 * air_temperature * air_temperature * wind_speed
        + -1.53347087e-5
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        + -5.72983704e-7
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        + -2.55090145e-9
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        + -0.751269505 * wind_speed * wind_speed
        + -0.00408350271 * air_temperature * wind_speed * wind_speed
        + -5.21670675e-5 * air_temperature * air_temperature * wind_speed * wind_speed
        + 1.94544667e-6
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        + 1.14099531e-8
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        + 0.158137256 * wind_speed * wind_speed * wind_speed
        + -6.57263143e-5 * air_temperature * wind_speed * wind_speed * wind_speed
        + 2.22697524e-7
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        * wind_speed
        + -4.16117031e-8
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        * wind_speed
        + -0.0127762753 * wind_speed * wind_speed * wind_speed * wind_speed
        + 9.66891875e-6
        * air_temperature
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        + 2.52785852e-9
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        + 4.56306672e-4 * wind_speed * wind_speed * wind_speed * wind_speed * wind_speed
        + -1.74202546e-7
        * air_temperature
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        + -5.91491269e-6
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        + 0.398374029 * d_tr
        + 1.83945314e-4 * air_temperature * d_tr
        + -1.73754510e-4 * air_temperature * air_temperature * d_tr
        + -7.60781159e-7 * air_temperature * air_temperature * air_temperature * d_tr
        + 3.77830287e-8
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * d_tr
        + 5.43079673e-10
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * d_tr
        + -0.0200518269 * wind_speed * d_tr
        + 8.92859837e-4 * air_temperature * wind_speed * d_tr
        + 3.45433048e-6 * air_temperature * air_temperature * wind_speed * d_tr
        + -3.77925774e-7
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        * d_tr
        + -1.69699377e-9
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        * d_tr
        + 1.69992415e-4 * wind_speed * wind_speed * d_tr
        + -4.99204314e-5 * air_temperature * wind_speed * wind_speed * d_tr
        + 2.47417178e-7
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        * d_tr
        + 1.07596466e-8
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        * d_tr
        + 8.49242932e-5 * wind_speed * wind_speed * wind_speed * d_tr
        + 1.35191328e-6 * air_temperature * wind_speed * wind_speed * wind_speed * d_tr
        + -6.21531254e-9
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        * wind_speed
        * d_tr
        + -4.99410301e-6 * wind_speed * wind_speed * wind_speed * wind_speed * d_tr
        + -1.89489258e-8
        * air_temperature
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        * d_tr
        + 8.15300114e-8
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        * d_tr
        + 7.55043090e-4 * d_tr * d_tr
        + -5.65095215e-5 * air_temperature * d_tr * d_tr
        + -4.52166564e-7 * air_temperature * air_temperature * d_tr * d_tr
        + 2.46688878e-8
        * air_temperature
        * air_temperature
        * air_temperature
        * d_tr
        * d_tr
        + 2.42674348e-10
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * d_tr
        * d_tr
        + 1.54547250e-4 * wind_speed * d_tr * d_tr
        + 5.24110970e-6 * air_temperature * wind_speed * d_tr * d_tr
        + -8.75874982e-8 * air_temperature * air_temperature * wind_speed * d_tr * d_tr
        + -1.50743064e-9
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        * d_tr
        * d_tr
        + -1.56236307e-5 * wind_speed * wind_speed * d_tr * d_tr
        + -1.33895614e-7 * air_temperature * wind_speed * wind_speed * d_tr * d_tr
        + 2.49709824e-9
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        * d_tr
        * d_tr
        + 6.51711721e-7 * wind_speed * wind_speed * wind_speed * d_tr * d_tr
        + 1.94960053e-9
        * air_temperature
        * wind_speed
        * wind_speed
        * wind_speed
        * d_tr
        * d_tr
        + -1.00361113e-8
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        * d_tr
        * d_tr
        + -1.21206673e-5 * d_tr * d_tr * d_tr
        + -2.18203660e-7 * air_temperature * d_tr * d_tr * d_tr
        + 7.51269482e-9 * air_temperature * air_temperature * d_tr * d_tr * d_tr
        + 9.79063848e-11
        * air_temperature
        * air_temperature
        * air_temperature
        * d_tr
        * d_tr
        * d_tr
        + 1.25006734e-6 * wind_speed * d_tr * d_tr * d_tr
        + -1.81584736e-9 * air_temperature * wind_speed * d_tr * d_tr * d_tr
        + -3.52197671e-10
        * air_temperature
        * air_temperature
        * wind_speed
        * d_tr
        * d_tr
        * d_tr
        + -3.36514630e-8 * wind_speed * wind_speed * d_tr * d_tr * d_tr
        + 1.35908359e-10
        * air_temperature
        * wind_speed
        * wind_speed
        * d_tr
        * d_tr
        * d_tr
        + 4.17032620e-10 * wind_speed * wind_speed * wind_speed * d_tr * d_tr * d_tr
        + -1.30369025e-9 * d_tr * d_tr * d_tr * d_tr
        + 4.13908461e-10 * air_temperature * d_tr * d_tr * d_tr * d_tr
        + 9.22652254e-12 * air_temperature * air_temperature * d_tr * d_tr * d_tr * d_tr
        + -5.08220384e-9 * wind_speed * d_tr * d_tr * d_tr * d_tr
        + -2.24730961e-11 * air_temperature * wind_speed * d_tr * d_tr * d_tr * d_tr
        + 1.17139133e-10 * wind_speed * wind_speed * d_tr * d_tr * d_tr * d_tr
        + 6.62154879e-10 * d_tr * d_tr * d_tr * d_tr * d_tr
        + 4.03863260e-13 * air_temperature * d_tr * d_tr * d_tr * d_tr * d_tr
        + 1.95087203e-12 * wind_speed * d_tr * d_tr * d_tr * d_tr * d_tr
        + -4.73602469e-12 * d_tr * d_tr * d_tr * d_tr * d_tr * d_tr
        + 5.12733497 * pa_pr
        + -0.312788561 * air_temperature * pa_pr
        + -0.0196701861 * air_temperature * air_temperature * pa_pr
        + 9.99690870e-4 * air_temperature * air_temperature * air_temperature * pa_pr
        + 9.51738512e-6
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * pa_pr
        + -4.66426341e-7
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * pa_pr
        + 0.548050612 * wind_speed * pa_pr
        + -0.00330552823 * air_temperature * wind_speed * pa_pr
        + -0.00164119440 * air_temperature * air_temperature * wind_speed * pa_pr
        + -5.16670694e-6
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        * pa_pr
        + 9.52692432e-7
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        * pa_pr
        + -0.0429223622 * wind_speed * wind_speed * pa_pr
        + 0.00500845667 * air_temperature * wind_speed * wind_speed * pa_pr
        + 1.00601257e-6
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        * pa_pr
        + -1.81748644e-6
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        * pa_pr
        + -1.25813502e-3 * wind_speed * wind_speed * wind_speed * pa_pr
        + -1.79330391e-4
        * air_temperature
        * wind_speed
        * wind_speed
        * wind_speed
        * pa_pr
        + 2.34994441e-6
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        * wind_speed
        * pa_pr
        + 1.29735808e-4 * wind_speed * wind_speed * wind_speed * wind_speed * pa_pr
        + 1.29064870e-6
        * air_temperature
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        * pa_pr
        + -2.28558686e-6
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        * pa_pr
        + -0.0369476348 * d_tr * pa_pr
        + 0.00162325322 * air_temperature * d_tr * pa_pr
        + -3.14279680e-5 * air_temperature * air_temperature * d_tr * pa_pr
        + 2.59835559e-6
        * air_temperature
        * air_temperature
        * air_temperature
        * d_tr
        * pa_pr
        + -4.77136523e-8
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * d_tr
        * pa_pr
        + 8.64203390e-3 * wind_speed * d_tr * pa_pr
        + -6.87405181e-4 * air_temperature * wind_speed * d_tr * pa_pr
        + -9.13863872e-6 * air_temperature * air_temperature * wind_speed * d_tr * pa_pr
        + 5.15916806e-7
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        * d_tr
        * pa_pr
        + -3.59217476e-5 * wind_speed * wind_speed * d_tr * pa_pr
        + 3.28696511e-5 * air_temperature * wind_speed * wind_speed * d_tr * pa_pr
        + -7.10542454e-7
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        * d_tr
        * pa_pr
        + -1.24382300e-5 * wind_speed * wind_speed * wind_speed * d_tr * pa_pr
        + -7.38584400e-9
        * air_temperature
        * wind_speed
        * wind_speed
        * wind_speed
        * d_tr
        * pa_pr
        + 2.20609296e-7
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        * d_tr
        * pa_pr
        + -7.32469180e-4 * d_tr * d_tr * pa_pr
        + -1.87381964e-5 * air_temperature * d_tr * d_tr * pa_pr
        + 4.80925239e-6 * air_temperature * air_temperature * d_tr * d_tr * pa_pr
        + -8.75492040e-8
        * air_temperature
        * air_temperature
        * air_temperature
        * d_tr
        * d_tr
        * pa_pr
        + 2.77862930e-5 * wind_speed * d_tr * d_tr * pa_pr
        + -5.06004592e-6 * air_temperature * wind_speed * d_tr * d_tr * pa_pr
        + 1.14325367e-7
        * air_temperature
        * air_temperature
        * wind_speed
        * d_tr
        * d_tr
        * pa_pr
        + 2.53016723e-6 * wind_speed * wind_speed * d_tr * d_tr * pa_pr
        + -1.72857035e-8
        * air_temperature
        * wind_speed
        * wind_speed
        * d_tr
        * d_tr
        * pa_pr
        + -3.95079398e-8 * wind_speed * wind_speed * wind_speed * d_tr * d_tr * pa_pr
        + -3.59413173e-7 * d_tr * d_tr * d_tr * pa_pr
        + 7.04388046e-7 * air_temperature * d_tr * d_tr * d_tr * pa_pr
        + -1.89309167e-8
        * air_temperature
        * air_temperature
        * d_tr
        * d_tr
        * d_tr
        * pa_pr
        + -4.79768731e-7 * wind_speed * d_tr * d_tr * d_tr * pa_pr
        + 7.96079978e-9 * air_temperature * wind_speed * d_tr * d_tr * d_tr * pa_pr
        + 1.62897058e-9 * wind_speed * wind_speed * d_tr * d_tr * d_tr * pa_pr
        + 3.94367674e-8 * d_tr * d_tr * d_tr * d_tr * pa_pr
        + -1.18566247e-9 * air_temperature * d_tr * d_tr * d_tr * d_tr * pa_pr
        + 3.34678041e-10 * wind_speed * d_tr * d_tr * d_tr * d_tr * pa_pr
        + -1.15606447e-10 * d_tr * d_tr * d_tr * d_tr * d_tr * pa_pr
        + -2.80626406 * pa_pr * pa_pr
        + 0.548712484 * air_temperature * pa_pr * pa_pr
        + -0.00399428410 * air_temperature * air_temperature * pa_pr * pa_pr
        + -9.54009191e-4
        * air_temperature
        * air_temperature
        * air_temperature
        * pa_pr
        * pa_pr
        + 1.93090978e-5
        * air_temperature
        * air_temperature
        * air_temperature
        * air_temperature
        * pa_pr
        * pa_pr
        + -0.308806365 * wind_speed * pa_pr * pa_pr
        + 0.0116952364 * air_temperature * wind_speed * pa_pr * pa_pr
        + 4.95271903e-4 * air_temperature * air_temperature * wind_speed * pa_pr * pa_pr
        + -1.90710882e-5
        * air_temperature
        * air_temperature
        * air_temperature
        * wind_speed
        * pa_pr
        * pa_pr
        + 0.00210787756 * wind_speed * wind_speed * pa_pr * pa_pr
        + -6.98445738e-4 * air_temperature * wind_speed * wind_speed * pa_pr * pa_pr
        + 2.30109073e-5
        * air_temperature
        * air_temperature
        * wind_speed
        * wind_speed
        * pa_pr
        * pa_pr
        + 4.17856590e-4 * wind_speed * wind_speed * wind_speed * pa_pr * pa_pr
        + -1.27043871e-5
        * air_temperature
        * wind_speed
        * wind_speed
        * wind_speed
        * pa_pr
        * pa_pr
        + -3.04620472e-6
        * wind_speed
        * wind_speed
        * wind_speed
        * wind_speed
        * pa_pr
        * pa_pr
        + 0.0514507424 * d_tr * pa_pr * pa_pr
        + -0.00432510997 * air_temperature * d_tr * pa_pr * pa_pr
        + 8.99281156e-5 * air_temperature * air_temperature * d_tr * pa_pr * pa_pr
        + -7.14663943e-7
        * air_temperature
        * air_temperature
        * air_temperature
        * d_tr
        * pa_pr
        * pa_pr
        + -2.66016305e-4 * wind_speed * d_tr * pa_pr * pa_pr
        + 2.63789586e-4 * air_temperature * wind_speed * d_tr * pa_pr * pa_pr
        + -7.01199003e-6
        * air_temperature
        * air_temperature
        * wind_speed
        * d_tr
        * pa_pr
        * pa_pr
        + -1.06823306e-4 * wind_speed * wind_speed * d_tr * pa_pr * pa_pr
        + 3.61341136e-6
        * air_temperature
        * wind_speed
        * wind_speed
        * d_tr
        * pa_pr
        * pa_pr
        + 2.29748967e-7 * wind_speed * wind_speed * wind_speed * d_tr * pa_pr * pa_pr
        + 3.04788893e-4 * d_tr * d_tr * pa_pr * pa_pr
        + -6.42070836e-5 * air_temperature * d_tr * d_tr * pa_pr * pa_pr
        + 1.16257971e-6
        * air_temperature
        * air_temperature
        * d_tr
        * d_tr
        * pa_pr
        * pa_pr
        + 7.68023384e-6 * wind_speed * d_tr * d_tr * pa_pr * pa_pr
        + -5.47446896e-7 * air_temperature * wind_speed * d_tr * d_tr * pa_pr * pa_pr
        + -3.59937910e-8 * wind_speed * wind_speed * d_tr * d_tr * pa_pr * pa_pr
        + -4.36497725e-6 * d_tr * d_tr * d_tr * pa_pr * pa_pr
        + 1.68737969e-7 * air_temperature * d_tr * d_tr * d_tr * pa_pr * pa_pr
        + 2.67489271e-8 * wind_speed * d_tr * d_tr * d_tr * pa_pr * pa_pr
        + 3.23926897e-9 * d_tr * d_tr * d_tr * d_tr * pa_pr * pa_pr
        + -0.0353874123 * pa_pr * pa_pr * pa_pr
        + -0.221201190 * air_temperature * pa_pr * pa_pr * pa_pr
        + 0.0155126038 * air_temperature * air_temperature * pa_pr * pa_pr * pa_pr
        + -2.63917279e-4
        * air_temperature
        * air_temperature
        * air_temperature
        * pa_pr
        * pa_pr
        * pa_pr
        + 0.0453433455 * wind_speed * pa_pr * pa_pr * pa_pr
        + -0.00432943862 * air_temperature * wind_speed * pa_pr * pa_pr * pa_pr
        + 1.45389826e-4
        * air_temperature
        * air_temperature
        * wind_speed
        * pa_pr
        * pa_pr
        * pa_pr
        + 2.17508610e-4 * wind_speed * wind_speed * pa_pr * pa_pr * pa_pr
        + -6.66724702e-5
        * air_temperature
        * wind_speed
        * wind_speed
        * pa_pr
        * pa_pr
        * pa_pr
        + 3.33217140e-5 * wind_speed * wind_speed * wind_speed * pa_pr * pa_pr * pa_pr
        + -0.00226921615 * d_tr * pa_pr * pa_pr * pa_pr
        + 3.80261982e-4 * air_temperature * d_tr * pa_pr * pa_pr * pa_pr
        + -5.45314314e-9
        * air_temperature
        * air_temperature
        * d_tr
        * pa_pr
        * pa_pr
        * pa_pr
        + -7.96355448e-4 * wind_speed * d_tr * pa_pr * pa_pr * pa_pr
        + 2.53458034e-5 * air_temperature * wind_speed * d_tr * pa_pr * pa_pr * pa_pr
        + -6.31223658e-6 * wind_speed * wind_speed * d_tr * pa_pr * pa_pr * pa_pr
        + 3.02122035e-4 * d_tr * d_tr * pa_pr * pa_pr * pa_pr
        + -4.77403547e-6 * air_temperature * d_tr * d_tr * pa_pr * pa_pr * pa_pr
        + 1.73825715e-6 * wind_speed * d_tr * d_tr * pa_pr * pa_pr * pa_pr
        + -4.09087898e-7 * d_tr * d_tr * d_tr * pa_pr * pa_pr * pa_pr
        + 0.614155345 * pa_pr * pa_pr * pa_pr * pa_pr
        + -0.0616755931 * air_temperature * pa_pr * pa_pr * pa_pr * pa_pr
        + 0.00133374846
        * air_temperature
        * air_temperature
        * pa_pr
        * pa_pr
        * pa_pr
        * pa_pr
        + 0.00355375387 * wind_speed * pa_pr * pa_pr * pa_pr * pa_pr
        + -5.13027851e-4 * air_temperature * wind_speed * pa_pr * pa_pr * pa_pr * pa_pr
        + 1.02449757e-4 * wind_speed * wind_speed * pa_pr * pa_pr * pa_pr * pa_pr
        + -0.00148526421 * d_tr * pa_pr * pa_pr * pa_pr * pa_pr
        + -4.11469183e-5 * air_temperature * d_tr * pa_pr * pa_pr * pa_pr * pa_pr
        + -6.80434415e-6 * wind_speed * d_tr * pa_pr * pa_pr * pa_pr * pa_pr
        + -9.77675906e-6 * d_tr * d_tr * pa_pr * pa_pr * pa_pr * pa_pr
        + 0.0882773108 * pa_pr * pa_pr * pa_pr * pa_pr * pa_pr
        + -0.00301859306 * air_temperature * pa_pr * pa_pr * pa_pr * pa_pr * pa_pr
        + 0.00104452989 * wind_speed * pa_pr * pa_pr * pa_pr * pa_pr * pa_pr
        + 2.47090539e-4 * d_tr * pa_pr * pa_pr * pa_pr * pa_pr * pa_pr
        + 0.00148348065 * pa_pr * pa_pr * pa_pr * pa_pr * pa_pr * pa_pr
    )

    return utci_approx


def utci_metadata(utci_collection: HourlyContinuousCollection, comfort_lower: float = 9, comfort_higher: float = 26, use_start_hour: int=7, use_end_hour: int=23) -> dict:
    """Returns a dictionary of useful metadata for the given collection dependant on the given comfortable range.
    
    Args:
        utci_collection (HourlyContinuousCollection):
            utci headered ladybug hourly collection
            
        comfort_lower (float):
            lower value for the comfortable temperature range, where temperatures exclusively below this are too cold.
            
        comfort_higher (float):
            higher value for the comfortable temperature range, where temperatures above and equal to this are too hot.
            
        use_start_hour (int):
            start hour to filter usage time, inclusive
        
        use_end_hour (int):
            end hour to filter usage time, exclusive
            
    Returns:
        dict:
            dictionary containing comfortable, hot and cold ratios, structured as follows:
            {
                'comfortable_ratio': ratio_of_comfortable_hours,
                'hot_ratio': ratio_of_hot_hours,
                'cold_ratio': ratio_of_cold_hours,
                'daytime_comfortable': daytime_comfortable_ratio,
                'daytime_hot': daytime_hot_ratio,
                'daytime_cold': daytime_cold_ratio
            }
    """
    if not isinstance(utci_collection.header.data_type, LB_UniversalThermalClimateIndex):
        raise ValueError("Input collection is not a UTCI collection.")

    if not comfort_lower < comfort_higher:
        raise ValueError(f"The lower comfort temperature {comfort_lower}, must be less than the higher comfort temperature {comfort_higher}.")

    series = collection_to_series(utci_collection)
    
    daytime = series.loc[(series.index.hour >= use_start_hour) & (series.index.hour < use_end_hour)]

    comfortable_ratio = ((series >= comfort_lower) & (series < comfort_higher)).sum() / len(series)
    hot_ratio = (series >= comfort_higher).sum() / len(series)
    cold_ratio = (series < comfort_lower).sum() / len(series)

    day_comfortable = ((daytime >= comfort_lower) & (daytime < comfort_higher)).sum() / len(daytime)
    day_hot = (daytime >= comfort_higher).sum() / len(daytime)
    day_cold = (daytime < comfort_lower).sum() / len(daytime)

    return {
        "comfortable_ratio": comfortable_ratio,
        "hot_ratio": hot_ratio,
        "cold_ratio": cold_ratio,
        "daytime_comfortable": day_comfortable,
        "daytime_hot": day_hot,
        "daytime_cold": day_cold
        }


def _utci_collection(
    air_temperature: HourlyContinuousCollection,
    mean_radiant_temperature: HourlyContinuousCollection,
    wind_speed: HourlyContinuousCollection,
    relative_humidity: HourlyContinuousCollection,
) -> HourlyContinuousCollection:
    """Calculate UTCI for a Ladybug HourlyContinuousCollection inputs.

    Args:
        air_temperature (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection describing air temperature in C.
        mean_radiant_temperature (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection describing mean radiant temperature in C.
        wind_speed (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection describing wind speed at 10m above ground in m/s.
        relative_humidity (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection describing relative humidity [0-100].

    Returns:
        HourlyContinuousCollection:
            The calculated UTCI for the given inputs.
    """

    return UTCI(
        air_temperature=air_temperature,
        rel_humidity=relative_humidity,
        rad_temperature=mean_radiant_temperature,
        wind_speed=wind_speed,
    ).universal_thermal_climate_index


@bhom_analytics()
def utci_parallel(
    ta: np.ndarray, tr: np.ndarray, vel: np.ndarray, rh: np.ndarray
) -> np.ndarray:
    """Calculate UTCI a bit faster!

    Args:
        ta (np.ndarray):
            Air temperature [C]
        tr (np.ndarray):
            Mean radiant temperature [C]
        vel (np.ndarray):
            Wind speed 10m above ground level [m/s]
        rh (np.ndarray):
            Relative humidity [%]

    Returns:
        np.ndarray:
            UTCI values.
    """

    if ta.shape != tr.shape != vel.shape != rh.shape:
        raise ValueError("Input arrays must be of the same shape.")
    if any(
        [
            len(ta.shape) != 2,
            len(tr.shape) != 2,
            len(vel.shape) != 2,
            len(rh.shape) != 2,
        ]
    ):
        raise ValueError("Input arrays must be of shape (n, m).")

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(_utci_ndarray, ta, tr, vel, rh),
                total=len(ta),
                desc="Calculating UTCI: ",
            )
        )

    return np.concatenate(results).reshape(ta.shape)


@bhom_analytics()
def utci(
    air_temperature: HourlyContinuousCollection
    | pd.DataFrame
    | pd.Series
    | npt.NDArray[np.float64],
    relative_humidity: HourlyContinuousCollection
    | pd.DataFrame
    | pd.Series
    | npt.NDArray[np.float64],
    mean_radiant_temperature: HourlyContinuousCollection
    | pd.DataFrame
    | pd.Series
    | npt.NDArray[np.float64],
    wind_speed: HourlyContinuousCollection
    | pd.DataFrame
    | pd.Series
    | npt.NDArray[np.float64],
) -> HourlyContinuousCollection | pd.DataFrame | npt.NDArray[np.float64]:
    """Return the UTCI for the given inputs.

    Arguments:
        air_temperature: HourlyContinuousCollection | pd.DataFrame | pd.Series | NDArray[np.float64]:
            The air temperature in Celsius.
        relative_humidity: HourlyContinuousCollection | pd.DataFrame | pd.Series | NDArray[np.float64]:
            The relative humidity in percentage.
        mean_radiant_temperature: HourlyContinuousCollection | pd.DataFrame | pd.Series | NDArray[np.float64]:
            The mean radiant temperature in Celsius.
        wind_speed: HourlyContinuousCollection | pd.DataFrame | pd.Series | NDArray[np.float64]:
            The wind speed in m/s.

    Returns:
        HourlyContinuousCollection:
            The calculated UTCI based on the shelter configuration for the given typology.
    """
    _inputs = [
        air_temperature,
        relative_humidity,
        mean_radiant_temperature,
        wind_speed,
    ]

    if all((isinstance(i, HourlyContinuousCollection) for i in _inputs)):
        # print("Calculating UTCI - Ladybug HourlyContinuousCollection")
        return UTCI(
            air_temperature=air_temperature,
            rel_humidity=relative_humidity,
            rad_temperature=mean_radiant_temperature,
            wind_speed=wind_speed,
        ).universal_thermal_climate_index

    if all((isinstance(i, (float, int)) for i in _inputs)):
        # print("Calculating UTCI - float/int")
        return _utci_ndarray(
            air_temperature=air_temperature,
            relative_humidity=relative_humidity,
            mean_radiant_temperature=mean_radiant_temperature,
            wind_speed=np.clip([wind_speed], 0, 17)[0],
        )

    if all((isinstance(i, pd.DataFrame) for i in _inputs)):
        # print("Calculating UTCI - pandas DataFrame")
        return pd.DataFrame(
            _utci_ndarray(
                air_temperature=air_temperature.values,
                relative_humidity=relative_humidity.values,
                mean_radiant_temperature=mean_radiant_temperature.values,
                wind_speed=wind_speed.clip(lower=0, upper=17).values,
            ),
            columns=_inputs[0].columns
            if len(_inputs[0].columns) > 1
            else ["Universal Thermal Climate Index (C)"],
            index=_inputs[0].index,
        )

    if all((isinstance(i, pd.Series) for i in _inputs)):
        # print("Calculating UTCI - pandas Series")
        return pd.Series(
            _utci_ndarray(
                air_temperature=air_temperature,
                relative_humidity=relative_humidity,
                mean_radiant_temperature=mean_radiant_temperature,
                wind_speed=wind_speed.clip(lower=0, upper=17),
            ),
            name="Universal Thermal Climate Index (C)",
            index=_inputs[0].index,
        )

    if all((isinstance(i, (list, tuple)) for i in _inputs)):
        # print("Calculating UTCI - List/Tuple")
        return _utci_ndarray(
            air_temperature=np.array(air_temperature),
            relative_humidity=np.array(relative_humidity),
            mean_radiant_temperature=np.array(mean_radiant_temperature),
            wind_speed=np.clip(np.array(wind_speed), 0, 17),
        )

    try:
        # print("Calculating UTCI - np.array")
        # assume numpy array
        return _utci_ndarray(
            air_temperature=air_temperature,
            relative_humidity=relative_humidity,
            mean_radiant_temperature=mean_radiant_temperature,
            wind_speed=np.clip(wind_speed, 0, 17),
        )
    except Exception as e:
        raise ValueError(
            "No possible means of calculating UTCI from that combination of inputs was found."
        ) from e


@bhom_analytics()
def compare_monthly_utci(
    utci_collections: list[HourlyContinuousCollection],
    utci_categories: CategoricalComfort = UTCI_DEFAULT_CATEGORIES,
    identifiers: tuple[str] = None,
    density: bool = False,
) -> pd.DataFrame:
    """Create a summary of a comparison between a "baseline" UTCI collection,
    and another.

    Args:
        utci_collections (list[HourlyContinuousCollection]):
            A list of UTCI collections to compare.
        utci_categories (Categories, optional):
            A set of categories to use for the comparison.
        identifiers (tuple[str]):
            A tuple of identifiers for the baseline and comparable collections.
        density (bool, optional):
            Return the proportion of time rather than the number of hours.
            Defaults to False.

    Returns:
        pd.DataFrame:
            A table showing the number of hours in each category for each
            collection.
    """

    # ensure each collection given it a UTCI collection
    if len(utci_collections) < 2:
        raise ValueError("At least two UTCI collections must be given to compare them.")

    if any(
        not isinstance(i.header.data_type, LB_UniversalThermalClimateIndex)
        for i in utci_collections
    ):
        raise ValueError("Input collection is not a UTCI collection.")

    if identifiers is None:
        identifiers = range(len(utci_collections))
    else:
        assert len(identifiers) == len(
            utci_collections
        ), "The identifiers given must be the same length as the collections given."

    dd = []
    for i in utci_collections:
        _df = utci_categories.timeseries_summary_monthly(
            collection_to_series(i), density=density
        )
        _df.columns = [utci_categories.interval_from_bin_name(i) for i in _df.columns]
        dd.append(_df)
    df = pd.concat(dd, axis=1, keys=identifiers)
    df = df.reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df = df.rename(
        columns=utci_categories._interval_bin_name  # pylint: disable=protected-access
    )

    return df


@bhom_analytics()
def shade_benefit_category(
    unshaded_utci: HourlyContinuousCollection | pd.Series,
    shaded_utci: HourlyContinuousCollection | pd.Series,
    comfort_limits: tuple[float] = (9, 26),
) -> pd.Series:
    """Determine shade-gap analysis category, indicating where shade is not
    beneificial.

    Args:
        unshaded_utci (HourlyContinuousCollection | pd.Series):
            A dataset containing unshaded UTCI values.
        shaded_utci (HourlyContinuousCollection | pd.Series):
            A dataset containing shaded UTCI values.
        comfort_limits (tuple[float], optional):
            The range within which "comfort" is achieved. Defaults to (9, 26).

    Returns:
        pd.Series:
            A catgorical series indicating shade-benefit.

    """

    # convert to series if not already
    if isinstance(unshaded_utci, HourlyContinuousCollection):
        unshaded_utci = collection_to_series(unshaded_utci)
    if isinstance(shaded_utci, HourlyContinuousCollection):
        shaded_utci = collection_to_series(shaded_utci)

    if len(unshaded_utci) != len(shaded_utci):
        raise ValueError(
            f"Input sizes do not match ({len(unshaded_utci)} != {len(shaded_utci)})"
        )

    if sum(unshaded_utci == shaded_utci) == len(unshaded_utci):
        raise ValueError("Input series are identical.")

    # get limits
    low, high = min(comfort_limits), max(comfort_limits)

    # get distance to comfort (degrees from edge of "comfortable")
    distance_from_comfort_unshaded = abs(
        np.where(
            unshaded_utci < low,
            unshaded_utci - low,
            np.where(unshaded_utci > high, unshaded_utci - high, 0),
        )
    )
    distance_from_comfort_shaded = abs(
        np.where(
            shaded_utci < low,
            shaded_utci - low,
            np.where(shaded_utci > high, shaded_utci - high, 0),
        )
    )

    # get boolean mask where comfortable
    comfortable_unshaded = unshaded_utci.between(low, high)
    comfortable_shaded = shaded_utci.between(low, high)

    # get masks for each category
    comfortable_without_shade = comfortable_unshaded
    comfortable_with_shade = ~comfortable_unshaded & comfortable_shaded
    shade_has_negative_impact = (
        distance_from_comfort_unshaded < distance_from_comfort_shaded
    )
    shade_has_positive_impact = (
        distance_from_comfort_unshaded > distance_from_comfort_shaded
    )

    # construct categorical series
    shade_categories = np.where(
        comfortable_without_shade,
        "Comfortable without shade",
        np.where(
            comfortable_with_shade,
            "Comfortable with shade",
            np.where(
                shade_has_negative_impact,
                "Shade is detrimental",
                np.where(shade_has_positive_impact, "Shade is beneficial", "Undefined"),
            ),
        ),
    )

    return pd.Series(shade_categories, index=unshaded_utci.index)


@bhom_analytics()
def distance_to_comfortable(
    utci_value: int
    | float
    | list
    | tuple
    | pd.Series
    | pd.DataFrame
    | HourlyContinuousCollection,
    comfort_thresholds: tuple[float] = (9, 26),
    distance_to_comfort_band_centroid: bool = True,
) -> int | float | list | tuple | pd.Series | pd.DataFrame | HourlyContinuousCollection:
    """
    Get the distance between the given value/s and the "comfortable" category.

    Args:
        utci_value (int | float | list | tuple | pd.Series | pd.DataFrame | HourlyContinuousCollection):
            A value or set of values representing UTCI temperature.
        comfort_thresholds (list[float], optional):
            The comfortable band of UTCI temperatures. Defaults to [9, 26].
        distance_to_comfort_band_centroid (bool, optional):
            If True, the distance to the centroid of the comfort band is
            plotted. If False, the distance to the edge of the comfort band is
            plotted. Defaults to False.

    Returns:
        int | float | list | tuple | pd.Series | pd.DataFrame | HourlyContinuousCollection:
            The original value/s distance from "comfortable"

    """

    if len(comfort_thresholds) != 2:
        raise ValueError("comfort_thresholds must be a list of length 2.")

    if len(set(comfort_thresholds)) != 2:
        raise ValueError("comfort_thresholds must contain two unique values.")

    if comfort_thresholds[0] > comfort_thresholds[1]:
        warnings.warn("comfort_thresholds are not increasing. Swapping the values.")

    low_limit = min(comfort_thresholds)
    high_limit = max(comfort_thresholds)
    midpoint = np.mean(comfort_thresholds)

    # get raw values as numpy array for ease of processing
    original_value = np.atleast_1d(utci_value)
    name = "Distance to comfortable (C)"

    # calculate distance from "comfort"
    if not distance_to_comfort_band_centroid:
        distance = np.where(
            original_value < low_limit,
            original_value - low_limit,
            np.where(original_value > high_limit, original_value - high_limit, 0),
        )
    else:
        distance = np.where(
            original_value < midpoint,
            -(midpoint - original_value),
            original_value - midpoint,
        )

    if isinstance(utci_value, pd.Series):
        return pd.Series(distance, index=utci_value.index, name=name)

    if isinstance(utci_value, pd.DataFrame):
        return pd.DataFrame(
            distance, index=utci_value.index, columns=utci_value.columns
        )

    if isinstance(utci_value, HourlyContinuousCollection):
        if not isinstance(utci_value.header.data_type, LB_UniversalThermalClimateIndex):
            raise ValueError("Input collection is not a UTCI collection.")
        return collection_from_series(
            pd.Series(
                original_value,
                index=collection_to_series(utci_value).index,
                name=name,
            )
        )

    if isinstance(utci_value, list):
        return distance.tolist()

    if isinstance(utci_value, tuple):
        return tuple(distance.tolist())

    return distance


@bhom_analytics()
def feasible_utci_limits(
    epw: EPW, include_additional_moisture: float = 0, as_dataframe: bool = False
) -> tuple[HourlyContinuousCollection] | pd.DataFrame:
    """Calculate the absolute min/max collections of UTCI based on possible
    shade, wind and moisture conditions.

    Args:
        epw (EPW):
            The EPW object for which limits will be calculated.
        include_additional_moisture (float, optional):
            Include the effect of evaporative cooling on the UTCI limits.
            Default is 0, for no evaporative cooling.
        as_dataframe (bool):
            Return the output as a dataframe with two columns, instread of two
            separate collections.

    Returns:
        tuple[HourlyContinuousCollection] | pd.DataFrame:
            The lowest possible UTCI and highest UTCI temperatures for each
            hour of the year.
    """
    mrt_unshaded = OutdoorSolarCal(
        epw.location,
        epw.direct_normal_radiation,
        epw.diffuse_horizontal_radiation,
        epw.horizontal_infrared_radiation_intensity,
        epw.dry_bulb_temperature,
    ).mean_radiant_temperature
    dbt_evap, rh_evap = np.array(
        [
            evaporative_cooling_effect(
                dry_bulb_temperature=_dbt,
                relative_humidity=_rh,
                evaporative_cooling_effectiveness=include_additional_moisture,
                atmospheric_pressure=_atm,
            )
            for _dbt, _rh, _atm in list(
                zip(
                    *[
                        epw.dry_bulb_temperature,
                        epw.relative_humidity,
                        epw.atmospheric_station_pressure,
                    ]
                )
            )
        ]
    ).T
    dbt_evap = epw.dry_bulb_temperature.get_aligned_collection(dbt_evap)
    rh_evap = epw.relative_humidity.get_aligned_collection(rh_evap)

    dbt_rh_options = (
        [[dbt_evap, rh_evap], [epw.dry_bulb_temperature, epw.relative_humidity]]
        if include_additional_moisture != 0
        else [[epw.dry_bulb_temperature, epw.relative_humidity]]
    )

    utcis = []
    utcis = []
    for _dbt, _rh in dbt_rh_options:
        for _ws in [
            epw.wind_speed,
            epw.wind_speed.get_aligned_collection(0),
            epw.wind_speed * 1.1,
        ]:
            for _mrt in [epw.dry_bulb_temperature, mrt_unshaded]:
                utcis.append(
                    UTCI(
                        air_temperature=_dbt,
                        rad_temperature=_mrt,
                        rel_humidity=_rh,
                        wind_speed=_ws,
                    ).universal_thermal_climate_index,
                )
    df = pd.concat([collection_to_series(i) for i in utcis], axis=1)
    min_utci = collection_from_series(
        df.min(axis=1).rename("Universal Thermal Climate Index (C)")
    )
    max_utci = collection_from_series(
        df.max(axis=1).rename("Universal Thermal Climate Index (C)")
    )

    if as_dataframe:
        return pd.concat(
            [
                collection_to_series(min_utci),
                collection_to_series(max_utci),
            ],
            axis=1,
            keys=["lowest", "highest"],
        )

    return min_utci, max_utci


@bhom_analytics()
def feasible_utci_category_limits(
    epw: EPW,
    include_additional_moisture: float = 0,
    utci_categories: CategoricalComfort = UTCI_DEFAULT_CATEGORIES,
    density: bool = False,
    mask: list[bool] = None,
):
    """Calculate the upper and lower proportional limits of UTCI categories
    based on possible shade, wind and moisture conditions.

    Args:
        epw (EPW):
            The EPW object for which limits will be calculated.
        include_additional_moisture (float, optional):
            Include the effect of evaporative cooling on the UTCI limits.
            Default is 0, for no evaporative cooling.
        utci_categories (Categorical, optional):
            A set of categories to use for the comparison.
        density (bool, optional):
            Return the proportion of time rather than the number of hours.
            Defaults to False.
        mask (list[bool], optional):
            A list of booleans to mask the data. Defaults to None.

    Returns:
        pd.DataFrame:
            A table with monthly time binned UTCI data.
    """

    lims = feasible_utci_limits(
        epw, include_additional_moisture=include_additional_moisture, as_dataframe=True
    )

    if mask is not None:
        lims = lims[mask]
        if lims.index.month.nunique() != 12:
            raise ValueError("Masked data include at least one value per month.")

    df = (
        pd.concat(
            [
                utci_categories.timeseries_summary_monthly(
                    lims.lowest, density=density
                ),
                utci_categories.timeseries_summary_monthly(
                    lims.highest, density=density
                ),
            ],
            axis=1,
            keys=lims.columns,
        )
        .reorder_levels([1, 0], axis=1)
        .sort_index(axis=1, ascending=[True, False])
    )

    df.index = [month_abbr[i] for i in df.index]

    return df


@bhom_analytics()
def month_hour_binned(
    utci_data: pd.Series | HourlyContinuousCollection,
    month_bins: tuple[tuple[int]] = None,
    hour_bins: tuple[tuple[int]] = None,
    month_labels: tuple[str] = None,
    hour_labels: tuple[str] = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """Create a table with monthly hour binned UTCI data.

    Args:
        utci_data (pd.Series | HourlyContinuousCollection):
            A collection of UTCI values.
        month_bins (tuple[list[int]]):
            A list of lists of months to group data into.
        hour_bins (tuple[list[int]]):
            A list of lists of hours to group data into.
        month_labels (list[str], optional):
            A list of labels for the month bins. Defaults to None.
        hour_labels (list[str], optional):
            A list of labels for the hour bins. Defaults to None.
        agg (str, optional):
            The aggregation method to use for the data within each bin.
            Defaults to "mean".

    Returns:
        pd.DataFrame:
            A table with monthly time binned UTCI data.
    """

    # check the utci_data is either a pd.Series or a HourlyContinuousCollection
    if not isinstance(utci_data, (pd.Series, HourlyContinuousCollection)):
        raise TypeError(
            "utci_data must be either a pandas.Series or a ladybug.datacollection.HourlyContinuousCollection"
        )

    if isinstance(utci_data, HourlyContinuousCollection):
        utci_data = collection_to_series(utci_data)

    df = month_hour_binned_series(
        series=utci_data,
        month_bins=month_bins,
        hour_bins=hour_bins,
        month_labels=month_labels,
        hour_labels=hour_labels,
        agg=agg,
    )

    return df


@bhom_analytics()
def met_rate_adjustment(
    utci_collection: HourlyContinuousCollection, met: float
) -> HourlyContinuousCollection:
    """Adjust a UTCI data collection using a target met-rate.

    References:
        This method uses the relationshp between UTCI and MET rate described in
        LINDNER-CENDROWSKA, Katarzyna and BRÖDE, Peter, 2021. The evaluation of
        biothermal conditions for various forms of climatic therapy based on
        UTCI adjusted for activity [online]. 2021. IGiPZ PAN.
        [Accessed 27 October 2022]. Available from:
        http://rcin.org.pl/igipz/Content/194924/PDF/WA51_229409_r2021-t94-no2_G-Polonica-Linder.pdf.

    +---------------------------------------------------------------+----------------------+
    | Activity                                                      | Metabolic rate (MET) |
    +---------------------------------------------------------------+----------------------+
    | Neutral (resting in sitting or standing position)             | 1.1                  |
    +---------------------------------------------------------------+----------------------+
    | Slow walk (on even path, without load, at 3-4 km·h-1)         | 2.3                  |
    | (default for UTCI calculation)                                |                      |
    +---------------------------------------------------------------+----------------------+
    | Fast walk (on even path, without load, at ~5 km·h-1)          | 3.4                  |
    +---------------------------------------------------------------+----------------------+
    | Marching (on even path, without load, at ~5.5 km·h-1)         | 4.0                  |
    +---------------------------------------------------------------+----------------------+
    | Bicycling (for pleasure, on flat terrain, at < 16 km·h-1)     | 4.0                  |
    +---------------------------------------------------------------+----------------------+
    | Nordic walking (for exercise, on flat terrain, at 5-6 km·h-1) | 4.8                  |
    +---------------------------------------------------------------+----------------------+

    Args:
        utci_collection (HourlyContinuousCollection):
            A UTCI data collection.
        met (float, optional):
            The metabolic rate to apply to this data collection.

    Returns:
        HourlyContinuousCollection:
            An adjusted UTCI data collection.
    """

    if met < 1.1:
        raise ValueError(
            "met_rate must be >= 1.1 (representative of a human body at rest)."
        )
    if met > 4.8:
        raise ValueError(
            "met_rate must be <= 4.8 (representative of a exercise at 5-6km·h-1)."
        )

    # data below extracted from https://doi.org/10.7163/GPol.0199 in the format {MET: [UTCI, ΔUTCI]}
    data = {
        4.8: [
            [-50, 71.81581439393939],
            [-46.47239263803681, 71.5127840909091],
            [-42.94478527607362, 71.36126893939394],
            [-38.95705521472392, 70.14914772727273],
            [-34.96932515337423, 67.421875],
            [-30.521472392638035, 64.69460227272728],
            [-26.993865030674847, 61.66429924242425],
            [-23.006134969325153, 58.33096590909091],
            [-17.944785276073617, 52.87642045454545],
            [-12.576687116564415, 48.63399621212122],
            [-6.441717791411044, 43.63399621212122],
            [-1.380368098159508, 40.300662878787875],
            [4.29447852760736, 35.755208333333336],
            [9.815950920245399, 30.906723484848484],
            [15.490797546012274, 26.512784090909093],
            [23.00613496932516, 20.755208333333336],
            [28.06748466257669, 16.967329545454547],
            [30.98159509202455, 15.452178030303031],
            [35.889570552147234, 12.573390151515156],
            [42.63803680981596, 9.846117424242422],
            [50, 6.967329545454547],
        ],
        4.0: [
            [-50, 46.05823863636364],
            [-47.69938650306749, 47.421875],
            [-45.0920245398773, 48.63399621212122],
            [-43.86503067484662, 50.60369318181819],
            [-40.1840490797546, 51.96732954545455],
            [-36.04294478527608, 51.36126893939394],
            [-33.74233128834356, 50.906723484848484],
            [-30.67484662576687, 49.543087121212125],
            [-27.300613496932513, 47.87642045454545],
            [-24.846625766871163, 46.36126893939394],
            [-23.159509202453986, 44.99763257575758],
            [-20.39877300613497, 42.87642045454545],
            [-16.871165644171782, 39.69460227272728],
            [-14.11042944785276, 37.573390151515156],
            [-11.04294478527607, 35.60369318181819],
            [-7.515337423312886, 34.24005681818182],
            [-3.374233128834355, 32.27035984848485],
            [1.0736196319018418, 29.84611742424243],
            [4.141104294478531, 28.02793560606061],
            [7.055214723926383, 25.45217803030303],
            [9.969325153374236, 23.482481060606062],
            [13.036809815950924, 21.664299242424242],
            [16.41104294478528, 19.543087121212125],
            [19.631901840490798, 17.27035984848485],
            [22.852760736196316, 14.846117424242422],
            [26.22699386503068, 12.724905303030305],
            [29.601226993865026, 11.664299242424242],
            [33.28220858895706, 9.846117424242422],
            [36.65644171779141, 8.785511363636367],
            [39.87730061349693, 8.330965909090907],
            [42.63803680981596, 7.421875],
            [46.012269938650306, 6.664299242424242],
            [50, 5.452178030303031],
        ],
        3.4: [
            [-50, 27.724905303030305],
            [-47.239263803680984, 28.785511363636367],
            [-44.47852760736196, 29.543087121212125],
            [-41.717791411042946, 30.45217803030303],
            [-38.34355828220859, 31.05823863636364],
            [-35.2760736196319, 31.05823863636364],
            [-33.43558282208589, 31.361268939393938],
            [-30.98159509202454, 31.815814393939398],
            [-28.220858895705522, 31.967329545454547],
            [-26.380368098159508, 31.05823863636364],
            [-24.846625766871163, 30.300662878787882],
            [-23.006134969325153, 29.694602272727273],
            [-21.165644171779142, 28.482481060606062],
            [-18.558282208588956, 26.967329545454547],
            [-16.411042944785272, 25.755208333333336],
            [-14.11042944785276, 24.39157196969697],
            [-11.65644171779141, 23.02793560606061],
            [-8.742331288343557, 23.179450757575758],
            [-5.828220858895705, 23.330965909090914],
            [-2.4539877300613497, 22.573390151515156],
            [0.6134969325153392, 21.361268939393938],
            [3.374233128834355, 19.846117424242422],
            [6.74846625766871, 17.876420454545453],
            [9.662576687116562, 16.512784090909093],
            [12.883435582822088, 14.846117424242422],
            [16.104294478527606, 14.088541666666664],
            [19.631901840490798, 11.815814393939398],
            [22.085889570552155, 10.300662878787882],
            [24.84662576687117, 8.785511363636367],
            [28.52760736196319, 8.027935606060609],
            [31.44171779141105, 7.1188446969696955],
            [34.20245398773007, 6.058238636363637],
            [37.269938650306756, 5.755208333333336],
            [40.03067484662577, 5.452178030303031],
            [42.94478527607362, 4.846117424242426],
            [46.012269938650306, 4.088541666666668],
            [50, 3.9370265151515156],
        ],
        2.3: [
            [-50, -0.15151515151515227],
            [50, 0],
        ],
        1.1: [
            [-50, -27.87878787878788],
            [-48.15950920245399, -26.96969696969697],
            [-45.858895705521476, -26.21212121212121],
            [-44.6319018404908, -24.09090909090909],
            [-42.484662576687114, -21.96969696969697],
            [-39.57055214723926, -19.848484848484848],
            [-36.04294478527608, -19.242424242424242],
            [-32.20858895705521, -20],
            [-29.447852760736197, -21.363636363636363],
            [-25.460122699386503, -23.03030303030303],
            [-22.54601226993865, -24.848484848484848],
            [-19.32515337423313, -26.666666666666668],
            [-16.257668711656443, -28.484848484848484],
            [-13.34355828220859, -30.151515151515152],
            [-9.509202453987726, -30.90909090909091],
            [-5.981595092024541, -31.060606060606062],
            [-3.067484662576689, -29.393939393939394],
            [-1.380368098159508, -27.272727272727273],
            [0, -24.848484848484848],
            [1.0736196319018418, -21.666666666666668],
            [2.760736196319016, -18.78787878787879],
            [4.141104294478531, -15.909090909090908],
            [6.74846625766871, -13.484848484848484],
            [9.969325153374236, -11.818181818181818],
            [13.190184049079754, -11.363636363636363],
            [16.257668711656436, -9.242424242424242],
            [19.785276073619627, -9.696969696969697],
            [23.00613496932516, -10.303030303030303],
            [26.380368098159508, -9.242424242424242],
            [29.447852760736197, -7.272727272727273],
            [32.515337423312886, -6.363636363636363],
            [35.582822085889575, -5.303030303030301],
            [38.650306748466264, -5.151515151515152],
            [41.25766871165645, -4.3939393939393945],
            [44.018404907975466, -3.787878787878789],
            [46.31901840490798, -3.333333333333332],
            [50, -3.6363636363636367],
        ],
    }

    matrix = []
    for k, v in data.items():
        x, y = np.array(v).T
        f = interp1d(x, y)
        new_x = np.linspace(-50, 50, 1000)
        new_y = f(new_x)
        matrix.append(pd.Series(index=new_x, data=new_y, name=k))
    met_rate, utci_val, utci_delta = (
        pd.concat(matrix, axis=1).unstack().reset_index().values.T
    )

    # create 2d interpolator between [MET, UTCI] and [ΔUTCI]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster = interp2d(met_rate, utci_val, utci_delta)

    # Calculate ΔUTCI
    original_utci = collection_to_series(utci_collection)
    utci_delta = [forecaster(met, i)[0] for i in original_utci.values]

    return collection_from_series(original_utci + utci_delta)
