from concurrent.futures import ProcessPoolExecutor
from typing import Any, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug_comfort.collection.utci import UTCI
from tqdm import tqdm


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


def utci(
    air_temperature: Union[
        HourlyContinuousCollection,
        pd.DataFrame,
        pd.Series,
        npt.NDArray[np.float64],
    ],
    relative_humidity: Union[
        HourlyContinuousCollection,
        pd.DataFrame,
        pd.Series,
        npt.NDArray[np.float64],
    ],
    mean_radiant_temperature: Union[
        HourlyContinuousCollection,
        pd.DataFrame,
        pd.Series,
        npt.NDArray[np.float64],
    ],
    wind_speed: Union[
        HourlyContinuousCollection,
        pd.DataFrame,
        pd.Series,
        npt.NDArray[np.float64],
    ],
) -> Union[HourlyContinuousCollection, pd.DataFrame, npt.NDArray[np.float64]]:
    """Return the UTCI for the given inputs.

    Arguments:
        air_temperature: Union[HourlyContinuousCollection, pd.DataFrame, pd.Series, NDArray[np.float64]]:
            The air temperature in Celsius.
        relative_humidity: Union[HourlyContinuousCollection, pd.DataFrame, pd.Series, NDArray[np.float64]]:
            The relative humidity in percentage.
        mean_radiant_temperature: Union[HourlyContinuousCollection, pd.DataFrame, pd.Series, NDArray[np.float64]]:
            The mean radiant temperature in Celsius.
        wind_speed: Union[HourlyContinuousCollection, pd.DataFrame, pd.Series, NDArray[np.float64]]:
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
        print("Calculating UTCI - Ladybug HourlyContinuousCollection")
        return UTCI(
            air_temperature=air_temperature,
            rel_humidity=relative_humidity,
            rad_temperature=mean_radiant_temperature,
            wind_speed=wind_speed,
        ).universal_thermal_climate_index

    if all((isinstance(i, (float, int)) for i in _inputs)):
        print("Calculating UTCI - float/int")
        return _utci_ndarray(
            air_temperature=air_temperature,
            relative_humidity=relative_humidity,
            mean_radiant_temperature=mean_radiant_temperature,
            wind_speed=np.clip([wind_speed], 0, 17)[0],
        )

    if all((isinstance(i, pd.DataFrame) for i in _inputs)):
        print("Calculating UTCI - pandas DataFrame")
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
        print("Calculating UTCI - pandas Series")
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

    if all((isinstance(i, (List, Tuple)) for i in _inputs)):
        print("Calculating UTCI - List/Tuple")
        return _utci_ndarray(
            air_temperature=np.array(air_temperature),
            relative_humidity=np.array(relative_humidity),
            mean_radiant_temperature=np.array(mean_radiant_temperature),
            wind_speed=np.clip(np.array(wind_speed), 0, 17),
        )

    try:
        print("Calculating UTCI - np.array")
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
        )
