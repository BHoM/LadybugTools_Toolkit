from pathlib import Path

import numpy as np
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.sunpath import Sunpath
from ladybugtools_toolkit.external_comfort.cityscale.shadow import shadow
from ladybugtools_toolkit.external_comfort.cityscale.sky_view import sky_view
from ladybugtools_toolkit.external_comfort.simulate.simulation_result import (
    SimulationResult,
)


def mrt(
    asc_file: Path,
    simulation_result: SimulationResult,
    month: int,
    day: int,
    hour: int,
) -> np.ndarray:
    """Estimate the MRT for a given LIDAR generated topography map. This method works best with
        detailed Lidar scans <2m. The output may not be accurate, but it's enough for an
        approximation.

    Args:
        asc_file (Path):
            The path to a TIF-style file containing topographic height data.
        simulation_result (SimulationResult):
            A pre-run simulation result from which to interpolate the resultant MRT.
        month (int):
            The month to estimate MRT for.
        day (int):
            The day to estimate MRT for.
        hour (int):
            The hour to estimate MRT for.

    Returns:
        np.ndarray:
            A matrix of values, one-per-pixel fo the original input TIF-style image.
    """

    # calculate sun position
    sp = Sunpath.from_location(simulation_result.epw.location)
    sun = sp.calculate_sun(month, day, hour)

    # get mrt for selected time
    ap = AnalysisPeriod(
        st_month=month,
        end_month=month,
        st_hour=hour,
        end_hour=hour,
        st_day=day,
        end_day=day,
    )
    mrt_shaded = (
        simulation_result.shaded_mean_radiant_temperature.filter_by_analysis_period(
            ap
        ).values[0]
    )
    mrt_unshaded = (
        simulation_result.unshaded_mean_radiant_temperature.filter_by_analysis_period(
            ap
        ).values[0]
    )

    # load sky-view file to get sky-dome visibility
    sky_view_matrix = sky_view(asc_file)

    # get interpolation mask (shaded/unshaded)
    if sun.altitude > 0:
        interp_matrix = shadow(asc_file, sun.altitude, sun.azimuth) * sky_view_matrix
        mrt_matrix = np.interp(interp_matrix, [0, 1], [mrt_shaded, mrt_unshaded])
    else:
        mrt_matrix = np.interp(
            sky_view_matrix.flatten(), [0, 1], [mrt_shaded, mrt_unshaded]
        ).reshape(sky_view_matrix.shape)

    return mrt_matrix
