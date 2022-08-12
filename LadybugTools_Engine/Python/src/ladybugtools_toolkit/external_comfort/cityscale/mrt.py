from pathlib import Path

import cv2
import numpy as np
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.sunpath import Sunpath
from ladybugtools_toolkit.external_comfort.cityscale.shadow import shadow
from ladybugtools_toolkit.external_comfort.cityscale.skyviewfactor import skyviewfactor
from ladybugtools_toolkit.external_comfort.simulate.simulation_result import (
    SimulationResult,
)
from ladybugtools_toolkit.ladybug_extension.analysis_period.to_datetimes import (
    to_datetimes,
)
from PIL import Image


def mrt(
    tif_file: Path,
    simulation_result: SimulationResult,
    analysis_period: AnalysisPeriod = AnalysisPeriod(
        st_month=3, st_day=21, st_hour=9, end_month=3, end_day=21, end_hour=9
    ),
    output_dir: Path = None,
) -> np.ndarray:
    """Estimate the MRT for a given LIDAR generated topography map. This method works best with
        detailed Lidar scans <2m. The output may not be accurate, but it's enough for an
        approximation.

    Args:
        tif_file (Path):
            The path to a TIF-style file containing topographic height data.
        simulation_result (SimulationResult):
            A pre-run simulation result from which to interpolate the resultant MRT.
        analysis_period (AnalysisPeriod):
            The analysis period to calculate MRT for (each time-step within that period).
        output_dir (Path, optional):
            A directory in which to store the generated TIF files.

    Returns:
        np.ndarray:
            A matrix of values, one-per-pixel representing MRT.
    """

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if analysis_period.timestep != 1:
        raise ValueError(
            "analysis_periods with time-steps other than 1-per-hour cannot be processed using this method."
        )

    # get datetimes to process
    datetimes = to_datetimes(analysis_period)

    sp = Sunpath.from_location(simulation_result.epw.location)

    mrt_shaded = (
        simulation_result.shaded_mean_radiant_temperature.filter_by_analysis_period(
            analysis_period
        ).values
    )
    mrt_unshaded = (
        simulation_result.unshaded_mean_radiant_temperature.filter_by_analysis_period(
            analysis_period
        ).values
    )

    # create sky-view array
    print("- Calculating terrain elevation sky-view factor")
    svf_file = skyviewfactor(tif_file, output_file=output_dir / "sky_view_factor.tif")
    sky_view_matrix = cv2.imread(svf_file.as_posix(), -1).clip(0, 1)

    # Calculate MRT for each datetime
    mrt_images = []
    for n, dt in enumerate(datetimes):
        if sp.calculate_sun_from_date_time(dt, is_solar_time=True).altitude > 0:
            print(f"- [{dt:%b %d %H:%M}] Calculating MRT using sun shadows")
            sun_shadow_file = shadow(
                tif_file, dt, output_dir / f"sun_shadow_{dt:%m%d%H}.tif"
            )
            sun_shadow_matrix = cv2.imread(sun_shadow_file.as_posix(), -1)
            sun_shadow_matrix = np.interp(
                sun_shadow_matrix,
                [np.amin(sun_shadow_matrix), np.amax(sun_shadow_matrix)],
                [0, 1],
            )
            mrt_matrix = np.interp(
                sun_shadow_matrix, [0, 1], [mrt_shaded[n], mrt_unshaded[n]]
            )
        else:
            print(f"- [{dt:%b %d %H:%M}] Calculating MRT using sky-view-factor")
            mrt_matrix = np.interp(
                sky_view_matrix, [0, 1], [mrt_shaded[n], mrt_unshaded[n]]
            )

        mrt_image = Image.fromarray(mrt_matrix)
        mrt_image.save(output_dir / f"mean_radiant_temperature_{dt:%m%d%H}.tif")
        mrt_images.append(mrt_image)

    return mrt_images
