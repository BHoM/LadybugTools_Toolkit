# TODO - this file not really used - methods now within other classes. Tidy up and extract useful parts

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from tqdm import tqdm

from ...bhomutil.analytics import CONSOLE_LOGGER
from ...helpers import load_dataset, store_dataset
from ...honeybee_extension.results import load_ill, load_pts, load_res, make_annual
from ...ladybug_extension.datacollection import to_series
from ...ladybug_extension.epw import sun_position_list
from ..moisture import evaporative_cooling_effect
from ..utci import utci_parallel
from .calculate import shaded_unshaded_interpolation
from .cfd import spatial_wind_speed
from .metric import SpatialMetric
from .moisture import load_moisture_sources


def ws_epw(
    simulation_directory: Path,
    epw: EPW = None,
) -> pd.DataFrame:
    """Return the wind speed from the simulation directory using a the EPW file to
        assign hourly WS to each point in the simulation, and create the file to store this as
        a compressed object if not already done.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame:
            A dataframe with the spatial wind speed.
    """

    metric = SpatialMetric.WS_EPW
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )
    spatial_points = points(simulation_directory)
    n_pts = len(spatial_points.index)

    series = to_series(epw.wind_speed)
    df = pd.DataFrame(np.tile(series.values, (n_pts, 1)).T, index=series.index).round(2)
    store_dataset(df, fp, downcast=True)

    return df


def ws_cfd(
    simulation_directory: Path,
    epw: EPW = None,
) -> pd.DataFrame:
    """Return the wind speed from the simulation directory using results from a CFD simulation,
        and create the file to store this as a compressed object if not already done.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame:
            A dataframe with the spatial wind speed.
    """

    metric = SpatialMetric.WS_CFD
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )

    df = spatial_wind_speed(simulation_directory, epw).round(2)
    store_dataset(df, fp, downcast=True)

    return df


def wd_epw(
    simulation_directory: Path,
    epw: EPW = None,
) -> pd.DataFrame:
    """Return the wind direction from the simulation directory using a the EPW file to
        assign hourly WD to each point in the simulation, and create the file to store this as
        a compressed object if not already done.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame:
            A dataframe with the spatial wind direction.
    """

    metric = SpatialMetric.WD_EPW
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )

    spatial_points = points(simulation_directory)
    n_pts = len(spatial_points.index)

    series = to_series(epw.wind_direction)
    df = pd.DataFrame(np.tile(series.values, (n_pts, 1)).T, index=series.index).round(0)
    store_dataset(df, fp, downcast=True)

    return df


def utci_interpolated(
    simulation_directory: Path,
    unshaded_universal_thermal_climate_index: HourlyContinuousCollection,
    shaded_universal_thermal_climate_index: HourlyContinuousCollection,
    total_irradiance: pd.DataFrame,
    sky_view_df: pd.DataFrame,
    epw: EPW,
) -> pd.DataFrame:
    """Return the UTCI from the simulation directory, and create the file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Path):
            The simulation directory.
        unshaded_universal_thermal_climate_index (HourlyContinuousCollection):
            A collection containing unshaded UTCI values.
        shaded_universal_thermal_climate_index (HourlyContinuousCollection):
            A collection containing shaded UTCI values.
        total_irradiance (pd.DataFrame):
            A dataframe containing spatial total irradiance values.
        sky_view (pd.DataFrame):
            A dataframe containing point-wise sky-view values.
        epw (EPW):
            The associate EPW file.

    Returns:
        pd.DataFrame:
            A dataframe containing spatial UTCI values.
    """

    metric = SpatialMetric.UTCI_INTERPOLATED
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )

    df = shaded_unshaded_interpolation(
        unshaded_value=unshaded_universal_thermal_climate_index.values,
        shaded_value=shaded_universal_thermal_climate_index.values,
        total_irradiance=total_irradiance.values,
        sky_view=sky_view_df.squeeze().values,
        sun_up=[i.altitude > 0 for i in sun_position_list(epw)],
    ).round(2)
    store_dataset(df, fp, downcast=True)

    return df


def utci_calculated(
    simulation_directory: Path,
    epw: EPW,
    unshaded_mean_radiant_temperature: HourlyContinuousCollection,
    shaded_mean_radiant_temperature: HourlyContinuousCollection,
    total_irradiance: pd.DataFrame,
    sky_view_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return the UTCI calculated using available matrices from moisture effects and/or CFD.

    Args:
        simulation_directory (Path):
            The simulation directory.
        epw (EPW):
            The associate EPW file.
        unshaded_mean_radiant_temperature (HourlyContinuousCollection):
            A collection containing unshaded MRT values.
        shaded_mean_radiant_temperature (HourlyContinuousCollection):
            A collection containing shaded MRT values.
        total_irradiance (pd.DataFrame):
            A dataframe containing spatial total irradiance values.
        sky_view (pd.DataFrame):
            A dataframe containing point-wise sky-view values.

    Returns:
        pd.DataFrame:
            A dataframe containing spatial UTCI values.
    """

    metric = SpatialMetric.UTCI_CALCULATED
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )

    # check that moisture impacted RH/DBT is available, and use that if it is
    if SpatialMetric.DBT_EVAP.filepath(simulation_directory).exists():
        dbt = dbt_evap(simulation_directory, epw)
    else:
        dbt = dbt_epw(simulation_directory, epw)

    if SpatialMetric.RH_EVAP.filepath(simulation_directory).exists():
        rh = rh_evap(simulation_directory, epw)
    else:
        rh = rh_epw(simulation_directory, epw)

    # check that CFD wind speeds are available, and use that if it is
    if SpatialMetric.WS_CFD.filepath(simulation_directory).exists():
        ws = ws_cfd(simulation_directory, epw)
    else:
        ws = ws_epw(simulation_directory, epw)

    mrt = mrt_interpolated(
        simulation_directory,
        unshaded_mean_radiant_temperature,
        shaded_mean_radiant_temperature,
        total_irradiance,
        sky_view_df,
        epw,
    )

    df = pd.DataFrame(
        utci_parallel(dbt.values, mrt.values, ws.values, rh.values), index=mrt.index
    ).round(2)
    store_dataset(df, fp, downcast=True)

    return df


def sky_view(simulation_directory: Path) -> pd.DataFrame:
    """Get the sky view from the simulation directory.

    Args:
        simulation_directory (Path):
            The simulation directory containing a sky view RES file.

    Returns:
        pd.DataFrame:
            The sky view dataframe.
    """

    metric = SpatialMetric.SKY_VIEW
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )

    res_files = list((simulation_directory / "sky_view" / "results").glob("*.res"))
    df = load_res(res_files).clip(lower=0, upper=100).round(2)
    store_dataset(df, fp, downcast=True)

    return df


def rh_evap(simulation_directory: Path, epw: EPW) -> List[pd.DataFrame]:
    """Return the relative humidity from the simulation directory calculating the effective
        RH values following addition of moisture into the air, then create the file to store
        this as a compressed object if not already done.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame:
            A dataframe with the spatial relative-humidity.
    """

    metric = SpatialMetric.RH_EVAP
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )

    return dbt_rh_evap(simulation_directory, epw)[1]


def rh_epw(
    simulation_directory: Path,
    epw: EPW = None,
) -> pd.DataFrame:
    """Return the relative humidity from the simulation directory using a the EPW file to
        assign hourly RH to each point in the simulation, and create the file to store this as
        a compressed object if not already done.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame:
            A dataframe with the spatial relative humidity.
    """

    metric = SpatialMetric.RH_EPW
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )

    spatial_points = points(simulation_directory)
    n_pts = len(spatial_points.index)

    series = to_series(epw.relative_humidity)
    df = pd.DataFrame(np.tile(series.values, (n_pts, 1)).T, index=series.index).round(1)
    store_dataset(df, fp, downcast=True)

    return df


def rad_total(simulation_directory: Path) -> pd.DataFrame:
    """Return the total irradiance from the simulation directory, and create the file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Path):
            The directory containing a total irradiance ILL file.

    Returns:
        pd.DataFrame:
            A dataframe with the total irradiance.
    """

    metric = SpatialMetric.RAD_TOTAL
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )

    ill_files = list(
        (simulation_directory / "annual_irradiance" / "results" / "total").glob("*.ill")
    )
    df = (
        make_annual(load_ill(ill_files))
        .fillna(0)
        .clip(lower=0)
        .droplevel(0, axis=1)
        .round(0)
    )
    store_dataset(df, fp, downcast=True)

    return df


def rad_direct(simulation_directory: Path) -> pd.DataFrame:
    """Return the direct irradiance from the simulation directory, and create the file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Path):
            The directory containing a direct irradiance ILL file.

    Returns:
        pd.DataFrame:
            A dataframe with the direct irradiance.
    """

    metric = SpatialMetric.RAD_DIRECT
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )

    ill_files = list(
        (simulation_directory / "annual_irradiance" / "results" / "direct").glob(
            "*.ill"
        )
    )
    df = (
        make_annual(load_ill(ill_files))
        .fillna(0)
        .clip(lower=0)
        .droplevel(0, axis=1)
        .round(0)
    )
    store_dataset(df, fp, downcast=True)

    return df


def rad_diffuse(
    simulation_directory: Path,
    total_irradiance: pd.DataFrame = None,
    direct_irradiance: pd.DataFrame = None,
) -> pd.DataFrame:
    """Return the diffuse irradiance from the simulation directory, and create the file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Path):
            The directory containing a diffuse irradiance ILL file.
        total_irradiance (pd.DataFrame, optional):
            If given along with direct_irradiance, then calculation will be completed faster.
            Default is None.
        direct_irradiance (pd.DataFrame, optional):
            If given along with total_irradiance, then calculation will be completed faster.
            Default is None.

    Returns:
        pd.DataFrame: A dataframe with the diffuse irradiance.
    """

    metric = SpatialMetric.RAD_DIFFUSE
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )

    if (total_irradiance is None) and (direct_irradiance is None):
        total_irradiance = rad_total(simulation_directory)
        direct_irradiance = rad_direct(simulation_directory)

    df = (total_irradiance - direct_irradiance).round(0).clip(lower=0)
    store_dataset(df, fp, downcast=True)

    return df


def points(simulation_directory: Path) -> pd.DataFrame:
    """Return the points results from the simulation directory, and create the file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Path):
            The directory containing a pts file.

    Returns:
        pd.DataFrame:
            A dataframe with the points locations.
    """

    metric = SpatialMetric.POINTS
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )
    points_files = list(
        (simulation_directory / "sky_view" / "model" / "grid").glob("*.pts")
    )
    df = load_pts(points_files).droplevel(0, axis=1)
    store_dataset(df, fp, downcast=True)
    return df


def mrt_interpolated(
    simulation_directory: Path,
    unshaded_mean_radiant_temperature: HourlyContinuousCollection,
    shaded_mean_radiant_temperature: HourlyContinuousCollection,
    total_irradiance_df: pd.DataFrame,
    sky_view_df: pd.DataFrame,
    epw: EPW,
) -> pd.DataFrame:
    """Return the MRT from the simulation directory, and create the file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Path):
            The simulation directory.
        unshaded_mean_radiant_temperature (HourlyContinuousCollection):
            A collection containing unshaded MRT values.
        shaded_mean_radiant_temperature (HourlyContinuousCollection):
            A collection containing shaded MRT values.
        total_irradiance (pd.DataFrame):
            A dataframe containing spatial total irradiance values.
        sky_view (pd.DataFrame):
            A dataframe containing point-wise sky-view values.
        epw (EPW):
            The associate EPW file.

    Returns:
        pd.DataFrame:
            A dataframe containing spatial MRT values.
    """
    metric = SpatialMetric.MRT_INTERPOLATED
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )
    df = shaded_unshaded_interpolation(
        unshaded_value=unshaded_mean_radiant_temperature.values,
        shaded_value=shaded_mean_radiant_temperature.values,
        total_irradiance=total_irradiance_df.values,
        sky_view=sky_view_df.squeeze().values,
        sun_up=[i.altitude > 0 for i in sun_position_list(epw)],
    ).round(2)
    store_dataset(df, fp, downcast=True)

    return df


def evap_clg_magnitude(simulation_directory: Path, epw: EPW) -> pd.DataFrame:
    """Calculate the magnitude of evaporative cooling across a spatial case.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame: A spatial matrix of time/pt containing evaporative cooling effective.
    """

    metric = SpatialMetric.EVAP_CLG
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )

    # create an index to attribute calculated values
    idx = to_series(epw.dry_bulb_temperature).index

    # load moisture sources
    moisture_sources = load_moisture_sources(simulation_directory)

    # load spatial points in list of [[X, Y], [X, Y], [X, Y]]
    spatial_points = points(simulation_directory)[["x", "y"]].values

    # get moisture matrices per moisture source, and resultant matrix
    df = pd.DataFrame(
        np.amax(
            [
                i.spatial_moisture(
                    spatial_points, epw, simulation_directory=simulation_directory
                )
                for i in moisture_sources
            ],
            axis=0,
        ),
        index=idx,
    ).round(2)

    store_dataset(df, fp, downcast=True)

    return df


def dbt_rh_evap(simulation_directory: Path, epw: EPW) -> List[pd.DataFrame]:
    """Calculate the effective DBT/RH from a spatially distributed set of moisture sources.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        List[pd.DataFrame]: A list of effective DBT and effective RH.

    """

    dbt_fp = SpatialMetric.DBT_EVAP.filepath(simulation_directory)
    rh_fp = SpatialMetric.RH_EVAP.filepath(simulation_directory)

    if dbt_fp.exists() and rh_fp.exists():
        return load_dataset(dbt_fp, upcast=True), load_dataset(rh_fp, upcast=True)

    # load evaporative cooling moisture magnitude matrix
    moisture_matrix = evap_clg_magnitude(simulation_directory, epw)

    # create matrix of dbt, rh based on moisture matrix
    # this calculates both dbt and rh, as the both are linked and should be calculated in parallel
    dbt_matrix = []
    rh_matrix = []
    for n, (_, row) in tqdm(
        enumerate(moisture_matrix.iterrows()),
        desc="Calculating evaporatively cooled DBT/RH",
    ):
        dbt_base = epw.dry_bulb_temperature[n]
        rh_base = epw.relative_humidity[n]
        atm_base = epw.atmospheric_station_pressure[n]
        if row.sum() == 0:
            dbt_matrix.append(np.tile(dbt_base, len(row)))
            rh_matrix.append(np.tile(rh_base, len(row)))
        else:
            dbt, rh = evaporative_cooling_effect(dbt_base, rh_base, row, atm_base)
            dbt_matrix.append(dbt)
            rh_matrix.append(rh)

    idx = to_series(epw.dry_bulb_temperature).index

    dbt_df = pd.DataFrame(np.array(dbt_matrix), index=idx).round(2)
    store_dataset(dbt_df, dbt_fp, downcast=True)

    rh_df = pd.DataFrame(np.array(rh_matrix), index=idx).round(1)
    store_dataset(rh_df, rh_fp, downcast=True)

    return dbt_df, rh_df


def dbt_evap(simulation_directory: Path, epw: EPW) -> List[pd.DataFrame]:
    """Return the dry-bulb-temperatures from the simulation directory calculating the effective
        DBT values following addition of moisture into the air, then create the file to store
        this as a compressed object if not already done.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame:
            A dataframe with the spatial dry-bulb-temperature.
    """
    metric = SpatialMetric.DBT_EVAP
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )

    return dbt_rh_evap(simulation_directory, epw)[0]


def dbt_epw(
    simulation_directory: Path,
    epw: EPW = None,
) -> pd.DataFrame:
    """Return the dry-bulb-temperatures from the simulation directory using an EPW file to
        assign hourly DBT to each point in the simulation, and create the file to store this as
        a compressed object if not already done.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame:
            A dataframe with the spatial dry-bulb-temperature.
    """
    metric = SpatialMetric.DBT_EPW
    fp = metric.filepath(simulation_directory)

    if fp.exists():
        CONSOLE_LOGGER.info(
            f"[{simulation_directory.name}] - Loading {metric.description()}"
        )
        return load_dataset(fp, upcast=True)

    CONSOLE_LOGGER.info(
        f"[{simulation_directory.name}] - Generating {metric.description()}"
    )
    spatial_points = points(simulation_directory)
    n_pts = len(spatial_points.index)

    series = to_series(epw.dry_bulb_temperature)
    df = pd.DataFrame(np.tile(series.values, (n_pts, 1)).T, index=series.index).round(2)
    store_dataset(df, fp, downcast=True)

    return df
