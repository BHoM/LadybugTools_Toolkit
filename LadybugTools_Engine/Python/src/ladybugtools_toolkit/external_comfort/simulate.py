"""_"""
# pylint: disable=E0401
import itertools
import warnings

# pylint: enable=E0401

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..bhom import decorator_factory
from ..helpers import evaporative_cooling_effect
from ..ladybug_extension.epw import AnalysisPeriod, collection_to_series
from ._simulatebase import SimulationResult
from .utci import utci


@decorator_factory()
def ranked_mitigations(
    simulation_result: SimulationResult,
    n_steps: int = 8,
    analysis_period: AnalysisPeriod = None,
    comfort_limits: tuple[float] = (9, 26),
    evaporative_cooling_effectiveness: float = 0.7,
) -> pd.DataFrame:
    """Determine the relative impact of different measures to adjust UTCI.

    Args:
        simulation_result (SimulationResult):
            A SimulationResult object containing the results of a simulation.
        n_steps (int, optional):
            The number of steps to calculate per variable
            (n_steps**3 is the number of calculations run).
            Defaults to 8.
        analysis_period (AnalysisPeriod, optional):
            A period to apply to results. Defaults to None.
        comfort_limits (tuple[float], optional):
            Optional limits to set what is considered comfortable.
            Defaults to (9, 26).
        evaporative_cooling_effectiveness (float, optional):
            The effectiveness of evaporative cooling to apply to all timesteps.
            Defaults to 0.7.

    Returns:
        pd.DataFrame:
            A table of relative UTCI impact proportions.
    """
    warnings.warn(
        "This method is not yet complete, or at least it requires some further "
        "thought to make it faster, better, stronger!"
    )

    # TODO - break method out into parts - namely basic inputs and single
    # Series output, referenced from elsewhere

    if analysis_period is None:
        analysis_period = AnalysisPeriod()

    # get comfort limits as single values
    comfort_low = min(comfort_limits)
    comfort_mid = np.mean(comfort_limits)
    comfort_high = max(comfort_limits)

    # construct dataframe containing inputs to this process
    epw = simulation_result.epw
    atm = collection_to_series(epw.atmospheric_station_pressure, "atm")
    dbt = collection_to_series(epw.dry_bulb_temperature, "dbt")
    rh = collection_to_series(epw.relative_humidity, "rh")
    ws = collection_to_series(epw.wind_speed, "ws")
    mrt_unshaded = collection_to_series(
        simulation_result.unshaded_mean_radiant_temperature, "mrt_unshaded"
    )
    mrt_shaded = collection_to_series(
        simulation_result.shaded_mean_radiant_temperature, "mrt_shaded"
    )
    utci_unshaded = utci(dbt, rh, mrt_unshaded, ws).rename("utci_unshaded")
    df = pd.concat([atm, dbt, rh, ws, mrt_unshaded, mrt_shaded, utci_unshaded], axis=1)

    # filter by analysis period
    df = df.iloc[list(analysis_period.hoys_int)]

    # get comfort mask for baseline
    df["utci_unshaded_comfortable"] = df.utci_unshaded.between(
        comfort_low, comfort_high
    )

    # get distance from comfortable (midpoint) for each timestep in baseline
    df["utci_unshaded_distance_from_comfortable_midpoint"] = (
        df.utci_unshaded - comfort_mid
    )
    df[
        "utci_unshaded_distance_from_comfortable"
    ] = df.utci_unshaded_distance_from_comfortable_midpoint.where(
        ~df.utci_unshaded_comfortable, 0
    )

    # get possible values for shade/shelter/evapclg
    shading_proportions = np.linspace(1, 0, n_steps)
    wind_shelter_proportions = np.linspace(0, 1, n_steps)
    evap_clg_proportions = np.linspace(0, 1, n_steps)

    def _temp(
        dbt,
        rh,
        atm,
        ws,
        mrt_unshaded,
        mrt_shaded,
        utci_unshaded_distance_from_comfortable_midpoint,
        name,
    ):
        # create feasible ranges of values
        dbts, rhs = np.array(
            [
                evaporative_cooling_effect(
                    dbt, rh, evap_x * evaporative_cooling_effectiveness, atm
                )
                for evap_x in evap_clg_proportions
            ]
        ).T
        wss = ws * wind_shelter_proportions
        mrts = np.interp(shading_proportions, [0, 1], [mrt_unshaded, mrt_shaded])

        # create all possible combinations of inputs
        dbts, rhs, mrts, wss = np.array(list(itertools.product(dbts, rhs, mrts, wss))).T
        utcis = utci([dbts], [rhs], [mrts], [wss])[0]
        shad_props, _, windshlt_props, evapclg_props = np.array(
            list(
                itertools.product(
                    shading_proportions,
                    shading_proportions,
                    wind_shelter_proportions,
                    evap_clg_proportions,
                )
            )
        ).T

        # reshape matrix
        mtx = pd.DataFrame(
            [
                dbts,
                rhs,
                mrts,
                wss,
                utcis,
                shad_props,
                windshlt_props,
                evapclg_props,
            ],
            index=[
                "dbt",
                "rh",
                "mrt",
                "ws",
                "utci",
                "shade",
                "wind shelter",
                "evaporative cooling",
            ],
        ).T

        # get comfort mask for current timestep
        mtx["utci_comfortable"] = mtx.utci.between(comfort_low, comfort_high)

        # get distance from comfortable (midpoint) for current timestep
        mtx["utci_distance_from_comfortable_midpoint"] = mtx.utci - comfort_mid
        mtx[
            "utci_distance_from_comfortable"
        ] = mtx.utci_distance_from_comfortable_midpoint.where(~mtx.utci_comfortable, 0)

        # determine whether comfort has improved, and drop rows where it hasnt
        mtx["comfort_improved"] = abs(
            mtx.utci_distance_from_comfortable_midpoint
        ) < abs(utci_unshaded_distance_from_comfortable_midpoint)
        mtx = mtx[mtx.comfort_improved]

        # sort by distance to comfort midpoint, to get optimal conditions
        mtx["utci_distance_from_comfortable_midpoint_absolute"] = abs(
            mtx.utci_distance_from_comfortable_midpoint
        )
        mtx = mtx.sort_values("utci_distance_from_comfortable_midpoint_absolute")

        # normalise variables
        temp = (
            mtx[["shade", "wind shelter", "evaporative cooling"]]
            / mtx[["shade", "wind shelter", "evaporative cooling"]].sum(axis=0)
        ).reset_index(drop=True)

        # get topmost 25%
        ranks = temp.head(int(len(temp) / 4)).mean(axis=0)
        ranks = ranks / ranks.sum()
        ranks.name = name

        # TODO - include "do nothing" as an option for ranking!

        return ranks

    tqdm.pandas(
        desc=f"Calculating ranked beneficial impact of comfort mitigation measures for {epw}"
    )
    return df.progress_apply(
        lambda row: _temp(
            row.dbt,
            row.rh,
            row.atm,
            row.ws,
            row.mrt_unshaded,
            row.mrt_shaded,
            row.utci_unshaded_distance_from_comfortable_midpoint,
            row.name,
        ),
        axis=1,
    )
