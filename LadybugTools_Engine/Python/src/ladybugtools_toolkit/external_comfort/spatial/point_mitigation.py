"""Methods for modifying single points in-situ."""

from copy import copy  # pylint: disable=E0401

import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection

from ...ladybug_extension.datacollection import collection_from_series
from ..externalcomfort import ExternalComfort, Typology
from ..simulate import SimulationResult


class PointMitigation:
    """Adjust point-location-based UTCI using adjustments to the composite
    DBT, MRT, RH and WS values in-situ.

    Args:
        simulation_result (SimulationResult):
            The simulation result to get point-values from.
        point_dbt (pd.Series | HourlyContinuousCollection):
            The DBT values to set for the theoretical point.
        point_mrt (pd.Series | HourlyContinuousCollection):
            The MRT values to set for the theoretical point.
        point_rh (pd.Series | HourlyContinuousCollection):
            The RH values to set for the theoretical point.
        point_ws (pd.Series | HourlyContinuousCollection):
            The WS values to set for the theoretical point.

    Returns:
        PointMitigation:
            The PointMitigation object.
    """

    def __init__(
        self,
        simulation_result: SimulationResult,
        point_dbt: pd.Series | HourlyContinuousCollection,
        point_mrt: pd.Series | HourlyContinuousCollection,
        point_rh: pd.Series | HourlyContinuousCollection,
        point_ws: pd.Series | HourlyContinuousCollection,
    ) -> "PointMitigation":
        self._point_dbt = (
            point_dbt
            if isinstance(point_dbt, HourlyContinuousCollection)
            else collection_from_series(point_dbt)
        )
        self._point_mrt = (
            point_mrt
            if isinstance(point_mrt, HourlyContinuousCollection)
            else collection_from_series(point_mrt)
        )
        self._point_rh = (
            point_rh
            if isinstance(point_rh, HourlyContinuousCollection)
            else collection_from_series(point_rh)
        )
        self._point_ws = (
            point_ws
            if isinstance(point_ws, HourlyContinuousCollection)
            else collection_from_series(point_ws)
        )

        # adjust values in sim res to contain values from point location
        sim_res = copy(simulation_result)
        sim_res.UnshadedMeanRadiantTemperature = self._point_mrt
        setattr(
            getattr(
                sim_res.epw,
                "wind_speed"),
            "values",
            self._point_ws.values)
        setattr(
            getattr(
                sim_res.epw,
                "relative_humidity"),
            "values",
            self._point_rh.values)
        setattr(
            getattr(sim_res.epw, "dry_bulb_temperature"),
            "values",
            self._point_dbt.values,
        )
        self.simulation_result = sim_res

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def apply_mitigation(self, typology: Typology) -> ExternalComfort:
        """Apply the new typology to the the ExternalComfort result.

        Args:
            typology (Typology):
                The new typology to apply to the result.
        Returns:
            ExternalComfort:
                The ExternalComfort result with the new typology applied.
        """
        return ExternalComfort(self.simulation_result, typology)
