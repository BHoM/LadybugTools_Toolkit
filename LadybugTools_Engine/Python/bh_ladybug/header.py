from __future__ import annotations

from typing import Dict, List, TypeVar

import pandas as pd

from .analysisperiod import BH_AnalysisPeriod

T = TypeVar("T")

from ladybug.datatype.base import DataTypeBase
from ladybug.header import Header


class BH_Header(Header):
    def __init__(
        self,
        data_type: DataTypeBase,
        unit: str,
        analysis_period: BH_AnalysisPeriod,
        metadata: Dict[str, str],
    ) -> BH_Header:
        super().__init__(data_type, unit, analysis_period, metadata)

    def _type(self) -> str:
        return self.__class__.__name__

    @property
    def analysis_period(self) -> BH_AnalysisPeriod:
        """Return analysis period data."""
        _ = self._analysis_period
        return BH_AnalysisPeriod(
            _.st_month,
            _.st_day,
            _.st_hour,
            _.end_month,
            _.end_day,
            _.end_hour,
            _.timestep,
            _.is_leap_year,
        )

    @property
    def datatype_str(self) -> str:
        return f"{self.data_type} ({self.unit})"
