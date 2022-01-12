from __future__ import annotations
from datetime import datetime

from typing import List, TypeVar

import pandas as pd

T = TypeVar("T")

from ladybug.analysisperiod import AnalysisPeriod


class BH_AnalysisPeriod(AnalysisPeriod):
    def __init__(
        self,
        st_month: int = 1,
        st_day: int = 1,
        st_hour: int = 0,
        end_month: int = 12,
        end_day: int = 31,
        end_hour: int = 23,
        timestep: int = 1,
        is_leap_year: bool = False,
    ) -> BH_AnalysisPeriod:
        super().__init__(
            st_month,
            st_day,
            st_hour,
            end_month,
            end_day,
            end_hour,
            timestep,
            is_leap_year,
        )

    def _type(self) -> str:
        return self.__class__.__name__
    
    def to_datetimes(self) -> List[datetime]:
        return pd.to_datetime(self.datetimes).to_list()
