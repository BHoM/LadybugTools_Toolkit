from __future__ import annotations

from typing import Dict, List, TypeVar

import pandas as pd

from analysisperiod import BH_AnalysisPeriod

T = TypeVar("T")

from ladybug.header import Header
from ladybug.datatype.base import DataTypeBase

class BH_Header(Header):
    def __init__(self, data_type: DataTypeBase, unit: str, analysis_period: BH_AnalysisPeriod, metadata: Dict[str, str]):
        super().__init__(data_type, unit, analysis_period, metadata)

    def _type(self):
        return self.__class__.__name__
