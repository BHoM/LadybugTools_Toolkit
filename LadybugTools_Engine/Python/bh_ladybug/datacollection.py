from __future__ import annotations

from typing import List, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")

from ladybug.datacollection import HourlyContinuousCollection, MonthlyCollection

from .header import BH_Header


class BH_HourlyContinuousCollection(HourlyContinuousCollection):
    def __init__(self, header: BH_Header, values: List[T]) -> BH_HourlyContinuousCollection:
        super().__init__(header, values)

    def _type(self) -> str:
        return self.__class__.__name__
    
    @property
    def header(self) -> BH_Header:
        """Return header."""
        _ = self._header
        return BH_Header(
            _.data_type,
            _.unit,
            _.analysis_period,
            _.metadata,
        )


    def to_series(self) -> pd.Series:
        """Convert a Ladybug hourlyContinuousCollection object into a Pandas Series object.

        Args:
            collection (HourlyContinuousCollection): A ladybug data collection.

        Returns:
            str: A Pandas Series.
        """

        return pd.Series(
            data=list(self.values),
            index=self.header.analysis_period.to_datetimes(),
            name=self.header.datatype_str,
        )

    def to_array(self) -> np.ndarray:
        """The BH_HourlyContinuousCollection as a numpy array.

        Returns:
            numpy.ndarray: A numpy array.
        """

        return np.array(self.values)


class BH_MonthlyCollection(MonthlyCollection):
    def __init__(self, header: BH_Header, values: List[T], datetimes: List[int]) -> BH_MonthlyCollection:
        super().__init__(header=header, values=values, datetimes=datetimes)

    def _type(self) -> str:
        return self.__class__.__name__
    
    @property
    def header(self) -> BH_Header:
        """Return header."""
        _ = self._header
        return BH_Header(
            _.data_type,
            _.unit,
            _.analysis_period,
            _.metadata,
        )

    def to_series(self) -> pd.Series:
        """Convert a Ladybug MonthlyContinuousCollection object into a Pandas Series object."""
        year = self.header.analysis_period.to_datetimes()[0].year
        return pd.Series(
            data=list(self.values),
            index=pd.date_range(f"{year}-01-01", periods=12, freq="MS"),
            name=self.header.datatype_str,
        )

    def to_array(self) -> np.ndarray:
        """The BH_MonthlyContinuousCollection as a numpy array."""

        return np.array(self.values)
    
    def to_hourly(self, method: str = None) -> BH_HourlyContinuousCollection:
        """Convert a Ladybug MonthlyContinuousCollection object into a Ladybug HourlyContinuousCollection object.
        
        Args:
            method (str): The method to use for annualizing the monthly values.
        
        Returns:
            BH_HourlyContinuousCollection: A Ladybug HourlyContinuousCollection object.
        """

        if method is None:
            method = "smooth"
        
        series = self.to_series()
        annual_hourly_index = pd.date_range(f"{series.index[0].year}-01-01", periods=8760, freq="H")
        series_annual = series.reindex(annual_hourly_index)
        series_annual[series_annual.index[-1]] = series_annual[series_annual.index[0]]
        
        if method == "smooth":
            values = series_annual.interpolate(method="quadratic").values
        elif method == "step":
            values = series_annual.interpolate(method="pad").values
        elif method == "linear":
            values = series_annual.interpolate(method="linear").values
        else:
            raise ValueError("Invalid method.")
        return BH_HourlyContinuousCollection(
            header=self.header,
            values=values,
        )
