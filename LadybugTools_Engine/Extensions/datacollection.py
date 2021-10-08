from typing import List, TypeVar

import pandas as pd

T = TypeVar("T")

from ladybug.datacollection import HourlyContinuousCollection
from ladybug.header import Header


class BH_HourlyContinuousCollection(HourlyContinuousCollection):
    def __init__(self, header: Header, values: List[T]):
        super().__init__(header, values)

    def _type(self):
        return self.__class__.__name__

    def datetime_index(self) -> pd.DatetimeIndex:
        """Generate a pandas DatetimeIndex for the current collection.

        Returns:
            DatetimeIndex: A pandas DatetimeIndex object.
        """

        n_hours = 8784 if self.header.analysis_period.is_leap_year else 8760
        year = 2020 if self.header.analysis_period.is_leap_year else 2021
        return pd.date_range(
            f"{year}-01-01 00:30:00", freq="60T", periods=n_hours, name="timestamp"
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
            index=self.datetime_index(),
            name=(self.location_str, self.datatype_str),
        )

    @property
    def datatype_str(self) -> str:
        try:
            return f"{self.header.metadata['description']} {self.header.data_type} ({self.header.unit})"
        except KeyError as e:
            return f"{self.header.data_type} ({self.header.unit})"

    @property
    def location_str(self) -> str:
        location_dict = {
            "city": "Unknown city",
            "country": "Unknown country",
            "source": "Unknown source",
        }

        for k in location_dict.keys():
            try:
                if self.header.metadata[k].strip() is "":
                    continue
                else:
                    location_dict[k] = self.header.metadata[k].strip()
            except KeyError as e:
                pass

        return f"{location_dict['city']} ({location_dict['country']}), {location_dict['source']}"
