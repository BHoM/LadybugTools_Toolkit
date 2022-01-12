from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")

from ladybug.location import Location


class BH_Location(Location):
    def __init__(
        self,
        city: str = None,
        state: str = None,
        country: str = None,
        latitude: float = 0,
        longitude: float = 0,
        time_zone: str = None,
        elevation: float = 0,
        station_id: str = None,
        source: str = None,
    ) -> BH_Location:
        super().__init__(
            city,
            state,
            country,
            latitude,
            longitude,
            time_zone,
            elevation,
            station_id,
            source,
        )

    def _type(self) -> str:
        return self.__class__.__name__
