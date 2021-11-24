from __future__ import annotations

from enum import Enum


class ForecastYear(Enum):
    _2020 = 2020
    _2050 = 2050
    _2080 = 2080

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{__class__.__name__}[{self}]"

    @classmethod
    def from_str(cls, str: str) -> ForecastYear:
        return getattr(ForecastYear, f"_{str}")

    @classmethod
    def from_int(cls, int: int) -> ForecastYear:
        return getattr(ForecastYear, f"_{int}")


class EmissionsScenario(Enum):
    A2a = "A2a"
    A2b = "A2b"
    A2c = "A2c"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{__class__.__name__}[{self}]"

    @classmethod
    def from_str(cls, str: str) -> EmissionsScenario:
        return getattr(EmissionsScenario, f"{str}")
