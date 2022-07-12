from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ..encoder import Encoder
from ..shelter import Shelter


@dataclass(frozen=True)
class Typology:
    name: str = field(init=True, repr=True)
    shelters: List[Shelter] = field(init=True, repr=True, default_factory=list)
    evaporative_cooling_effectiveness: float = field(init=True, repr=True, default=0)
    wind_speed_multiplier: float = field(init=True, repr=True, default=1)

    def __post_init__(self) -> Typology:
        if self.shelters is None:
            object.__setattr__(self, "shelters", [])

        if Shelter._overlaps(self.shelters):
            raise ValueError("Shelters overlap")

        if self.wind_speed_multiplier < 0:
            raise ValueError("Wind speed multiplier must be greater than 0")

    @property
    def description(self) -> str:
        """Return a human readable description of the Typology object."""

        if self.wind_speed_multiplier == 1:
            wind_str = "wind speed per weatherfile"
        elif self.wind_speed_multiplier > 1:
            wind_str = f"wind speed increased by {self.wind_speed_multiplier - 1:0.0%}"
        else:
            wind_str = f"wind speed decreased by {1 - self.wind_speed_multiplier:0.0%}"

        # Remove shelters that provide no shelter
        shelters = [i for i in self.shelters if i.description != "unsheltered"]
        if len(shelters) > 0:
            shelter_str = " and ".join(
                [i.description for i in self.shelters]
            ).capitalize()
        else:
            shelter_str = "unsheltered".capitalize()

        if (self.evaporative_cooling_effectiveness != 0) and (
            self.wind_speed_multiplier != 1
        ):
            return f"{self.name}: {shelter_str}, with {self.evaporative_cooling_effectiveness} evaporative cooling effectiveness, and {wind_str}"
        elif self.evaporative_cooling_effectiveness != 0:
            return f"{self.name}: {shelter_str}, with {self.evaporative_cooling_effectiveness} evaporative cooling effectiveness"
        else:
            return f"{self.name}: {shelter_str}, with {wind_str}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, {[i for i in self.shelters]}, {self.evaporative_cooling_effectiveness}, {self.wind_speed_multiplier})"

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        d = {
            "name": self.name,
            "shelters": [i.to_dict() for i in self.shelters],
            "evaporative_cooling_effectiveness": self.evaporative_cooling_effectiveness,
            "wind_speed_multiplier": self.wind_speed_multiplier,
        }
        return d

    def to_json(self, file_path: str) -> Path:
        """Write the content of this object to a JSON file

        Returns:
            Path: The path to the newly created JSON file.
        """

        file_path: Path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)

        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, cls=Encoder, indent=4)

        return file_path
