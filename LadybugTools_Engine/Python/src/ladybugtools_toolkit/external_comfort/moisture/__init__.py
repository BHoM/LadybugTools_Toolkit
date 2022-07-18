from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass()
class Moisture:
    id: str = field(init=True, repr=True)
    magnitude: float = field(init=True, repr=False)
    points: List[int] = field(init=True, repr=False)

    @classmethod
    def from_json(cls, json_file: Path) -> List[Moisture]:
        """Create a set of MoistureSource objects from a json file.

        Args:
            json_file (Path): A file containing MoistureSource objects in json format.

        Returns:
            List[MoistureSource]: A list of MoistureSource objects.
        """
        objs = []
        with open(json_file, "r", encoding="utf-8") as fp:
            water_sources = json.load(fp)
        for ws in water_sources:
            objs.append(
                Moisture(
                    ws["id"], ws["magnitude"], np.array(ws["points"]).astype(np.float16)
                )
            )
        return objs

    def __repr__(self) -> str:
        return f"MoistureSource(id={self.id})"

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        d = {
            "id": self.id,
            "magnitude": self.magnitude,
            "points": self.points,
        }
        return d

    @property
    def points_xy(self) -> List[List[float]]:
        """Return a list of x, y coordinates for each point in the moisture source."""
        return self.points[:, :2]

    @property
    def pathsafe_id(self) -> str:
        return "".join(
            [c for c in self.id if c.isalpha() or c.isdigit() or c == " "]
        ).rstrip()
