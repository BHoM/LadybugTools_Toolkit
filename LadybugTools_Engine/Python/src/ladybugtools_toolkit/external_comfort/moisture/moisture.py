from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class MoistureSource:
    def __init__(
        self, identifier: str, magnitude: float, points: List[int]
    ) -> MoistureSource:
        """An object defining where moisture is present in a spatial thermal comfort simulation

        Args:
            identifier (str): The ID of this moisture source.
            magnitude (float): The evaporative cooling effectiveness of this moisture source.
            points (List[int]): The point indices (related to a list of points) where this moisture
                source is applied.

        Returns:
            _type_: _description_
        """
        self.id = identifier
        self.magnitude = magnitude
        self.points = points

    @classmethod
    def from_json(cls, json_file: Path) -> List[MoistureSource]:
        """Create a set of MoistureSource objects from a json file.

        Args:
            json_file (Path): A file containing MoistureSource objects in json format.

        Returns:
            List[MoistureSource]: A list of MoistureSource objects.
        """
        objs = []
        with open(json_file, "r", encoding="utf-8") as fp:
            water_sources = json.load(fp)
        for water_source in water_sources:
            objs.append(
                MoistureSource(
                    water_source["id"],
                    water_source["magnitude"],
                    np.array(water_source["points"]).astype(np.float16),
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

        return {
            "id": self.id,
            "magnitude": self.magnitude,
            "points": self.points,
        }

    @property
    def points_xy(self) -> List[List[float]]:
        """Return a list of x, y coordinates for each point in the moisture source."""
        return self.points[:, :2]

    @property
    def pathsafe_id(self) -> str:
        """Create a path-safe ID for this object"""
        return "".join(
            [c for c in self.id if c.isalpha() or c.isdigit() or c == " "]
        ).rstrip()
