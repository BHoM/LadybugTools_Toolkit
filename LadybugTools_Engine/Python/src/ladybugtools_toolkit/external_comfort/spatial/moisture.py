"""Methods for handling moisture."""

# pylint: disable=E0401
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ladybug.epw import EPW, AnalysisPeriod
from ladybug_geometry.geometry2d.pointvector import Point2D, Vector2D
from tqdm import tqdm

from ...helpers import (DecayMethod, angle_to_vector,
                        convert_keys_to_snake_case, proximity_decay)
from ...ladybug_extension.datacollection import analysis_period_to_datetimes
from ...ladybug_geometry_extension import pt_distances

# pylint: enable=E0401


@dataclass(init=True, eq=True, repr=True)
class MoistureSource:
    """An object defining where moisture is present in a spatial thermal comfort simulation.

    Args:
        identifier (str):
            The ID of this moisture source.
        magnitude (float):
            The evaporative cooling effectiveness of this moisture source.
        point_indices (list[int]):
            The point indices (related to a list of points) where this moisture source is applied.
        decay_method: (str):
            The method with which moisture effects will "drop off" at distance from the emitter.
        schedule (list[int]):
            A list of hours in the year where the moisture emitter will be active. If empty, then
            the moisture source will not be active for all hours of the year.
    """

    identifier: str
    evaporative_cooling_effect: list[float]
    decay_method: DecayMethod = DecayMethod.LINEAR
    max_decay_distance: float = 5
    plume_angle: float = 25

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.identifier})"

    def __post_init__(self):
        """_"""
        if isinstance(self.evaporative_cooling_effect, np.ndarray):
            self.evaporative_cooling_effect = self.evaporative_cooling_effect.tolist()

        if len(self.evaporative_cooling_effect) != 8760:
            raise ValueError(
                "evaporative_cooling_effect must be a list of 8760 values."
            )
        if any(not isinstance(i, (float, int))
                for i in self.evaporative_cooling_effect):
            raise ValueError(
                "evaporative_cooling_effect must be a list of numeric values."
            )

        if any(i < 0 for i in self.evaporative_cooling_effect):
            raise ValueError(
                "evaporative_cooling_effect must be a list of positive values."
            )
        if any(i > 1 for i in self.evaporative_cooling_effect):
            raise ValueError(
                "evaporative_cooling_effect must be a list of values less than or equal to 1."
            )

        self.decay_method  # pylint: disable=W0104
        # for some reason, if this is not called, the check below fails

        if not isinstance(self.decay_method, DecayMethod):
            raise ValueError("decay_method must be a DecayMethod enum value.")

        if not isinstance(self.max_decay_distance, (float, int)):
            raise ValueError("max_decay_distance must be a numeric value.")
        if self.max_decay_distance <= 0:
            raise ValueError(
                "max_decay_distance must be a positive value greater than 0."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert this object to a dictionary."""
        d = {
            "_t": "BH.oM.LadybugTools.MoistureSource",
            "Identifier": self.identifier,
            "DecayMethod": str(self.decay_method.name),
            "EvaporativeCoolingEffect": self.evaporative_cooling_effect,
            "MaxDecayDistance": self.max_decay_distance,
        }

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MoistureSource":
        """Create this object from a dictionary."""

        d = convert_keys_to_snake_case(d)

        if isinstance(d["decay_method"], str):
            d["decay_method"] = DecayMethod[d["decay_method"]]

        return cls(
            identifier=d["identifier"],
            evaporative_cooling_effect=d["evaporative_cooling_effect"],
            decay_method=d["decay_method"],
            max_decay_distance=d["max_decay_distance"],
        )

    def to_json(self) -> str:
        """Create a JSON string from this object."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_string: str) -> "MoistureSource":
        """Create this object from a JSON string."""

        return cls.from_dict(json.loads(json_string))

    def to_file(self, path: Path) -> Path:
        """Write this object to a JSON file."""

        if Path(path).suffix != ".json":
            raise ValueError("path must be a JSON file.")

        with open(Path(path), "w") as fp:
            fp.write(self.to_json())

        return Path(path)

    @classmethod
    def from_file(cls, path: Path) -> "MoistureSource":
        """Create this object from a JSON file."""

        with open(Path(path), "r") as fp:
            return cls.from_json(fp.read())

    def spatial_evaporative_cooling_effect(
        self,
        epw: EPW,
        source: Point2D | list[float],
        points: list[Point2D] | list[list[float]],
    ) -> pd.DataFrame:
        """Return a value per points_xy, giving the effective
        evaporative cooling amount per point from the source point.

        Args:
            epw (EPW):
                An EPW object.
            source (Point2D | list[float]):
                The source point. Can be a Point2D or an [X, Y] coordinate list.
            points (list[Point2D]):
                A list of points to check for downwindedness.
                Can be a list of Point2D or [[X, Y], ...] coordinate lists.

        Returns:
            pd.DataFrame:
                A dataframe containing the evaporative cooling effect for each point,
                for each hour of the year
        """

        idx = analysis_period_to_datetimes(AnalysisPeriod())

        distances = pt_distances(source, points)
        downwinds = is_point_downwind(
            epw=epw, source=source, points=points, plume_angle=self.plume_angle
        )
        ws_adj = np.interp(
            epw.wind_speed, [epw.wind_speed.min, epw.wind_speed.max], [0, 1]
        )

        all_vals = []
        pbar = tqdm(list(enumerate(zip(epw.wind_direction, ws_adj))))
        for n, (angle, speed) in pbar:
            pbar.set_description(f"Processing {self} {idx[n]:%b %d %H:%M}")
            dist = np.where(downwinds[angle], distances, np.nan)
            all_vals.append(
                proximity_decay(
                    value=self.evaporative_cooling_effect[n],
                    decay_method=self.decay_method,
                    distance_to_value=dist,
                    max_distance=self.max_decay_distance * speed,
                )[0]
            )
        return pd.DataFrame(all_vals, index=idx).fillna(0)


def is_point_downwind(
    epw: EPW,
    source: Point2D | list[float],
    points: list[Point2D] | list[list[float]],
    plume_angle: float = 25,
) -> dict[float, list[bool]]:
    """Create a dictionary containing a list of booleans for each unique wind direction, defining whether each point
    is downwind of the source point.

    Args:
        epw (EPW):
            An EPW object.
        source (Point2D | list[float]):
            The source point. Can be a Point2D or an [X, Y] coordinate list.
        points (list[Point2D]):
            A list of points to check for downwindedness. Can be a list of Point2D or [[X, Y], ...] coordinate lists.
        plume_angle (float, optional):
            The angle of the plume defining the spread of moisture. Defaults to 25 degrees.

    Returns:
        dict[float, list[bool]]:
            A dictionary of wind direction keys and a list of booleans defining whether each point is downwind of the
            source point.
    """

    if isinstance(source, list):
        if all(isinstance(i, (int, float)) for i in source):
            source = Point2D(*source)
        else:
            raise ValueError(
                "source must be a Point2D or [X, Y] coordinate list.")

    if isinstance(points, list):
        if all(isinstance(i, Point2D) for i in points):
            pass
        elif all(isinstance(i, list) for i in points):
            if all(isinstance(i, (int, float)) for i in points[0]):
                points = [Point2D(*i) for i in points]
        else:
            raise ValueError(
                "points must be a list of Point2D or [[X, Y], ...] coordinate lists."
            )

    if (plume_angle <= 0) or (plume_angle > 180):
        raise ValueError("plume_angle must be a value between 0 and 180.")

    angles = np.unique(epw.wind_direction)
    vectors = [-Vector2D(*i) for i in np.array(angle_to_vector(angles)).T]
    pts_vectors = [Vector2D(*i) for i in (np.array(source) - np.array(points))]

    in_plume = {}
    pbar = tqdm(list(zip(angles, vectors)))
    for angle, vector in pbar:
        pbar.set_description(
            f"Calculating downwindedness for {angle}° wind from {source}"
        )
        temp = []
        for pts_vector in pts_vectors:
            try:
                temp.append(
                    vector.angle(pts_vector) < np.radians(
                        plume_angle / 2))
            except ZeroDivisionError:
                temp.append(True)
        in_plume[angle] = temp

    return in_plume

    # def spatial_evaporative_cooling_effect(self, spatial_points: list[list[float]]) -> list[float]:
    #     """Return a subset of a list, containing the point X,Y locations of moisture sources."""
    #     return np.array(self.evaporative_cooling_effect)


def spatial_evaporative_cooling_effect(
    moisture_sources: list[MoistureSource],
    epw: EPW,
    sources: list[Point2D] | list[list[float]],
    points: list[Point2D] | list[list[float]],
    save_path: Path = None,
) -> pd.DataFrame:
    """Return a value per points, giving the effective
    evaporative cooling amount per point from the source point.

    Args:
        moisture_sources (list[MoistureSource]):
            A list of moisture source objects.
        epw (EPW):
            An EPW object.
        sources (list[Point2D] | list[list[float]]):
            The source points. Can be a list of Point2D or [[X, Y], ...] coordinate lists.
        points (list[Point2D] | list[list[float]]):
            A list of points to check for downwindedness. Can be a list of Point2D or [[X, Y], ...] coordinate lists.
        save_path (Path, optional):
            A path to save the results to. Defaults to None.

    Returns:
        pd.DataFrame:
            A dataframe containing the evaporative cooling effect for each point, for each hour of the year
    """
    if len(moisture_sources) != len(sources):
        raise ValueError(
            "The number of moisture sources must match the number of source points."
        )
    if save_path is not None:
        if save_path.suffix != ".parquet":
            raise ValueError("save_path must be a parquet file.")
        if save_path.exists():
            return pd.read_parquet(save_path)

    df = (
        sum(
            ms.spatial_evaporative_cooling_effect(epw=epw, source=src, points=points)
            for src, ms in zip(sources, moisture_sources)
        )
        .clip(upper=0.99)
        .astype(np.float32)
    )

    if save_path is not None:
        df.to_parquet(save_path, compression="gzip")

    return df
