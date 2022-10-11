from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.helpers.angle_from_north import angle_from_north
from ladybugtools_toolkit.helpers.proximity_decay import proximity_decay
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import \
    to_series
from ladybugtools_toolkit.ladybug_extension.epw.unique_wind_speed_direction import \
    unique_wind_speed_direction
from scipy.spatial.distance import cdist


class MoistureSource:
    """An object defining where moisture is present in a spatial thermal comfort simulation

    Args:
        identifier (str):
            The ID of this moisture source.
        magnitude (float):
            The evaporative cooling effectiveness of this moisture source.
        point_indices (List[int]):
            The point indices (related to a list of points) where this moisture source is applied.
        decay_function: (str):
            The method with which moisture effects will "drop off" at distance from the emitter.
        schedule (List[int]):
            A list of hours in the year where the moisture emitter will be active. If empty, then
            the moisture source will not be active for all hours of the year.
    """

    def __init__(
        self,
        identifier: str,
        magnitude: float,
        point_indices: List[int],
        decay_function: str,
        schedule: List[int],
    ) -> MoistureSource:
        self.identifier = identifier
        self.magnitude = magnitude
        self.point_indices = np.array(point_indices)
        self.decay_function = decay_function
        if len(schedule) == 0:
            schedule = np.arange(0, 8760, 1)
        self.schedule = np.array(schedule)

    @classmethod
    def from_json(cls, json_file: Path) -> List[MoistureSource]:
        """Create a set of MoistureSource objects from a json file.

        Args:
            json_file (Path): A file containing MoistureSource objects in json format.

        Returns:
            List[MoistureSource]: A list of MoistureSource objects.
        """

        with open(json_file, "r", encoding="utf-8") as fp:
            moisture_sources_dict = json.load(fp)

        objs = []
        for moisture_source_dict in moisture_sources_dict:
            objs.append(cls.from_dict(moisture_source_dict))

        return objs

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> MoistureSource:
        """Create a MoistureSource object from a dict.

        Args:
            dict (Dict[str, Any]): _description_

        Returns:
            MoistureSource: _description_
        """
        return cls(
            identifier=dictionary["id"],
            magnitude=dictionary["magnitude"],
            point_indices=dictionary["point_indices"],
            decay_function=dictionary["decay_function"],
            schedule=dictionary["schedule"],
        )

    def __repr__(self) -> str:
        return f"MoistureSource(id={self.identifier})"

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        return {
            "id": self.identifier,
            "magnitude": self.magnitude,
            "point_indices": self.point_indices,
            "decay_function": self.decay_function,
            "schedule": self.schedule,
        }

    def moisture_source_points(
        self, spatial_points: List[List[float]]
    ) -> List[List[float]]:
        """Return a subset of a list, containing the point X,Y locations of moisture sources."""
        return np.array(spatial_points[self.point_indices])

    def point_distances(self, spatial_points: List[List[float]]) -> List[List[float]]:
        """Return the distance from each moisture_source pt to each other point in the input list"""

        # get the domain and source points
        spatial_points = np.array(spatial_points)
        emitter_points = self.moisture_source_points(spatial_points)

        # for each emitter, get the distance to all "receiving" points
        distances = cdist(emitter_points, spatial_points)

        return distances

    def point_angles(self, spatial_points: List[List[float]]) -> List[List[float]]:
        """Return the relative north angle from each moisture_source pt to each other point in the
        input list"""

        # get the domain and source points
        spatial_points = np.array(spatial_points)
        emitter_points = self.moisture_source_points(spatial_points)

        # calculate the vector between each point
        vectors = emitter_points[:, np.newaxis] - spatial_points

        # calculate angle to north for each vector
        angles = np.array([angle_from_north(i.T).T for i in vectors])

        return angles

    def plume(
        self,
        point_distances: List[List[float]],
        point_angles: List[List[float]],
        wind_speed: float,
        wind_direction: float,
        plume_width: float = 25,
    ) -> List[float]:
        """Calculate the spatial moisture plume from a single location to other locations under
            given wind speed and direction.

        Args:
            point_distances (List[List[float]]): An array of distances to other points.
            point_angles (List[List[float]]): An array of angles to other points.
            wind_speed (float): A value describing current wind speed.
            wind_direction (float): A value describing current wind direction
            plume_width (float, optional): The spread of the plume to be generated. Defaults to 25.

        Returns:
            List[float]: _description_
        """

        # get downwind angles
        if (wind_direction > 360 - (plume_width / 2)) or (
            wind_direction < (0 + (plume_width / 2))
        ):
            downwind = (point_angles > 360 - (plume_width / 2)) | (
                point_angles < (0 + (plume_width / 2))
            )
        else:
            downwind = (point_angles > wind_direction - (plume_width / 2)) & (
                point_angles < wind_direction + (plume_width / 2)
            )

        # calculate proximity decayed values
        mag_dist_values = proximity_decay(
            self.magnitude, point_distances, 2 * wind_speed, self.decay_function
        )

        # get overall magnitude based on down-winded-ness
        return np.where(downwind, mag_dist_values, 0).max(axis=0)

    def spatial_moisture(
        self,
        spatial_points: List[List[float]],
        epw: EPW,
        plume_width: float = 25,
        simulation_directory: Path = None,
    ) -> Dict[Tuple(float), List[List[float]]]:

        if simulation_directory is not None:
            output_dir = simulation_directory / "moisture"
            if (output_dir / f"{self.pathsafe_id}_matrix.parquet").exists():
                print(
                    f"[{simulation_directory.stem}] - Loading moisture effectiveness data for {self.identifier}"
                )
                moisture_df = pd.read_parquet(
                    output_dir / f"{self.pathsafe_id}_matrix.parquet"
                )
                moisture_df.columns = moisture_df.columns.astype(int)
                return moisture_df

        # get point distances
        pt_distances = self.point_distances(spatial_points)

        # get point angles
        pt_angles = self.point_angles(spatial_points)

        # get unique ws/wd
        unique_ws_wd = unique_wind_speed_direction(epw, self.schedule)

        # construct a dict ready for results
        d = {}
        for ws, wd in unique_ws_wd:
            d[(ws, wd)] = self.plume(pt_distances, pt_angles, ws, wd, plume_width)

        # for each time period in the year construct the resultant moisture matrix
        moisture_matrix = []
        for n, (ws, wd) in enumerate(list(zip(*[epw.wind_speed, epw.wind_direction]))):
            print(
                f"[{simulation_directory.stem}] - Calculating moisture effectiveness data for {self.identifier} ({n/8760:03.2%})",
                end="\r",
            )
            if n not in self.schedule:
                moisture_matrix.append(np.zeros_like(pt_distances[0]))
            else:
                moisture_matrix.append(d[(ws, wd)])

        moisture_df = pd.DataFrame(
            moisture_matrix, index=to_series(epw.dry_bulb_temperature).index
        )

        if simulation_directory is not None:
            output_dir = Path(simulation_directory) / "moisture"
            moisture_df.columns = moisture_df.columns.astype(str)
            moisture_df.to_parquet(output_dir / f"{self.pathsafe_id}_matrix.parquet")

        return moisture_df

    @property
    def pathsafe_id(self) -> str:
        """Create a path-safe ID for this object"""
        return "".join(
            [c for c in self.identifier if c.isalpha() or c.isdigit() or c == " "]
        ).rstrip()
