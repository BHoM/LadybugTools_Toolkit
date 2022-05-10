from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from cached_property import cached_property
from ladybug.datatype.temperature import WetBulbTemperature
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug.psychrometrics import wet_bulb_from_db_rh
from ladybug_extension.datacollection import to_series


@dataclass()
class MoistureSource:
    id: str = field(init=True, repr=True)
    magnitude: float = field(init=True, repr=False)
    points: List[int] = field(init=True, repr=False)

    @classmethod
    def from_json(cls, json_file: Path) -> List[MoistureSource]:
        """Create a set of MoistureSource objects from a json file.

        Args:
            json_file (Path): A file containing MoistureSource objects in json format.

        Returns:
            List[MoistureSource]: A list of MoistureSource objects.
        """
        objs = []
        with open(json_file, "r") as fp:
            water_sources = json.load(fp)
        for ws in water_sources:
            objs.append(
                MoistureSource(
                    ws["id"], ws["magnitude"], np.array(ws["points"]).astype(np.float16)
                )
            )
        return objs

    @property
    def pathsafe_id(self) -> str:
        return (
            self.id.replace("*", "")
            .replace(":", "")
            .replace("/", "")
            .replace("\\", "")
            .replace("?", "")
            .replace('"', "")
            .replace(">", "")
            .replace("<", "")
            .replace("|", "")
        )

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
        """Return a list of x,y coordinates for each point in the moisture source."""
        return self.points[:, :2]


def evaporative_cooling_effect(
    dry_bulb_temperature: float,
    relative_humidity: float,
    evaporative_cooling_effectiveness: float,
    atmospheric_pressure: float = None,
) -> List[float]:
    """For the inputs, calculate the effective DBT and RH values for the evaporative cooling effectiveness given.

    Args:
        dry_bulb_temperature (float): A dry bulb temperature in degrees Celsius.
        relative_humidity (float): A relative humidity in percent (0-100).
        evaporative_cooling_effectiveness (float): The evaporative cooling effectiveness. Defaults to 0.3.
        atmospheric_pressure (float, optional): A pressure in Pa.

    Returns:
        List[float]: A list of two values for the effective dry bulb temperature and relative humidity.
    """
    wet_bulb_temperature = wet_bulb_from_db_rh(
        dry_bulb_temperature, relative_humidity, atmospheric_pressure
    )

    return [
        dry_bulb_temperature
        - (
            (dry_bulb_temperature - wet_bulb_temperature)
            * evaporative_cooling_effectiveness
        ),
        (relative_humidity * (1 - evaporative_cooling_effectiveness))
        + evaporative_cooling_effectiveness * 100,
    ]


def evaporative_cooling_effect_collection(
    epw: EPW, evaporative_cooling_effectiveness: float = 0.3
) -> List[HourlyContinuousCollection]:
    """Calculate the effective DBT and RH considering effects of evaporative cooling.

    Args:
        epw (EPW): A ladybug EPW object.
        evaporative_cooling_effectiveness (float, optional): The proportion of difference betwen DBT and WBT by which to adjust DBT. Defaults to 0.3 which equates to 30% effective evaporative cooling, roughly that of Misting.

    Returns:
        List[HourlyContinuousCollection]: Adjusted dry-bulb temperature and relative humidity collections incorporating evaporative cooling effect.
    """

    if (evaporative_cooling_effectiveness > 1) or (
        evaporative_cooling_effectiveness < 0
    ):
        raise ValueError("evaporative_cooling_effectiveness must be between 0 and 1.")

    wbt = HourlyContinuousCollection.compute_function_aligned(
        wet_bulb_from_db_rh,
        [
            epw.dry_bulb_temperature,
            epw.relative_humidity,
            epw.atmospheric_station_pressure,
        ],
        WetBulbTemperature(),
        "C",
    )
    dbt = epw.dry_bulb_temperature.duplicate()
    dbt = dbt - ((dbt - wbt) * evaporative_cooling_effectiveness)
    dbt.header.metadata[
        "evaporative_cooling"
    ] = f"{evaporative_cooling_effectiveness:0.0%}"

    rh = epw.relative_humidity.duplicate()
    rh = (rh * (1 - evaporative_cooling_effectiveness)) + (
        evaporative_cooling_effectiveness * 100
    )
    rh.header.metadata[
        "evaporative_cooling"
    ] = f"{evaporative_cooling_effectiveness:0.0%}"

    return [dbt, rh]

    #################################################

    """For a given EPW file, create a moisture adjustment matrix for a given set of points based on their inclusion within a moisture source wake/plume.

    Args:
        moisture_source (MoistureSource): A MoistureSource object.
        epw (EPW): A ladybug EPW object.
        all_points (Union[pd.DataFrame, shapely.geometry.MultiPoint]): A pandas DataFrame or shapely MultiPoint object containing all the points to be used in the analysis.
        boundary_layers (List[float]): A list of floats representing the boundary layers of the plume.
        boundary_layer_effectiveness (List[float]): A list of floats representing the effectiveness of the boundary layers of the plume.
        output_directory (Path, optional): A directoy in which to save results if given. Defaults to None.

    Returns:
        List[Dict[str, List[List[int]]]]: _description_
    """

    if len(boundary_layer_effectiveness) != len(boundary_layers):
        raise ValueError(
            "boundary_layer_effectiveness must be the same length as boundary_layers"
        )
    if not all(
        i > j
        for i, j in zip(boundary_layer_effectiveness, boundary_layer_effectiveness[1:])
    ):
        raise ValueError(
            "Boundary layer effectiveness must be given in decreasing order."
        )

    if isinstance(all_points, shapely.geometry.MultiPoint):
        all_points_mp = all_points
        all_points_df = multipoint_to_frame(all_points)
    else:
        all_points_mp = frame_to_multipoint(all_points)
        all_points_df = all_points

    temp_1 = []
    for moisture_source in moisture_sources:
        print(
            f"- Calculating annual hourly moisture matrix for {moisture_source.pathsafe_id}"
        )
        hourly_point_indices = wake_indices(
            moisture_source, epw, all_points_mp, boundary_layers, output_directory
        )

        temp_0 = []
        for hour in range(len(epw.dry_bulb_temperature)):
            key = f"{epw.wind_speed[hour]}, {epw.wind_direction[hour]}"
            pt_indices = hourly_point_indices[key]
            temp = []
            for n_wake_level, wake_level in enumerate(pt_indices):
                temp.append(
                    np.where(
                        np.isin(range(len(all_points_df)), wake_level),
                        moisture_source.magnitude
                        * boundary_layer_effectiveness[n_wake_level],
                        0,
                    )
                )
            temp_0.append(temp)
        temp_0 = np.swapaxes(temp_0, 0, 1)
        temp_1.append(temp_0)
    temp = pd.DataFrame(
        np.amax(np.amax(temp_1, axis=0), axis=0),
        index=to_series(epw.dry_bulb_temperature).index,
    )

    return temp
