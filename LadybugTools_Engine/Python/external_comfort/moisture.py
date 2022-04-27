from __future__ import annotations

from dataclasses import dataclass, field
from os import stat
from pathlib import Path
from typing import List, Union
import warnings
import shapely.geometry
import shapely.ops
import shapely.affinity
import json
import numpy as np
from ladybug.epw import EPW
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed



@dataclass()
class MoistureSource:
    id: str = field(init=True, repr=True)
    magnitude: float = field(init=True, repr=False)
    point_indices: List[int] = field(init=True, repr=False)

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
            objs.append(MoistureSource(ws["id"], ws["magnitude"], ws["point_indices"]))
        return objs
    
    @staticmethod
    def frame_to_multipoint(df: pd.DataFrame) -> List[shapely.geometry.MultiPoint]:
        """Convert a pandas DataFrame containing a header row and x, y, z columns to a shapely MultiPoint.

        Args:
            df (pd.DataFrame): A "points" dataframe

        Returns:
            List[MultiPoint]: A list of shapely MultiPoint object (one for each column grouping in the dataframe).
        """
        points = []
        if isinstance(df.columns, pd.MultiIndex):
            if len(df.columns[0]) > 2:
                raise ValueError("The dataframe must have a multiindex header with only two levels, the first being teh point group, and the second the point values.")
            
            for grp in df.columns.get_level_values(0).unique():
                points.append(shapely.geometry.MultiPoint([shapely.geometry.Point(i.x, i.y) for _, i in df[grp].iterrows()]))
        else:
            points.append(shapely.geometry.MultiPoint([shapely.geometry.Point(i.x, i.y) for _, i in df.iterrows()]))
        return points
    
    @staticmethod
    def multipoint_to_frame(multipoint: shapely.geometry.MultiPoint) -> pd.DataFrame:
        """Convert a shapely MultiPoint object to a pandas DataFrame.

        Args:
            multipoint (shapely.geometry.MultiPoint): A shapely MultiPoint object.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the x, y coordinates of the points in the multipoint.
        """        
        return pd.DataFrame([[j[0] for j in i.xy] for i in multipoint.geoms], columns=["x", "y"])

    def create_moisture_points(self, df: pd.DataFrame) -> shapely.geometry.MultiPoint:
        """Create a shapely MultiPoint object denoting the points where a moisture source is located.

        Args:
            df (pd.DataFrame): A "points" dataframe

        Returns:
            shapely.geometry.MultiPoint: A shapely MultiPoint object.
        """
        all_points = MoistureSource.frame_to_multipoint(df)
        return shapely.geometry.MultiPoint([all_points[0].geoms[i] for i in self.point_indices])

    @staticmethod
    def create_wakes(buffer_distances: List[float], wind_direction: float, wind_speed: float, points: shapely.geometry.MultiPoint = None) -> List[shapely.geometry.MultiPolygon]:
        """For a given set of buffer distances around a moisture source, create a set of wake polygons showing effects of that moisture downwind.

        Args:
            buffer_distances (List[float]): Distances for effectiveness layers of the wake.
            wind_direction (float): The direction in which the wake will be pulled
            wind_speed (float): The speed applied to the wake, stretching it out in the direction of the wind.
            point (shapely.geometry.MultiPoint, optional): A set of points to which the wake polygons will be moved nad merged if given.

        Returns:
            shapely.geometry.MultiPolygon: The 
        """        
        
        if min(buffer_distances) == 0:
            raise ValueError("Buffer distances must be greater than zero.")
        if not all(i < j for i, j in zip(buffer_distances, buffer_distances[1:])):
            raise ValueError("Buffer distances must be in ascending order.")
        
        base_wake = []
        base_pt = shapely.geometry.Point(0, 0)
        for buffer_distance in buffer_distances:
            if wind_speed == 0:
                base_wake.append(base_pt.buffer(buffer_distance))
            else:
                splitter = shapely.geometry.LineString([
                    shapely.affinity.translate(base_pt, xoff=buffer_distance), 
                    shapely.affinity.translate(base_pt, xoff=-buffer_distance)
                ])
                downwind, upwind = shapely.ops.SplitOp.split(base_pt.buffer(buffer_distance), splitter).geoms
                scaled_downwind = shapely.affinity.scale(downwind, yfact=1 + (wind_speed * buffer_distance), origin=base_pt)
                joined = shapely.ops.unary_union([upwind, scaled_downwind])
                joined_rotated = shapely.affinity.rotate(joined, -wind_direction, origin=base_pt)
                base_wake.append(joined_rotated)
        base_wake = shapely.geometry.MultiPolygon(base_wake)

        wake_regions = []
        if points:
            # Move wakes to each point location, and merge into single shape, then subtract more signifigant wakes
            translated_wakes = np.array([shapely.affinity.translate(base_wake, xoff=i.x, yoff=i.y).geoms for i in points.geoms]).T
            unioned_wakes = [shapely.ops.unary_union(wake) for wake in translated_wakes]
            for n, g in enumerate(unioned_wakes):
                if n == 0:
                    wake_regions.append(g)
                else:
                    wake_regions.append(g.difference(unioned_wakes[n-1]))
        else:
            for n, g in enumerate(base_wake.geoms):
                wake_regions.append(g if n == 0 else g.difference(base_wake.geoms[n-1]))
        
        return wake_regions
    
    def get_wake_point_indices(self, all_points_df: pd.DataFrame, buffer_distances, wind_direction: float, wind_speed: float) -> List[int]:
        """Get the indices of the points in the dataframe that are within the wake of a given moisture source.

        Args:
            df (pd.DataFrame): A "points" dataframe
            buffer_distances (List[float]): Distances for effectiveness layers of the wake.
            wind_direction (float): The direction in which the wake will be pulled
            wind_speed (float): The speed applied to the wake, stretching it out in the direction of the wind.
            points (shapely.geometry.MultiPoint, optional): A set of points to which the wake polygons will be moved and merged if given.

        Returns:
            List[int]: A list of indices of the points within the wake.
        """

        warnings.warn("This process currently only works for single analysis grid set-up!")

        all_points = self.frame_to_multipoint(all_points_df)[0]
        moisture_source_points = shapely.geometry.MultiPoint([all_points.geoms[i] for i in self.point_indices])
        wakes = self.create_wakes(buffer_distances, wind_direction, wind_speed, moisture_source_points)

        ll = []
        for wake in wakes:
            sub_points = all_points.intersection(wake)

            contained_pt_indices = pd.merge(
                self.multipoint_to_frame(all_points).reset_index(), 
                self.multipoint_to_frame(sub_points), 
                how='inner', on=['x', "y"])["index"].values.tolist()
            ll.append(contained_pt_indices)

        return ll

    def annual_modifiable_indices(self, epw: EPW, all_points_df: pd.DataFrame, buffer_distances: List[float] = [0.33, 1.2]) -> List[List[int]]:

        def worker(idx):
            print("", end="\r")
            print(f"{idx/8769:0.2%}", end="\r")
            return self.get_wake_point_indices(all_points_df, buffer_distances, epw.wind_direction[idx], epw.wind_speed[idx])

        # with ThreadPoolExecutor() as executor:
        #     ll = list(executor.map(worker, range(3)))

        ll = []
        for idx in range(8760):
            ll.append(worker(idx))

        return ll
