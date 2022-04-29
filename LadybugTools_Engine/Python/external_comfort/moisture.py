from __future__ import annotations

from dataclasses import dataclass, field
from os import stat
from pathlib import Path
from threading import Thread
from typing import Dict, List, Union
import warnings
import shapely.geometry
import shapely.ops
import shapely.affinity
import json
import numpy as np
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug.psychrometrics import wet_bulb_from_db_rh
from ladybug.datatype.temperature import WetBulbTemperature
from ladybug_extension.datacollection import to_series
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
    
    @property
    def pathsafe_id(self) -> str:
        return self.id.replace('*', '').replace(':', '').replace('/', '').replace('\\', '').replace('?', '').replace('\"', '').replace('>', '').replace('<', '').replace('|', '')
    
    def __repr__(self) -> str:
        return f"MoistureSource(id={self.id})"

def evaporative_cooling_effect(dry_bulb_temperature: float, relative_humidity: float, evaporative_cooling_effectiveness: float, atmospheric_pressure: float = None) -> List[float]:
    """For the inputs, calculate the effective DBT and RH values for the evaporative cooling effectiveness given.

    Args:
        dry_bulb_temperature (float): A dry bulb temperature in degrees Celsius.
        relative_humidity (float): A relative humidity in percent (0-100).
        evaporative_cooling_effectiveness (float): The evaporative cooling effectiveness. Defaults to 0.3.
        atmospheric_pressure (float, optional): A pressure in Pa.

    Returns:
        List[float]: A list of two values for the effective dry bulb temperature and relative humidity.
    """
    wet_bulb_temperature = wet_bulb_from_db_rh(dry_bulb_temperature, relative_humidity, atmospheric_pressure)

    return [
        dry_bulb_temperature - ((dry_bulb_temperature - wet_bulb_temperature) * evaporative_cooling_effectiveness),
        (relative_humidity * (1 - evaporative_cooling_effectiveness)) + evaporative_cooling_effectiveness * 100
    ]

def evaporative_cooling_effect_collection(
    epw: EPW, evaporative_cooling_effectiveness: float = 0.3
) -> List[HourlyContinuousCollection]:
    """Calculate the effective DBT and RH considering effects of evaporative cooling.

    Args:
        epw (EPW): A ladybug EPW object.
        evaporative_cooling_effectiveness (float, optional): The proportion of difference betwen DBT and WBT by which to adjust DBT. Defaults to 0.3 which equates to 30% effective evaporative cooling, roughly that of Misting.

    Returns:
        HourlyContinuousCollection: An adjusted dry-bulb temperature collection with evaporative cooling factored in.
    """

    if (evaporative_cooling_effectiveness > 1) or (evaporative_cooling_effectiveness < 0):
        raise ValueError("evaporative_cooling_effectiveness must be between 0 and 1.")

    wbt = HourlyContinuousCollection.compute_function_aligned(
        wet_bulb_from_db_rh,
        [epw.dry_bulb_temperature, epw.relative_humidity, epw.atmospheric_station_pressure],
        WetBulbTemperature(),
        "C"
    )
    dbt = epw.dry_bulb_temperature.duplicate()
    dbt = dbt - ((dbt - wbt) * evaporative_cooling_effectiveness)
    dbt.header.metadata["evaporative_cooling"] = f"{evaporative_cooling_effectiveness:0.0%}"

    rh = epw.relative_humidity.duplicate()
    rh = (rh * (1 - evaporative_cooling_effectiveness)) + (evaporative_cooling_effectiveness * 100)
    rh.header.metadata["evaporative_cooling"] = f"{evaporative_cooling_effectiveness:0.0%}"

    return [dbt, rh]

#################################################


def frame_to_multipoint(df: pd.DataFrame) -> shapely.geometry.MultiPoint:
    """Convert a pandas DataFrame containing x, y, z columns to a shapely MultiPoint.

    Args:
        df (pd.DataFrame): A "points" dataframe

    Returns:
        MultiPoint: A shapely MultiPoint object.
    """
    return shapely.geometry.MultiPoint([shapely.geometry.Point(i.x, i.y) for _, i in df.iterrows()])

def multipoint_to_frame(multipoint: shapely.geometry.MultiPoint) -> pd.DataFrame:
    """Convert a shapely MultiPoint object to a pandas DataFrame.

    Args:
        multipoint (shapely.geometry.MultiPoint): A shapely MultiPoint object.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the x, y coordinates of the points in the multipoint.
    """        
    return pd.DataFrame([[j[0] for j in i.xy] for i in multipoint.geoms], columns=["x", "y"])

def create_plume(wind_speed: float, wind_direction: float, boundary_layers: List[float], origin: shapely.geometry.Point = shapely.geometry.Point(0, 0)) -> shapely.geometry.MultiPolygon:
    """Create a plume based on wind speed, direction and number and size of boundary layers.

    Args:
        wind_speed (float): The wind speed, by which to stretch the plume.
        wind_direction (float): The wind direction by which to orient the plume
        boundary_layers (List[float]): The radius of each plume boundary layer.

    Returns:
        shapely.geometry.MultiPolygon: A MultiPolygon object representing the plume boundary layer/s.
    """

    if min(boundary_layers) == 0:
        raise ValueError("Boundary layers must be greater than zero.")
    
    if not all(i < j for i, j in zip(boundary_layers, boundary_layers[1:])):
        raise ValueError("Boundary layers must be in ascending order.")
    
    plume_base = shapely.geometry.MultiPolygon([origin.buffer(pb) for pb in boundary_layers])

    if wind_speed == 0:
        plume_regions = []
        for n, g in enumerate(plume_base.geoms):
            if n == 0:
                plume_regions.append(g)
            else:
                plume_regions.append(g.difference(plume_regions[n-1]))
        return shapely.geometry.MultiPolygon(plume_regions)
    
    splitter = shapely.geometry.LineString([shapely.affinity.translate(origin, xoff=max(boundary_layers)), shapely.affinity.translate(origin, xoff=-max(boundary_layers))])
    plume_regions = []
    for n, g in enumerate(plume_base.geoms):
        downwind, upwind = shapely.ops.SplitOp.split(g, splitter).geoms
        unioned = shapely.affinity.rotate(shapely.ops.unary_union([shapely.affinity.scale(downwind, yfact=1 + (wind_speed * boundary_layers[n]), origin=origin), upwind]), -wind_direction, origin=origin)
        if n == 0:
            plume_regions.append(unioned)
        else:
            plume_regions.append(unioned.difference(plume_regions[n-1]))
    return shapely.geometry.MultiPolygon(plume_regions)

def move_geom_to_points(geom: shapely.geometry.MultiPolygon, points: shapely.geometry.MultiPoint) -> List[shapely.geometry.MultiPolygon]:
    """Move a MultiPolygon to each location within a MultiPoint.

    Args:
        geom (shapely.geometry.MultiPolygon): A MultiPolygon object.
        points (shapely.geometry.MultiPoint): A MultiPoint object.

    Returns:
        List[shapely.geometry.MultiPolygon]: A list of MultiPolygon objects.
    """
    return [shapely.affinity.translate(geom, xoff=i.x, yoff=i.y) for i in points.geoms]

def create_wake(plume: shapely.geometry.MultiPolygon, points: shapely.geometry.MultiPoint) -> List[shapely.geometry.MultiPolygon]:
    """Create a wake, constituting a joined set of plume boundary layers, at a set of points.

    Args:
        plume (shapely.geometry.MultiPolygon): A plume containing boundary layers.
        points (shapely.geometry.MultiPoint): Points where the plume should be moved and then joined.

    Returns:
        List[shapely.geometry.MultiPolygon]: A list of MultiPolygon objects representing the wake.
    """    
    plume_regions = []
    for n, g in enumerate(plume.geoms):
        unioned = shapely.ops.unary_union([shapely.affinity.translate(g, xoff=i.x, yoff=i.y) for i in points.geoms])
        # if not isinstance(unioned, shapely.geometry.MultiPolygon):
        #     unioned = shapely.geometry.MultiPolygon([unioned])
        if n == 0:
            plume_regions.append(unioned)
        else:
            plume_regions.append(shapely.geometry.MultiPolygon(unioned.difference(plume_regions[n-1])))
    return plume_regions

def points_within_wake(wake: List[shapely.geometry.MultiPolygon], test_points: shapely.geometry.MultiPoint, all_points: Union[pd.DataFrame, shapely.geometry.MultiPoint]) -> List[List[int]]:
    """Test whether points exist within a shapely MultiPolygon wake and return a list of booleans.

    Args:
        wake (List[shapely.geometry.MultiPolygon]): The wake for point containment.
        test_points (shapely.geometry.MultiPoint): The points to test for containment.
        all_points (Union[pd.DataFrame, shapely.geometry.MultiPoint]): The larger group of points, where inclusion indices are based on.

    Returns:
        List[List[int]]: A list of lists of integers, where each list corresponds to a wake, and each sub-list corresponds to the point integers included from the "all_points" input.
    """

    if isinstance(all_points, shapely.geometry.MultiPoint):
        all_points_df = multipoint_to_frame(all_points)
        # all_points_mp = all_points
    elif isinstance(all_points, pd.DataFrame):
        all_points_df = all_points
        # all_points_mp = frame_to_multipoint(all_points)

    point_indices = []
    for w in wake:
        sub_points = test_points.intersection(w)
        wake_point_indices = pd.merge(
            all_points_df.reset_index(), 
            multipoint_to_frame(sub_points), 
            how='inner', on=['x', "y"])["index"].values.tolist()
        point_indices.append(wake_point_indices)
    return point_indices

def wake_indices(moisture_source: MoistureSource, epw: EPW, all_points: Union[pd.DataFrame, shapely.geometry.MultiPoint], boundary_layers: List[float], output_directory: Path = None) -> Dict[str, List[List[int]]]:

    """For a given EPW file, create a boolear lookup for each wind speed/direction combination for use in generating a moisture adjustment matrix.

    Args:
        moisture_source (MoistureSource): A MoistureSource object.
        epw (EPW): A ladybug EPW object.
        all_points (Union[pd.DataFrame, shapely.geometry.MultiPoint]): A pandas DataFrame or shapely MultiPoint object containing all the points to be used in the analysis.
        boundary_layers (List[float]): A list of floats representing the boundary layers of the plume.
        output_directory (Path, optional): A directoy in which to save results if given. Defaults to None.

    Returns:
        List[Dict[str, List[List[int]]]]: _description_
    """

    output_file: Path = None
    if output_directory:
        output_directory.mkdir(exist_ok=True, parents=True)
        output_file = output_directory / f"wake_indices_{moisture_source.pathsafe_id}.json"
        if output_file.exists():
            print("- Loading moisture indices from file.")
            with open(output_file, 'r') as f:
                return json.load(f)
    
    # Get the list of unique wind_speed/wind_direction combinations
    wind_speeds = to_series(epw.wind_speed)
    wind_directions = to_series(epw.wind_direction)
    ws_wd_unique = pd.concat([wind_speeds, wind_directions], axis=1).drop_duplicates().values

    if isinstance(all_points, shapely.geometry.MultiPoint):
        all_points_mp = all_points
        all_points_df = multipoint_to_frame(all_points)
    else:
        all_points_mp = frame_to_multipoint(all_points)
        all_points_df = all_points
    
    # get associated moisture source pts
    test_points_mp = shapely.geometry.MultiPoint([all_points_mp.geoms[i] for i in moisture_source.point_indices])

    def worker(n, n_iterations, wind_speed, wind_direction, boundary_layers, test_points_mp, all_points_mp, all_points_df) -> Dict[str, List[List[int]]]:
        print("", end="\r")
        print(f"- Calculating moisture indices - {n/n_iterations:02.2%})", end="\r")
        key = f"{wind_speed}, {wind_direction}"
        plume = create_plume(wind_speed, wind_direction, boundary_layers)
        wake = create_wake(plume, test_points_mp)
        return {key: points_within_wake(wake, all_points_mp, all_points_df)}
    
    # iterate through year, and construct dict for each unique wd, ws, containing the points included in each wake-level
    n_iterations = len(epw.dry_bulb_temperature)
    results = []
    # with ThreadPoolExecutor() as executor:
    #     for n, (wind_speed, wind_direction) in enumerate(ws_wd_unique[0:n_iterations]):
    #         results.append(
    #             executor.submit(worker, n, n_iterations, wind_speed, wind_direction, boundary_layers, test_points_mp, all_points_mp, all_points_df)
    #         )
    # point_in_wake_lookup = {k: v for d in [x.result() for x in results] for k, v in d.items()}
    for n, (wind_speed, wind_direction) in enumerate(ws_wd_unique[0:n_iterations]):
        results.append(
            worker(n, n_iterations, wind_speed, wind_direction, boundary_layers, test_points_mp, all_points_mp, all_points_df)
        )
    point_in_wake_lookup = {k: v for d in results for k, v in d.items()}

    if output_file:
        with open(output_file, "w") as f:
            json.dump(point_in_wake_lookup, f)

    return point_in_wake_lookup

def annual_moisture_effectiveness_matrix(moisture_sources: List[MoistureSource], epw: EPW, all_points: Union[pd.DataFrame, shapely.geometry.MultiPoint], boundary_layers: List[float], boundary_layer_effectiveness: List[float], output_directory: Path = None) -> pd.DataFrame:
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
        raise ValueError("boundary_layer_effectiveness must be the same length as boundary_layers")
    if not all(i > j for i, j in zip(boundary_layer_effectiveness, boundary_layer_effectiveness[1:])):
        raise ValueError("Boundary layer effectiveness must be given in decreasing order.")
    
    if isinstance(all_points, shapely.geometry.MultiPoint):
        all_points_mp = all_points
        all_points_df = multipoint_to_frame(all_points)
    else:
        all_points_mp = frame_to_multipoint(all_points)
        all_points_df = all_points
    
    temp_1 = []
    for moisture_source in moisture_sources:
        print(f"- Calculating annual hourly moisture matrix for {moisture_source.pathsafe_id}")
        hourly_point_indices = wake_indices(moisture_source, epw, all_points_mp, boundary_layers, output_directory)

        temp_0 = []
        for hour in range(len(epw.dry_bulb_temperature)):
            key = f"{epw.wind_speed[hour]}, {epw.wind_direction[hour]}"
            pt_indices = hourly_point_indices[key]
            temp = []
            for n_wake_level, wake_level in enumerate(pt_indices):
                temp.append(np.where(np.isin(range(len(all_points_df)), wake_level), moisture_source.magnitude * boundary_layer_effectiveness[n_wake_level], 0))
            temp_0.append(temp)
        temp_0 = np.swapaxes(temp_0, 0, 1)
        temp_1.append(temp_0)
    temp = pd.DataFrame(np.amax(np.amax(temp_1, axis=0), axis=0), index=to_series(epw.dry_bulb_temperature).index)

    return temp