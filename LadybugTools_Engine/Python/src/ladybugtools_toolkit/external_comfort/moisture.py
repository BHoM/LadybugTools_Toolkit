import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.helpers import (
    CONSOLE_LOGGER,
    HourlyContinuousCollection,
    WetBulbTemperature,
    evaporative_cooling_effect,
    evaporative_cooling_effect_collection,
    wet_bulb_from_db_rh,
)
from ladybugtools_toolkit.ladybug_extension.analysisperiod import (
    AnalysisPeriod,
    analysis_period_to_datetimes,
)
from scipy.spatial.distance import cdist
from tqdm import tqdm

WIND_DIRECTIONS = np.arange(0, 360, 10)
SPREAD = 30  # degrees

np.random.seed(4)


if __name__ == "__main__":

    # set the directory where outputs will be stored
    moisture_dir = Path(r"C:\Users\tgerrish\Downloads\atlas_moisture")
    moisture_dir.mkdir(exist_ok=True, parents=True)

    # load EPW
    CONSOLE_LOGGER.info("Loading EPW file")
    epw_file = Path(
        r"C:\Users\tgerrish\Buro Happold\P064429 Project Atlas - Microclimate\weather_data\SHDR_TMYws_2030_RCP4.5_50_v2024.epw"
    )
    epw = EPW(epw_file)

    # load pts data
    CONSOLE_LOGGER.info("Loading PTS file")
    pts_file = Path(
        r"C:\Users\tgerrish\Buro Happold\P064429 Project Atlas - 02_Documents\03_Models\Simulation\20250117_ProjectAtlas\points.parquet"
    )
    pts = pd.read_parquet(pts_file)
    pts_xy = pts[["x", "y"]].values.astype(np.float16)

    # load moisture data
    CONSOLE_LOGGER.info("Loading moisture file")
    moisture_file = Path(
        r"C:\Users\tgerrish\Buro Happold\P064429 Project Atlas - 02_Documents\03_Models\Simulation\20250117_ProjectAtlas\moisture\sources.json"
    )
    with open(moisture_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    src_evap_clg_xy = np.array([[i["x"], i["y"]] for i in data], dtype=np.float16)
    src_evap_clg_effect = np.array(
        [i["evaporative_cooling_effectiveness"] for i in data], dtype=np.float16
    )
    src_evap_clg_dispersion = np.array(
        [i["evaporative_cooling_dispersion_factor"] for i in data], dtype=np.float16
    )

    # # TODO temporary filter for debugging over a smaller set of moisture pts
    # sample_idx = np.random.randint(0, len(src_evap_clg_xy), 50)
    # src_evap_clg_xy = src_evap_clg_xy[sample_idx]
    # src_evap_clg_effect = src_evap_clg_effect[sample_idx]
    # src_evap_clg_dispersion = src_evap_clg_dispersion[sample_idx]
    # # TODO end

    # calculate distances between sensors and moisture sources
    CONSOLE_LOGGER.info("Calculating distances between sensors and moisture sources")
    distances = cdist(pts_xy, src_evap_clg_xy).astype(np.float16)

    # for each sensor pt and moisture source, get the vector between them
    CONSOLE_LOGGER.info("Determining vectors between sensors and moisture sources")
    vectors = (pts_xy[:, np.newaxis, :] - src_evap_clg_xy[np.newaxis, :, :]).astype(np.float32)

    # convert the vectors into unit vectors
    CONSOLE_LOGGER.info("Converting vectors to unit vectors")
    unit_vectors = (
        vectors.reshape(-1, 2) / np.linalg.norm(vectors.reshape(-1, 2), axis=0)
    ).reshape(vectors.shape)

    # convert the unit vectors to angles (in radians), then fix the angles to degrees and the right reference plane
    CONSOLE_LOGGER.info("Converting unit vectors to angles")
    angles = (
        (np.arctan2(*unit_vectors.reshape(-1, 2).T[::-1]) - np.arctan2(1, 0)).reshape(
            unit_vectors.shape[:-1]
        )
    ).astype(np.float16)
    angles_deg = np.rad2deg(angles)
    angles_deg = np.where(angles_deg < 0, -angles_deg, 360 - angles_deg).astype(np.float16)

    # for a given wind direction, determine the sensors "downwind" of that direction, within the "spread"
    # this creates an array in the shape (wind_directions, sensor_points, moisture_points)
    # note, that "downwind" uses the convention that EPW files list wind_direction as the direction the wind is COMING FROM!
    angles_deg_inverted = (angles_deg + 180) % 360
    is_downwind = []
    for wd in tqdm(WIND_DIRECTIONS, desc="Determining down-windedness for each wind direction"):
        left_edge = (wd - (SPREAD / 2)) % 360
        right_edge = (wd + (SPREAD / 2)) % 360
        downwind_mask = []
        if left_edge < right_edge:
            # doesnt cross 0
            downwind_mask.append(
                np.where(
                    np.all(
                        [left_edge < angles_deg_inverted, angles_deg_inverted < right_edge],
                        axis=0,
                    ),
                    True,
                    False,
                )
            )
        else:
            # crosses 0
            downwind_mask.append(
                np.where(
                    np.any(
                        [angles_deg_inverted > left_edge, angles_deg_inverted < right_edge],
                        axis=0,
                    ),
                    True,
                    False,
                )
            )
        is_downwind.append(downwind_mask)
    is_downwind = np.array(is_downwind).astype(bool)

    # create dictionary look up for direction masks based on the possible wind directions
    wind_direction_masks = {}
    for wd in WIND_DIRECTIONS:
        if wd in wind_direction_masks:
            continue
        wind_direction_masks[wd] = is_downwind[int((wd / 360) * len(WIND_DIRECTIONS))][
            0
        ]  # sensors "downwind" mask

    # calculate the evaporative cooling effect at each sensor point, for each hour of the year
    def calculate_the_thing(
        wind_direction: float,
        wind_speed: float,
        downwind: np.ndarray,
        WIND_DIRECTIONS: list[float],
        src_evap_clg_dispersion: np.ndarray,
        angles_deg: np.ndarray,
        SPREAD: float,
        src_evap_clg_effect: np.ndarray,
    ) -> np.ndarray | None:
        """Calculate the evaporative cooling effect at each sensor point with relation to the inputs..."""

        # get the closest wind direction in 10s (and convert 360 to 0) to allow lookup from the wind direction/mask indexing
        wind_direction = round(wind_direction / 10) * 10
        if wind_direction == 360:
            wind_direction = 0

        # convert ws and wd to the right number of decimal places for indexing later
        wind_speed = round(float(wind_speed), 1)
        wind_direction = round(float(wind_direction), 1)

        # set savepath to store result
        sp = moisture_dir / f"evap_rate_{wind_direction}_{wind_speed}.npz"

        if wind_speed == 0:
            # if wind speed is 0, then there is no evaporative cooling effect
            # TODO - make this so that proximity to evap clg sites still has an effect, but for now thats too complicated :P
            evap_rate = np.zeros(shape=distances.shape[0], dtype=np.float16)
            np.savez_compressed(sp, evap_rate)
            return evap_rate

        # get the mask for the "downwind" sensors from the moisture points
        dir_mask = downwind[int((wind_direction / 360) * len(WIND_DIRECTIONS))][
            0
        ]  # sensors "downwind" mask

        # based on evaporative clg effect being greatest near moisture sources,
        # create a matrix of evapiorative cooling effects at each sensor point, based on relationship with moisture sources
        # and the properties of that moisture csource (how effective is it, and how far might it spread)

        max_evap_dist = (
            src_evap_clg_dispersion.repeat(distances.shape[0]).reshape(distances.shape) * wind_speed
        ).astype(
            np.float16
        )  # furthest distance away that evap clg felt
        proportion_of_max_dist = (distances / max_evap_dist).astype(
            np.float16
        )  # how far each pt is from max possible distance
        proximity_val = 1 - np.where(proportion_of_max_dist > 1, 1, proportion_of_max_dist).astype(
            np.float16
        )  # scale so that 1 is near, 0 is far, capped at furthest distance

        wind_direction_difference = angles_deg.astype(np.float16) - (
            (wind_direction - 180) % 360
        )  # how "wide" the downwindedness is for each point from teh wind direction
        direction_factor = (
            1
            - np.where(
                np.all(
                    [
                        -(SPREAD / 2) < wind_direction_difference,
                        wind_direction_difference < (SPREAD / 2),
                    ],
                    axis=0,
                ),
                np.abs(wind_direction_difference) / (SPREAD / 2),
                1,
            )
        ).astype(
            np.float16
        )  # factor for downwindedness

        evap_global = (
            proximity_val
            * src_evap_clg_effect.repeat(distances.shape[0]).reshape(distances.shape)
            * direction_factor
        ).astype(
            np.float16
        )  # multiply by clg effect for each src, and direction factor
        evap_local = np.where(dir_mask, evap_global, 0).astype(
            np.float16
        )  # clip by pts within correct direction
        evap_rate = evap_local.max(axis=1).astype(np.float16)

        # save file to disk for posterity
        sp = moisture_dir / f"evap_rate_{wind_direction}_{wind_speed}.npz"
        np.savez_compressed(sp, evap_rate)

        return evap_rate

    # determine unique combinations of wind direction and speed to calculate for, and also create lists of the
    ws_epw: list[float] = []
    wd_epw: list[float] = []
    wd_ws_combinations = []
    for wd, ws in list(zip(*[epw.wind_direction, epw.wind_speed])):
        # convert wind direction to the corresponding direction in the directions array
        to_nearest = 10
        wd = round(wd / to_nearest) * to_nearest
        if wd == 360:
            wd = 0
        # convert valeus to right number of decimal places for indexing later
        ws = round(float(ws), 1)
        wd = round(float(wd), 1)
        ws_epw.append(ws)
        wd_epw.append(wd)
        wd_ws_combinations.append([wd, ws])
    wdd, wss = np.unique(wd_ws_combinations, axis=0).T
    wdd: list[float] = wdd.tolist()
    wss: list[float] = wss.tolist()

    # use ThreadPoolExecutor to run the function in parallel
    evaporative_cooling_effectiveness_file = moisture_dir / "evaporative_cooling_effectiveness.npz"
    if evaporative_cooling_effectiveness_file.exists():
        CONSOLE_LOGGER.info("Loading evaporative cooling effectiveness from file")
        with np.load(evaporative_cooling_effectiveness_file) as data:
            evaporative_cooling_effectiveness = data["arr_0"]
    else:
        with tqdm(total=len(wdd), desc="Calculating spatial evaporative cooling") as pbar:
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = []
                # calculate each unique ws and wd combo
                for wd, ws in zip(*[wdd, wss]):
                    sp = moisture_dir / f"evap_rate_{wd}_{ws}.npz"
                    if sp.exists():
                        pbar.update(1)
                        continue
                    # run the process for the timestep
                    futures.append(
                        executor.submit(
                            calculate_the_thing,
                            wind_direction=wd,
                            wind_speed=ws,
                            downwind=is_downwind,
                            WIND_DIRECTIONS=WIND_DIRECTIONS,
                            src_evap_clg_dispersion=src_evap_clg_dispersion,
                            angles_deg=angles_deg,
                            SPREAD=SPREAD,
                            src_evap_clg_effect=src_evap_clg_effect,
                        )
                    )
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

        # load the reuslts files into a dictionary to reference
        devap = {}
        for wd, ws in tqdm(
            list(zip(*[wdd, wss])), desc="Loading evaporative cooling effectiveness interim files"
        ):
            sp = moisture_dir / f"evap_rate_{wd}_{ws}.npz"
            with np.load(sp) as data:
                devap[(wd, ws)] = data["arr_0"]

        # construct the evaporative cooling effectiveness array
        evaporative_cooling_effectiveness = []
        for wd, ws in tqdm(
            list(zip(*[epw.wind_direction, epw.wind_speed])),
            desc="Constructing annual evaporative cooling effectiveness",
        ):
            # convert wind direction to the corresponding direction in the directions array
            wd = round(wd / to_nearest) * to_nearest
            if wd == 360:
                wd = 0
            wd = round(float(wd), 1)
            ws = round(float(ws), 1)
            # populate from dict
            evaporative_cooling_effectiveness.append(devap[(wd, ws)])
        # convert to numpy array
        evaporative_cooling_effectiveness = np.array(evaporative_cooling_effectiveness).astype(
            np.float16
        )

        # write evap rate to file!
        CONSOLE_LOGGER.info("Saving evaporative cooling effectiveness to file")
        np.savez_compressed(
            evaporative_cooling_effectiveness_file, evaporative_cooling_effectiveness
        )

    # calculate the DBT and RH based on the evap clg effectiveness for each sensor point
    dbt = np.array(epw.dry_bulb_temperature)
    rh = np.array(epw.relative_humidity)
    atm = np.array(epw.atmospheric_station_pressure)
    wbt = np.array(
        HourlyContinuousCollection.compute_function_aligned(
            wet_bulb_from_db_rh,
            [
                epw.dry_bulb_temperature,
                epw.relative_humidity,
                epw.atmospheric_station_pressure,
            ],
            WetBulbTemperature(),
            "C",
        )
    )

    dbt_adj_file = moisture_dir / "dbt_adj.npz"
    if dbt_adj_file.exists():
        CONSOLE_LOGGER.info("Loading adjusted dry-bulb temperature from file")
        with np.load(dbt_adj_file) as data:
            dbt_adj = data["arr_0"]
    else:
        CONSOLE_LOGGER.info("Calculating moisture adjusted dry-bulb temperature")
        dbt_adj = (dbt - ((dbt - wbt) * evaporative_cooling_effectiveness.T)).T
        np.savez_compressed(dbt_adj_file, dbt_adj)

    rh_adj_file = moisture_dir / "rh_adj.npz"
    if rh_adj_file.exists():
        CONSOLE_LOGGER.info("Loading adjusted relative humidity from file")
        with np.load(rh_adj_file) as data:
            rh_adj = data["arr_0"]
    else:
        CONSOLE_LOGGER.info("Calculating moisture adjusted relative humidity")
        rh_adj = np.clip(
            (rh * (1 - evaporative_cooling_effectiveness.T))
            + (evaporative_cooling_effectiveness.T * 100),
            a_min=0,
            a_max=100,
        ).T
        np.savez_compressed(rh_adj_file, rh_adj)

    ############################################################################
    ############################################################################

    # OPTIONAL
    idx = analysis_period_to_datetimes(AnalysisPeriod())

    # convert to parquet files
    CONSOLE_LOGGER.info("Writing relative humidity parquet file")
    rh_parquet = moisture_dir / "relative_humidity.parquet"
    pd.DataFrame(index=idx, data=rh_adj).to_parquet(rh_parquet)

    CONSOLE_LOGGER.info("Writing dry-bulb temperature parquet file")
    dbt_parquet = moisture_dir / "dry_bulb_temperature.parquet"
    pd.DataFrame(index=idx, data=dbt_adj).to_parquet(dbt_parquet)

    CONSOLE_LOGGER.info("Writing evaporative cooling effectiveness parquet file")
    evap_parquet = moisture_dir / "evaporative_cooling_effectiveness.parquet"
    pd.DataFrame(index=idx, data=evaporative_cooling_effectiveness).to_parquet(evap_parquet)
