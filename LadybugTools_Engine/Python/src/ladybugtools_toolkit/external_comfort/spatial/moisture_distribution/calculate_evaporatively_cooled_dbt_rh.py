from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.moisture.evaporative_cooling_effect import (
    evaporative_cooling_effect,
)
from ladybugtools_toolkit.external_comfort.spatial.moisture_distribution.load_moisture_sources import (
    load_moisture_sources,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series


def calculate_evaporatively_cooled_dbt_rh(
    simulation_directory: Union[Path, str], spatial_points: List[List[float]], epw: EPW
) -> List[pd.DataFrame]:

    dbt_matrix_path = (
        Path(simulation_directory) / "moisture" / "dry_bulb_temperature_evap.h5"
    )
    rh_matrix_path = (
        Path(simulation_directory) / "moisture" / "relative_humidity_evap.h5"
    )

    print(f"- Processing moisture data for {simulation_directory.name}")

    # load moisture sources
    moisture_sources = load_moisture_sources(simulation_directory)

    # get moisture matrices per moisture source, and resultant matrix
    moisture_matrix = np.amax(
        [
            i.spatial_moisture(
                spatial_points, epw, simulation_directory=simulation_directory
            )
            for i in moisture_sources
        ],
        axis=0,
    )

    # create matrix of dbt, rh based on moisture matrix
    dbt_matrix = []
    rh_matrix = []
    for n, row in enumerate(moisture_matrix):
        print(f"- Calculating evaporatively cooled DBT/RH [{n / 8760:0.1%}]", end="\r")
        dbt_base = epw.dry_bulb_temperature[n]
        rh_base = epw.relative_humidity[n]
        atm_base = epw.atmospheric_station_pressure[n]
        if row.sum() == 0:
            dbt_matrix.append(np.tile(dbt_base, len(row)))
            rh_matrix.append(np.tile(rh_base, len(row)))
        else:
            dbt, rh = evaporative_cooling_effect(dbt_base, rh_base, row, atm_base)
            dbt_matrix.append(dbt)
            rh_matrix.append(rh)

    idx = to_series(epw.dry_bulb_temperature).index
    print("- Saving DBT matrix")
    dbt_df = pd.DataFrame(np.array(dbt_matrix), index=idx)
    dbt_df.to_hdf(dbt_matrix_path, "df", complevel=9, complib="blosc")

    print("- Saving RH matrix")
    rh_df = pd.DataFrame(np.array(rh_matrix), index=idx)
    rh_df.to_hdf(rh_matrix_path, "df", complevel=9, complib="blosc")

    return dbt_df, rh_df
