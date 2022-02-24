from __future__ import annotations

import sys
sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import copy
from typing import Dict

from ladybug.datacollection import AnalysisPeriod, Header, HourlyContinuousCollection
from ladybug.datatype.temperature import WetBulbTemperature
from ladybug.epw import EPW
from ladybug.psychrometrics import wet_bulb_from_db_rh


def get_evaporative_cooled_dbt_rh(
    epw: EPW, evaporative_cooling_effectiveness: float = 0.3
) -> Dict[str, HourlyContinuousCollection]:
    """Calculate the effective DBT and RH considering effects of evaporative cooling.

    Args:
        epw (EPW): A ladybug EPW object.
        evaporative_cooling_effectiveness (float, optional): The proportion of difference betwen DBT and WBT by which to adjust DBT. Defaults to 0.3 which equates to 30% effective evaporative cooling, roughly that of Misting.

    Returns:
        HourlyContinuousCollection: An adjusted dry-bulb temperature collection with evaporative cooling factored in.
    """

    dbt = copy.deepcopy(epw.dry_bulb_temperature)
    wbt = HourlyContinuousCollection(
        header=Header(
            data_type=WetBulbTemperature(), unit="C", analysis_period=AnalysisPeriod()
        ),
        values=[
            wet_bulb_from_db_rh(
                epw.dry_bulb_temperature[i],
                epw.relative_humidity[i],
                epw.atmospheric_station_pressure[i],
            )
            for i in range(8760)
        ],
    )
    dbt_adjusted = dbt - ((dbt - wbt) * evaporative_cooling_effectiveness)
    dbt_adjusted.header.metadata = {
        **dbt.header.metadata,
        **{
            "evaporative_cooling": f"{evaporative_cooling_effectiveness:0.0%}",
            "description": "Evaporatively Cooled Dry Bulb Temperature (C) - f'{evaporative_cooling_effectiveness:0.0%}' effective",
        },
    }

    rh = copy.deepcopy(epw.relative_humidity)
    rh_adjusted = (rh * (1 - evaporative_cooling_effectiveness)) + (
        evaporative_cooling_effectiveness * 100
    )
    rh_adjusted.header.metadata = {
        **rh.header.metadata,
        **{
            "evaporative_cooling": f"{evaporative_cooling_effectiveness:0.0%}",
            "description": "Evaporatively Cooled Relative Humidity (%) - f'{evaporative_cooling_effectiveness:0.0%}' effective",
        },
    }

    return {"dry_bulb_temperature": dbt_adjusted, "relative_humidity": rh_adjusted}

if __name__ == "__main__":
    pass