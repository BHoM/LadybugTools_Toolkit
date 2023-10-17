import json

import numpy as np
import pandas as pd

# create a test epw with fixed values to ascertain the impact of different things in the ExternalComfort modules
from ladybugtools_toolkit.ladybug_extension.epw import (
    EPW,
    Location,
    Path,
    epw_from_dataframe,
    epw_to_dataframe,
)
from ladybugtools_toolkit.new_external_comfort.material import get_material
from ladybugtools_toolkit.new_external_comfort.shelter import (
    SENSOR_LOCATION,
    Point3D,
    Shelter,
    Shelters,
    TreeSpecies,
    get_tree_shelter,
)
from ladybugtools_toolkit.new_external_comfort.simulate import (
    SimulationResult,
    collection_from_series,
    collection_to_series,
)
from ladybugtools_toolkit.new_external_comfort.typology import Typologies

print("Generating EPW file ...")
epw_file = r"C:\Users\tgerrish\BuroHappold\Sustainability and Physics - epws\ITA_SD_Cape.Bellavista.165500_TMYx.2004-2018.epw"
# epw_file = r"C:\Users\tgerrish\simulation\ARE_DU_Dubai.Intl.AP.411940_TMYx.2007-2021__Dry_Dust__Fabric\ARE_DU_Dubai.Intl.AP.411940_TMYx.2007-2021.epw"
epw = EPW(epw_file)
loc = epw.location
df = epw_to_dataframe(epw).droplevel([0, 1], axis=1)

for var in [
    "Diffuse Horizontal Illuminance (lux)",
    "Diffuse Horizontal Radiation (Wh/m2)",
    "Direct Normal Illuminance (lux)",
    "Direct Normal Radiation (Wh/m2)",
    "Extraterrestrial Direct Normal Radiation (Wh/m2)",
    "Extraterrestrial Horizontal Radiation (Wh/m2)",
    "Global Horizontal Illuminance (lux)",
    "Global Horizontal Radiation (Wh/m2)",
    "Zenith Luminance (cd/m2)",
]:
    df[var] = np.where(df[var] == 0, 0, df[var].quantile(0.8))

for var in [
    "Aerosol Optical Depth (fraction)",
    "Albedo (fraction)",
    "Atmospheric Station Pressure (Pa)",
    "Ceiling Height (m)",
    "Days Since Last Snowfall (day)",
    "Dew Point Temperature (C)",
    "Dry Bulb Temperature (C)",
    "Horizontal Infrared Radiation Intensity (W/m2)",
    "Liquid Precipitation Depth (mm)",
    "Liquid Precipitation Quantity (fraction)",
    "Opaque Sky Cover (tenths)",
    "Precipitable Water (mm)",
    "Relative Humidity (%)",
    "Snow Depth (cm)",
    "Total Sky Cover (tenths)",
    "Visibility (km)",
    "Wind Speed (m/s)",
]:
    df[var] = df[var].mean()

d = {}
for k, v in epw.monthly_ground_temperature.items():
    d[k] = v.get_aligned_collection(np.array(v.values).mean())

uniform_epw_file = Path(r"C:\Users\tgerrish\Desktop") / "uniform.epw"
_ = epw_from_dataframe(df, location=loc, monthly_ground_temperature=d).save(
    uniform_epw_file
)

print("Getting materials ...")
gnd_material = get_material("Asphalt Pavement")
shd_material = get_material("Fabric")

print("Getting simulation results ...")
res = SimulationResult(
    epw_file=uniform_epw_file,
    ground_material=gnd_material,
    shade_material=shd_material,
    identifier="pytest_EC0",
)

print("Getting shelter ...")
sh = Shelters.OVERHEAD_SMALL.value

print("Getting typology ...")
typ = Typologies.ENCLOSED.value
