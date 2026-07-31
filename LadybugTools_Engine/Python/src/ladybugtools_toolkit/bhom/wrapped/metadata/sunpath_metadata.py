from ladybug.sunpath import Sunpath
from ladybugtools_toolkit.ladybug_extension.sunpath import sunrise_sunset_azimuths
from python_toolkit.bhom.bhom_object import IObject

def convert_to_bhom(d) -> IObject:
    return IObject(
        _t = "BH.oM.LadybugTools.SunData",
        sunrise_azimuth = d["sunrise"]["azimuth"],
        sunrise_time = d["sunrise"]["time"],
        noon_altitude = d["noon"]["altitude"],
        noon_time = d["noon"]["time"],
        sunset_azimuth = d["sunset"]["azimuth"],
        sunset_time = d["sunset"]["time"]
    )

def sunpath_metadata(sunpath: Sunpath) -> IObject:
    """Return a dictionary containing equinox and solstice azimuths and altitudes at sunrise, noon and sunset for the given sunpath.

    Args:
        sunpath (Sunpath):
            A Ladybug sunpath object.

    Returns:
        IObject: an IObject of type "BH.oM.LadybugTools.SunPathData", see the oM definition in LadybugTools_oM/MetaData/SunPathData.cs for the structure.
    """
    
    december_solstice = convert_to_bhom(sunrise_sunset_azimuths(sunpath, 2023, 12, 22))
    march_equinox = convert_to_bhom(sunrise_sunset_azimuths(sunpath, 2023, 3, 20))
    june_solstice = convert_to_bhom(sunrise_sunset_azimuths(sunpath, 2023, 6, 21))
    september_equinox = convert_to_bhom(sunrise_sunset_azimuths(sunpath, 2023, 9, 22))

    return IObject(
        _t = "BH.oM.LadybugTools.SunPathData",
        december_solstice = december_solstice,
        march_equinox = march_equinox,
        june_solstice = june_solstice,
        september_equinox = september_equinox
    )