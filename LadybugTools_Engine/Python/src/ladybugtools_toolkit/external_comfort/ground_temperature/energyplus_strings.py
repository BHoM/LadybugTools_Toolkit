import inspect

from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.ground_temperature.monthly_ground_temperature import (
    monthly_ground_temperature,
)


from ladybugtools_toolkit import analytics


@analytics
def energyplus_strings(epw: EPW) -> str:
    """Generate strings to add into EnergyPlus simulation for more accurate ground surface
        temperature results.

    Args:
        epw (EPW):
            A ladybug EPW object.

    Returns:
        List[str]:
            Strings to append to the EnergyPlus IDF simulation input file.
    """

    monthly_ground_temperatures = monthly_ground_temperature(epw, 0.5)

    ground_temperature_str = inspect.cleandoc(
        f"""
            Site:GroundTemperature:BuildingSurface,
                {monthly_ground_temperatures[0]}, !- January Surface Ground Temperature {{C}}
                {monthly_ground_temperatures[1]}, !- February Surface Ground Temperature {{C}}
                {monthly_ground_temperatures[2]}, !- March Surface Ground Temperature {{C}}
                {monthly_ground_temperatures[3]}, !- April Surface Ground Temperature {{C}}
                {monthly_ground_temperatures[4]}, !- May Surface Ground Temperature {{C}}
                {monthly_ground_temperatures[5]}, !- June Surface Ground Temperature {{C}}
                {monthly_ground_temperatures[6]}, !- July Surface Ground Temperature {{C}}
                {monthly_ground_temperatures[7]}, !- August Surface Ground Temperature {{C}}
                {monthly_ground_temperatures[8]}, !- September Surface Ground Temperature {{C}}
                {monthly_ground_temperatures[9]}, !- October Surface Ground Temperature {{C}}
                {monthly_ground_temperatures[10]}, !- November Surface Ground Temperature {{C}}
                {monthly_ground_temperatures[11]}; !- December Surface Ground Temperature {{C}}
            """
    )

    return ground_temperature_str
