import inspect

from honeybee_energy.lib.scheduletypelimits import schedule_type_limit_by_identifier
from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.ground_temperature.hourly_ground_temperature import (
    hourly_ground_temperature,
)


from ladybugtools_toolkit import analytics


@analytics
def energyplus_strings_otherside_coefficient(epw: EPW) -> str:
    """Generate strings to add into EnergyPlus simulation for annual-hourly ground temperature
        values applied to sub-ground surfaces.

    Args:
        epw (EPW):
            A ladybug EPW object.

    Returns:
        List[str]:
            Strings to append to the EnergyPlus IDF simulation input file.
    """
    gnd_temp_collection = hourly_ground_temperature(epw, 0.5)

    gnd_schedule = ScheduleFixedInterval(
        "GroundTemperatureSchedule",
        gnd_temp_collection.values,
        schedule_type_limit_by_identifier("Temperature"),
    )

    ground_temperature_str = inspect.cleandoc(
        """
            SurfaceProperty:OtherSideCoefficients,
                GroundTemperature,          !- Name
                0,                          !- Combined Convective/Radiative Film Coefficient {{W/m2-K}}
                0.000000,                   !- Constant Temperature {{C}}"
                1.000000,                   !- Constant Temperature Coefficient"
                0.000000,                   !- External Dry-Bulb Temperature Coefficient"
                0.000000,                   !- Ground Temperature Coefficient"
                0.000000,                   !- Wind Speed Coefficient"
                0.000000,                   !- Zone Air Temperature Coefficient"
                GroundTemperatureSchedule;  !- Constant Temperature Schedule Name"
        """
    )

    ground_temperature_str += f"\n\n{gnd_schedule.to_idf_compact()}"

    return ground_temperature_str
