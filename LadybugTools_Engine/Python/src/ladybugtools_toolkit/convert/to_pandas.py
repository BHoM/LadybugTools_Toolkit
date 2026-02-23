"""
Methods for converting objects into Pandas objects.
"""

import datetime
from functools import singledispatch
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
from honeybee_energy.programtype import (
    ElectricEquipment,
    GasEquipment,
    Infiltration,
    Lighting,
    People,
    ProgramType,
    ServiceHotWater,
    Setpoint,
    Ventilation,
)
from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import (
    DailyCollection,
    HourlyContinuousCollection,
    HourlyDiscontinuousCollection,
    MonthlyCollection,
    MonthlyPerHourCollection,
)
from ladybug.datacollectionimmutable import (
    DailyCollectionImmutable,
    HourlyContinuousCollectionImmutable,
    HourlyDiscontinuousCollectionImmutable,
    MonthlyCollectionImmutable,
    MonthlyPerHourCollectionImmutable,
)
from ladybug.datatype.energyintensity import EnergyIntensity
from ladybug.datatype.fraction import Fraction, RelativeHumidity
from ladybug.datatype.temperature import Temperature
from ladybug.datatype.thermalcondition import ThermalCondition
from ladybug.datatype.volumeflowrate import VolumeFlowRate, VolumeFlowRateIntensity
from ladybug.epw import EPW
from ladybug.header import Header

from ..bhom.logger import CONSOLE_LOGGER

from ..ladybug_extension.calc import ( #TODO
    clearness_index,
    degree_time,
    enthalpy,
    humidity_ratio,
    wet_bulb_temperature,
)
from ..ladybug_extension.datatype import AirChangesPerHour, PeoplePerArea #TODO
from ..ladybug_extension.sunpath import (
    solar_altitude_degrees,
    solar_altitude_radians,
    solar_azimuth_degrees,
    solar_azimuth_radians,
    solar_geometry,
    solar_time_hour,
    suns_from_epw,
)
from ..ladybug_extension.util import metadata_str_to_dict, timezone_number_to_pytz #TODO


@singledispatch
def to_pandas(obj: Any) -> Any:
    """Convert a Ladybug object to a Pandas object."""
    raise NotImplementedError(f"Cannot convert {type(obj)} to a pandas object.")


@to_pandas.register(People)
def _(obj: People, **kwargs) -> pd.Series:
    """Convert a honeybee People into a data collection of Wh/m2 (load) OR occupant density."""

    ap = AnalysisPeriod()

    # get schedule as data collection
    try:
        occ_schedule: HourlyContinuousCollection = obj.occupancy_schedule.data_collection()  # type: ignore
    except TypeError:
        occ_schedule: HourlyContinuousCollection = obj.occupancy_schedule.data_collection  # type: ignore
    try:
        activity_level: HourlyContinuousCollection = obj.activity_schedule.data_collection()  # type: ignore
    except TypeError:
        activity_level: HourlyContinuousCollection = obj.activity_schedule.data_collection  # type: ignore

    # get the people density
    profile = occ_schedule * obj.people_per_area

    if kwargs.get("people_density", False):
        values = profile.values
        unit = "people/m2"
        data_type = PeoplePerArea()
        description = "People Density"
    else:
        # multiply by activity level to get gain profile
        values = (profile * activity_level).values
        unit = "Wh/m2"
        data_type = EnergyIntensity()
        description = "People Gain"
    

    # create the collection
    collection = HourlyContinuousCollection(
        header=Header(
            data_type=data_type,
            unit=unit,
            analysis_period=ap,
            metadata={"description": description, "time-zone": 0},
        ),
        values=values,
    )

    return to_pandas(collection)


@to_pandas.register(Infiltration)
def _(obj: Infiltration) -> pd.Series:
    """Convert a honeybee Infiltration load into a data collection of W/m2."""

    ap = AnalysisPeriod()

    # get schedule as data collection
    try:
        schedule = obj.schedule.data_collection()  # type: ignore
    except TypeError:
        schedule = obj.schedule.data_collection  # type: ignore

    # get the people density (person/m2) and multiply by gain per person (W/person)
    profile = schedule * obj.flow_per_exterior_area  # type: ignore

    # set the correct datatype
    collection = HourlyContinuousCollection(
        header=Header(
            data_type=VolumeFlowRateIntensity(),
            unit="m3/s-m2",
            analysis_period=ap,
            metadata={"description": "Infiltration Rate", "time-zone": 0},
        ),
        values=profile.values,
    )

    return to_pandas(collection)


@to_pandas.register(ElectricEquipment)
def _(obj: ElectricEquipment) -> pd.Series:
    """Convert a honeybee ElectricEquipment load into a data collection of W/m2."""

    ap = AnalysisPeriod()

    # get schedule as data collection
    try:
        schedule = obj.schedule.data_collection()  # type: ignore
    except TypeError:
        schedule = obj.schedule.data_collection  # type: ignore

    # get the gain density and multiply by schedule
    gain_profile = schedule * obj.watts_per_area  # type: ignore

    # set the correct datatype
    collection = HourlyContinuousCollection(
        header=Header(
            data_type=EnergyIntensity(),
            unit="Wh/m2",
            analysis_period=ap,
            metadata={"description": "Electric Equipment Gain", "time-zone": 0},
        ),
        values=gain_profile.values,
    )

    return to_pandas(collection)


@to_pandas.register(Lighting)
def _(obj: Lighting) -> pd.Series:
    """Convert a honeybee Lighting load into a data collection of W/m2."""

    ap = AnalysisPeriod()

    # get schedule as data collection
    try:
        schedule = obj.schedule.data_collection()  # type: ignore
    except TypeError:
        schedule = obj.schedule.data_collection  # type: ignore

    # get the gain density and multiply by schedule
    gain_profile = schedule * obj.watts_per_area  # type: ignore

    # set the correct datatype
    collection = HourlyContinuousCollection(
        header=Header(
            data_type=EnergyIntensity(),
            unit="Wh/m2",
            analysis_period=ap,
            metadata={"description": "Lighting Gain", "time-zone": 0},
        ),
        values=gain_profile.values,
    )

    return to_pandas(collection)


@to_pandas.register(Setpoint)
def _(obj: Setpoint) -> pd.DataFrame:
    """Convert a honeybee Setpoint object into a dataframe of heating,
    cooling, and (if available) humidification and dehumidification setpoints."""

    ap = AnalysisPeriod()

    collections = []

    # heating
    try:
        heating_schedule = obj.heating_schedule.data_collection()  # type: ignore
    except TypeError:
        heating_schedule = obj.heating_schedule.data_collection  # type: ignore
    collections.append(
        HourlyContinuousCollection(
            header=Header(
                data_type=Temperature(),
                unit="C",
                analysis_period=ap,
                metadata={"description": "Heating Setpoint", "time-zone": 0},
            ),
            values=heating_schedule.values,
        )
    )

    # cooling
    try:
        cooling_schedule = obj.cooling_schedule.data_collection()  # type: ignore
    except TypeError:
        cooling_schedule = obj.cooling_schedule.data_collection  # type: ignore
    collections.append(
        HourlyContinuousCollection(
            header=Header(
                data_type=Temperature(),
                unit="C",
                analysis_period=ap,
                metadata={"description": "Cooling Setpoint", "time-zone": 0},
            ),
            values=cooling_schedule.values,
        )
    )

    # humidification
    if obj.humidifying_schedule is not None:
        try:
            humidification_schedule = obj.humidifying_schedule.data_collection()  # type: ignore
        except TypeError:
            humidification_schedule = obj.humidifying_schedule.data_collection  # type: ignore
        collections.append(
            HourlyContinuousCollection(
                header=Header(
                    data_type=RelativeHumidity(),
                    unit="%",
                    analysis_period=ap,
                    metadata={"description": "Humidification Setpoint", "time-zone": 0},
                ),
                values=humidification_schedule.values,  # type: ignore
            )
        )

    if obj.dehumidifying_schedule is not None:
        try:
            dehumidification_schedule = obj.dehumidifying_schedule.data_collection()  # type: ignore
        except TypeError:
            dehumidification_schedule = obj.dehumidifying_schedule.data_collection  # type: ignore
        collections.append(
            HourlyContinuousCollection(
                header=Header(
                    data_type=RelativeHumidity(),
                    unit="%",
                    analysis_period=ap,
                    metadata={
                        "description": "Dehumidification Setpoint",
                        "time-zone": 0,
                    },
                ),
                values=dehumidification_schedule.values,  # type: ignore
            )
        )

    return pd.concat([to_pandas(i) for i in collections], axis=1)


@to_pandas.register(GasEquipment)
def _(obj: GasEquipment) -> pd.Series:
    """Convert a honeybee GasEquipment load into a data collection of W/m2."""

    ap = AnalysisPeriod()

    # get schedule as data collection
    try:
        schedule = obj.schedule.data_collection()  # type: ignore
    except TypeError:
        schedule = obj.schedule.data_collection  # type: ignore

    # get the gain density and multiply by schedule
    gain_profile = schedule * obj.watts_per_area  # type: ignore

    # set the correct datatype
    collection = HourlyContinuousCollection(
        header=Header(
            data_type=EnergyIntensity(),
            unit="Wh/m2",
            analysis_period=ap,
            metadata={"description": "Gas Equipment Gain", "time-zone": 0},
        ),
        values=gain_profile.values,
    )

    return to_pandas(collection)


@to_pandas.register(ServiceHotWater)
def _(obj: ServiceHotWater) -> pd.Series:
    """Convert a honeybee ServiceHotWater load into a data collection of W/m2."""

    ap = AnalysisPeriod()

    # get schedule as data collection
    try:
        schedule = obj.schedule.data_collection()  # type: ignore
    except TypeError:
        schedule = obj.schedule.data_collection  # type: ignore

    # get the gain density and multiply by schedule
    flow_profile = schedule * obj.flow_per_area  # type: ignore

    # set the correct datatype
    collection = HourlyContinuousCollection(
        header=Header(
            data_type=VolumeFlowRateIntensity(),
            unit="L/h-m2",
            analysis_period=ap,
            metadata={"description": "Hot Water", "time-zone": 0},
        ),
        values=flow_profile.values,
    )

    return to_pandas(collection)


@to_pandas.register(Ventilation)
def _(obj: Ventilation) -> pd.DataFrame:
    """Convert a honeybee Ventilation object into a dataframe of heating,
    cooling, and (if available) humidification and dehumidification Ventilation."""

    ap = AnalysisPeriod()

    try:
        schedule = obj.schedule.data_collection()  # type: ignore
    except (AttributeError, TypeError):
        try:
            schedule = obj.schedule.data_collection  # type: ignore
        except (AttributeError, TypeError):
            schedule = HourlyContinuousCollection(
                header=Header(
                    data_type=Fraction(),
                    unit="fraction",
                    analysis_period=ap,
                ),
                values=[1] * len(ap),  # type: ignore
            )

    collections = []

    collections.append(
        HourlyContinuousCollection(
            header=Header(
                data_type=AirChangesPerHour(),
                unit="ach",
                analysis_period=ap,
                metadata={"description": "Ventilation (ACH)", "time-zone": 0},
            ),
            values=schedule * obj.air_changes_per_hour,  # type: ignore
        )
    )

    collections.append(
        HourlyContinuousCollection(
            header=Header(
                data_type=VolumeFlowRateIntensity(),
                unit="m3/s-m2",
                analysis_period=ap,
                metadata={"description": "Ventilation (per-area)", "time-zone": 0},
            ),
            values=schedule * obj.flow_per_area,  # type: ignore
        )
    )

    collections.append(
        HourlyContinuousCollection(
            header=Header(
                data_type=VolumeFlowRate(),
                unit="m3/s",
                analysis_period=ap,
                metadata={"description": "Ventilation (per-person)", "time-zone": 0},
            ),
            values=schedule * obj.flow_per_person,  # type: ignore
        )
    )

    collections.append(
        HourlyContinuousCollection(
            header=Header(
                data_type=VolumeFlowRate(),
                unit="m3/s",
                analysis_period=ap,
                metadata={"description": "Ventilation (per-zone)", "time-zone": 0},
            ),
            values=schedule * obj.flow_per_zone,  # type: ignore
        )
    )

    return pd.concat([to_pandas(i) for i in collections], axis=1)

@to_pandas.register(ProgramType)
def _(obj: ProgramType) -> pd.DataFrame:
    """Convert a honeybee ProgramType into a dataframe of its loads."""
    pd_obj = []

    for attr in ["people", "lighting", "electric_equipment", "gas_equipment", "setpoint", "infiltration", "service_hot_water", "ventilation"]:
        attribute = getattr(obj, attr)
        if isinstance(attribute, People):
            pd_obj.append(to_pandas(attribute))
        elif isinstance(attribute, Lighting):
            pd_obj.append(to_pandas(attribute))
        elif isinstance(attribute, ElectricEquipment):
            pd_obj.append(to_pandas(attribute))
        elif isinstance(attribute, GasEquipment):
            pd_obj.append(to_pandas(attribute))
        elif isinstance(attribute, Setpoint):
            pd_obj.append(to_pandas(attribute))
        elif isinstance(attribute, Infiltration):
            pd_obj.append(to_pandas(attribute))
        elif isinstance(attribute, ServiceHotWater):
            pd_obj.append(to_pandas(attribute))
        elif isinstance(attribute, Ventilation):
            pd_obj.append(to_pandas(attribute))
        else:
            CONSOLE_LOGGER.error(f"Unknown attribute type \"{attr}\" in ProgramType conversion to pandas.")

    return pd.concat(pd_obj, axis=1)


@to_pandas.register(Header)
def _(obj: Header) -> tuple:
    """Convert a ladybug Header object to a tuple useful as a pandas Series name."""
    new_header = obj.duplicate()

    # add condition metadata if the header dtype is a ThermalCondition
    if isinstance(new_header.data_type, ThermalCondition):
        for k, v in new_header.data_type._unit_descr.items():
            new_header.metadata[f"condition_{k}"] = v

    # add time-zone metadata if the header has not got one
    if "time-zone" not in new_header.metadata:
        new_header.metadata["time-zone"] = 0

    # sort metadata keys for consistency
    new_header.metadata = dict(sorted(new_header.metadata.items()))

    return tuple(new_header.to_csv_strings())


@to_pandas.register(AnalysisPeriod)
def _(obj: AnalysisPeriod) -> pd.DatetimeIndex:
    """Convert a ladybug AnalysisPeriod to a pandas DatetimeIndex."""
    datetimeindex = pd.to_datetime(obj.datetimes)

    # modify datetimes where the year crosses over the year boundary
    if obj.is_reversed:
        # get the year crossing threshold datetime
        year_end_threshold = np.argmax(datetimeindex) + 1
        before_datetimes = datetimeindex[:year_end_threshold]
        after_datetimes = datetimeindex[year_end_threshold:]
        if (
            np.any((datetimeindex.month == 2) & (datetimeindex.day == 29))
            & obj.is_leap_year
        ):
            # determine whether a leap day is in either the before/after datetimes
            if np.any((before_datetimes.month == 2) & (before_datetimes.day == 29)):
                before_datetimes = [i.replace(year=2016) for i in before_datetimes]
                after_datetimes = [i.replace(year=2017) for i in after_datetimes]
            else:
                before_datetimes = [i.replace(year=2015) for i in before_datetimes]
                after_datetimes = [i.replace(year=2016) for i in after_datetimes]

        if not obj.is_leap_year:
            before_datetimes = [i.replace(year=2017) for i in before_datetimes]
            after_datetimes = [i.replace(year=2018) for i in after_datetimes]
        else:
            before_datetimes = [i.replace(year=2016) for i in before_datetimes]
            after_datetimes = [i.replace(year=2017) for i in after_datetimes]

        # combine the before and after datetimes into a single datetime array
        datetimeindex = pd.DatetimeIndex(before_datetimes + after_datetimes)

    return datetimeindex


@to_pandas.register(HourlyDiscontinuousCollection)
@to_pandas.register(HourlyDiscontinuousCollectionImmutable)
@to_pandas.register(HourlyContinuousCollection)
@to_pandas.register(HourlyContinuousCollectionImmutable)
def _(
    obj: Union[HourlyContinuousCollection, HourlyDiscontinuousCollection],
) -> pd.Series:
    """Convert a ladybug HourlyDiscontinuousCollection of HourlyContinuousCollection to a pandas Series."""
    obj_dup = obj.duplicate()

    # get the collection type and assign to metadata
    obj_dup.header.metadata["__type__"] = obj_dup.__class__.__name__

    # get the header strings to use as a name for the series
    name = to_pandas(obj_dup.header)

    # get the datetime index from the analysis period
    index = to_pandas(obj_dup.header.analysis_period)

    # assign time-zone (if possible)
    if "time-zone" in obj_dup.header.metadata:
        index = index.tz_localize(
            timezone_number_to_pytz(obj_dup.header.metadata["time-zone"]),
        )

    return pd.Series(
        data=obj_dup.values,
        index=index,
        name=name,
    )


@to_pandas.register(MonthlyCollection)
@to_pandas.register(MonthlyCollectionImmutable)
def _(obj: MonthlyCollection) -> pd.Series:
    """Convert a ladybug MonthlyCollection to a pandas Series."""
    obj_dup = obj.duplicate()

    # get the collection type and assign to metadata
    obj_dup.header.metadata["__type__"] = obj_dup.__class__.__name__

    # get the header strings to use as a name for the series
    name = to_pandas(obj_dup.header)

    # get the datetime index from the analysis period
    index = to_pandas(obj_dup.header.analysis_period)

    # filter the index using the months from the object datetimes
    _new_index: list[datetime.datetime] = []
    for month in obj_dup.datetimes:
        dt = (index[index.month == month + 1][0]).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        _new_index.append(dt)
    new_index: pd.DatetimeIndex = pd.DatetimeIndex(_new_index)

    # assign time-zone (if possible)
    if "time-zone" in obj_dup.header.metadata:
        new_index = new_index.tz_localize(
            timezone_number_to_pytz(obj_dup.header.metadata["time-zone"]),
        )

    return pd.Series(
        data=obj_dup.values,
        index=new_index,
        name=name,
    )


@to_pandas.register(MonthlyPerHourCollection)
@to_pandas.register(MonthlyPerHourCollectionImmutable)
def _(obj: MonthlyPerHourCollection) -> pd.Series:
    """Convert a ladybug MonthlyPerHourCollection to a pandas Series."""
    obj_dup = obj.duplicate()

    # get the collection type and assign to metadata
    obj_dup.header.metadata["__type__"] = obj_dup.__class__.__name__

    # get the header strings to use as a name for the series
    name = to_pandas(obj_dup.header)

    # get the datetime index from the analysis period
    index = to_pandas(obj_dup.header.analysis_period)

    # get the months and hours from the object datetimes
    new_idx = []
    for month, hour, minute in obj_dup.datetimes:
        dt = index[
            np.all(
                [index.month == month, index.hour == hour, index.minute == minute],
                axis=0,
            )
        ][0]
        new_idx.append(dt.replace(day=1, second=0, microsecond=0))
    new_index: pd.DatetimeIndex = pd.DatetimeIndex(new_idx).sort_values()

    # assign time-zone (if possible)
    if "time-zone" in obj_dup.header.metadata:
        new_index = new_index.tz_localize(
            timezone_number_to_pytz(obj_dup.header.metadata["time-zone"]),
        )

    return pd.Series(
        data=obj_dup.values,
        index=new_index,
        name=name,
    )


@to_pandas.register(DailyCollection)
@to_pandas.register(DailyCollectionImmutable)
def _(obj: DailyCollection) -> pd.Series:
    """Convert a ladybug DailyCollection to a pandas Series."""
    obj_dup = obj.duplicate()

    # get the collection type and assign to metadata
    obj_dup.header.metadata["__type__"] = obj_dup.__class__.__name__

    # get the header strings to use as a name for the series
    name = to_pandas(obj_dup.header)

    # get the datetime index from the analysis period
    index = to_pandas(obj_dup.header.analysis_period)

    # filter the index using the days from the object datetimes
    new_index = []
    for day in obj_dup.datetimes:
        dt = index[index.day_of_year == day][0]
        new_index.append(dt.replace(hour=0, minute=0, second=0, microsecond=0))
    new_index = pd.DatetimeIndex(new_index).sort_values()

    # assign time-zone (if possible)
    if "time-zone" in obj_dup.header.metadata:
        new_index = new_index.tz_localize(
            timezone_number_to_pytz(obj_dup.header.metadata["time-zone"]),
        )

    return pd.Series(
        data=obj_dup.values,
        index=new_index,
        name=name,
    )


@to_pandas.register(ScheduleFixedInterval)
def _(obj: ScheduleFixedInterval) -> pd.Series:
    """Convert a Honeybee Energy ScheduleFixedInterval to a pandas Series."""
    try:
        return to_pandas(obj.data_collection())  # type: ignore
    except TypeError:
        return to_pandas(obj.data_collection)  # type: ignore


@to_pandas.register(ScheduleRuleset)
def _(obj: ScheduleRuleset) -> pd.Series:
    """Convert a Honeybee Energy ScheduleRuleset to a pandas Series."""
    return to_pandas(obj.data_collection())  # type: ignore


@to_pandas.register(EPW)
def _(obj: EPW) -> pd.DataFrame:
    """Convert a ladybug EPW object to a pandas DataFrame."""
    # this part is required to register the data collections with the EPW object
    obj.dry_bulb_temperature

    collections_as_series = []
    for i in obj._data:
        if isinstance(i, (HourlyContinuousCollection, HourlyDiscontinuousCollection)):
            collections_as_series.append(to_pandas(i))
    df = pd.concat(collections_as_series, axis=1)

    # create additional columns for useful data
    additional_series = []

    # obtain ground temperature data
    for _, ground_temp in obj.monthly_ground_temperature.items():
        temp = to_pandas(ground_temp)
        temp = temp.reindex(df.index)
        temp.iloc[-1] = temp.iloc[0]
        temp = temp.interpolate(method="polynomial", order=2)
        temp.name = (
            temp.name[0],
            temp.name[1],
            temp.name[2].replace("MonthlyCollection", "HourlyContinuousCollection"),
        )
        additional_series.append(temp)

    # sun stuff
    sun_objects = suns_from_epw(epw=obj).tolist()
    additional_series.append(
        to_pandas(solar_altitude_degrees(epw=obj, sun_objects=sun_objects))
    )
    additional_series.append(
        to_pandas(solar_altitude_radians(epw=obj, sun_objects=sun_objects))
    )
    additional_series.append(
        to_pandas(solar_azimuth_degrees(epw=obj, sun_objects=sun_objects))
    )
    additional_series.append(
        to_pandas(solar_azimuth_radians(epw=obj, sun_objects=sun_objects))
    )

    solar_declination, eq_of_time = solar_geometry(epw=obj)
    additional_series.append(to_pandas(solar_declination))
    additional_series.append(to_pandas(eq_of_time))

    sol_time = solar_time_hour(epw=obj, eot=eq_of_time)
    additional_series.append(to_pandas(sol_time))

    # clearness index
    ci = clearness_index(epw=obj, sun_objects=sun_objects)
    additional_series.append(to_pandas(ci))

    # wet bulb temperature
    wbt = wet_bulb_temperature(epw=obj)
    additional_series.append(to_pandas(wbt))

    # humidity ratio
    hr = humidity_ratio(epw=obj)
    additional_series.append(to_pandas(hr))

    # enthalpy
    ent = enthalpy(epw=obj, hum_ratio=hr)
    additional_series.append(to_pandas(ent))

    # sky temperature
    additional_series.append(to_pandas(obj.sky_temperature))

    # degree time
    degree_days = degree_time(epw=obj, return_type="days")
    for dd in degree_days:
        additional_series.append(to_pandas(dd))
    degree_hours = degree_time(epw=obj, return_type="hours")
    for dh in degree_hours:
        additional_series.append(to_pandas(dh))

    return pd.concat([df] + additional_series, axis=1)


def get_pandas_metadata_attr(
    pd_obj: Union[pd.Series, pd.DataFrame], key: str
) -> list[Any]:
    """Get a metadata attribute from a pandas Series or DataFrame, assuming it uses the convention for X, Y, METADATA_AS_STR.

    Args:
        pd_obj: A pandas Series or DataFrame.
        key: The metadata key to get.
    Returns:
        The metadata value for the given key.
    """
    if isinstance(pd_obj, pd.Series):
        dtype_strings = pd_obj.name[0]  # type: ignore
        unit_strings = pd_obj.name[1]  # type: ignore
        metadata_strings = [pd_obj.name[2]]  # type: ignore
    elif isinstance(pd_obj, pd.DataFrame):
        dtype_strings = [i for i in pd_obj.columns.get_level_values(0)]
        unit_strings = [i for i in pd_obj.columns.get_level_values(1)]
        metadata_strings = [i for i in pd_obj.columns.get_level_values(2)]
    else:
        raise TypeError("pd_obj must be a pandas Series or DataFrame.")
    
    allowable_keys = {"unit", "data_type"}
    attribute_strings = []
    if key == "unit":
        attribute_strings = unit_strings
    elif key in ("data_type",):
        attribute_strings = dtype_strings
    else:
        metadata_dicts = [
            metadata_str_to_dict(metadata_str) for metadata_str in metadata_strings
        ]
        # check that all metadata contain the key
        [[allowable_keys.add(i) for i in j.keys()] for j in metadata_dicts]
        allowable_keys.discard("__type__")
        for metadata_dict in metadata_dicts:
            if key not in metadata_dict:
                raise KeyError(f"Metadata key '{key}' not found in pandas object. It must be one of: {allowable_keys}.")
            attribute_strings.append(metadata_dict.get(key))
    
    if len(attribute_strings) == 1 and isinstance(pd_obj, pd.Series):
        return attribute_strings[0]
    
    return attribute_strings


def reindex_by_metadata_attr(pd_obj: Union[pd.Series, pd.DataFrame], keys: Sequence[str]) -> Union[pd.Series, pd.DataFrame]:
    """Re-index a pandas Series or DataFrame using metadata attributes.

    Args:
        pd_obj: A pandas Series or DataFrame.
        keys: The metadata keys to use for renaming.
    Returns:
        The renamed pandas Series or DataFrame.
    """

    if not isinstance(keys, (list, tuple)):
        raise TypeError("keys must be a list or tuple.")
    
    # use the get_pandas_metadata_attr for validation
    new_arrays = []
    for key in keys:
        new_arrays.append(get_pandas_metadata_attr(pd_obj, key))
    
    if isinstance(pd_obj, pd.Series):
        if len(keys) == 1:
            # single level index
            pd_obj.index = pd.Index(data=new_arrays[0])
        else:
            # multi-level index
            pd_obj.index = pd.MultiIndex.from_arrays(arrays=new_arrays)
    elif isinstance(pd_obj, pd.DataFrame):
        if len(keys) == 1:
            # single level columns
            pd_obj.columns = pd.Index(data=new_arrays[0])
        else:
            # multi-level columns
            pd_obj.columns = pd.MultiIndex.from_arrays(arrays=new_arrays)
    return pd_obj
