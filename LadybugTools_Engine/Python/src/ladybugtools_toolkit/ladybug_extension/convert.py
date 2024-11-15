"""Useful methods used to convert datacollections between units."""

# pylint: disable=C0302
# pylint: disable=E0401
import numpy as np
from honeybee.room import Room
from honeybee_energy.load.equipment import ElectricEquipment
from honeybee_energy.load.lighting import Lighting
from honeybee_energy.load.people import People
from honeybee_energy.load.setpoint import Setpoint
from ladybug.datatype.area import Area
from ladybug.datatype.energy import Energy
from ladybug.datatype.energyintensity import EnergyIntensity
from ladybug.datatype.fraction import RelativeHumidity
from ladybug.datatype.generic import GenericType
from ladybug.datatype.temperature import Temperature
from ladybug.datatype.volume import Volume
from ladybug.datatype.volumeflowrate import VolumeFlowRate, VolumeFlowRateIntensity
from ladybug.epw import AnalysisPeriod, Header, HourlyContinuousCollection
from python_toolkit.bhom.analytics import bhom_analytics
from python_toolkit.bhom.logging import CONSOLE_LOGGER

from .datacollection import collection_to_series

# pylint: enable=E0401


def convert_people_collection(
    collection: HourlyContinuousCollection,
    target_unit: str,
    area: str = None,
    area_unit: str = None,
) -> HourlyContinuousCollection:
    """Convert a collection of people to a different unit.

    Args:
        collection (HourlyContinuousCollection):
            The people collection.
        target_unit (str):
            The target unit.
        area (float, optional):
            The area. Optional if conversion is between area per person and person per area.
        area_unit (str, optional):
            The unit to use for area-included conversions. Must be provided if area conversion is happening.

    Returns:
        HourlyContinuousCollection:
            The converted collection.
    """

    # TODO - add in ability to handle non-SI units here, inspired by methods below

    possible_units = ["people"]
    for au in Area.units:
        possible_units.append(f"person/{au}")
        possible_units.append(f"{au}/person")

    if collection.header.unit not in possible_units:
        raise ValueError(f"Collection unit must be one of {possible_units}.")
    if target_unit not in possible_units:
        raise ValueError(f"Target unit must be one of {possible_units}.")

    if collection.header.unit == target_unit:
        return collection

    # get input collection parameters
    _meta = collection.header.metadata
    _ap = collection.header.analysis_period

    # ensure area is provided if converting from an area normalised collection to a not normalised collection
    if (collection.header.unit == "people") or (target_unit == "people"):
        if (area is None) or (area_unit is None):
            raise ValueError(
                "'area' and 'area_unit' must be provided if converting from an area normalised collection to a not normalised collection"
            )
        if area_unit not in Area().units:
            raise ValueError(
                f"'area_unit' must be provided in one of the following units: {Area().units}"
            )

    # get the area in m2 units
    area_m2 = area * Area().to_unit(values=[1], unit="m2", from_unit=area_unit)[0]

    # convert all inputs into person/m2 for uniformity in downstream calculations
    if collection.header.unit == "people":
        area_m2 = area * Area().to_unit(values=[1], unit="m2", from_unit=area_unit)[0]
        values = [i / area_m2 for i in collection.values]
    elif "person/" in collection.header.unit:
        # convert to m2
        src_area_unit = collection.header.unit.replace("person/", "")
        area_m2 = src_area_unit * Area().to_unit(values=[1], unit="m2", from_unit=area_unit)[0]

    # handle conversion from "people" to person/m2
    if (collection.header.unit == "people") or (target_unit == "people"):
        if (area is None) or (area_unit is None):
            raise ValueError(
                "'area' and 'area_unit' must be provided if converting from an area normalised collection to a not normalised collection"
            )
        if area_unit not in Area().units:
            raise ValueError(
                f"'area_unit' must be provided in one of the following units: {Area().units}"
            )
        # convert number of people to person/m2
        area_m2 = area * Area().to_unit(values=[1], unit="m2", from_unit=area_unit)[0]
        values = [i / area_m2 for i in collection.values]

        # get the target unit area string
        target_area_unit = target_unit.split("/")[1]
        # convert area unit to SI (m2)
        area *= Area().to_unit(values=[1], unit=target_area, from_unit=area_unit)[0]
        # get collection as person/m2
        collection_normed_si = HourlyContinuousCollection(
            header=Header(
                data_type=GenericType(
                    name="PersonPerArea",
                    unit="person/m2",
                    min=0,
                    abbreviation="ppa",
                ),
                unit="person/m2",
                analysis_period=_ap,
                metadata=_meta,
            ),
            values=[i / area for i in collection.values],
        )
        # convert to target unit

    # handle conversion from "people" to other units
    if collection.header.unit == "people":
        if area is None:
            pass
    input_per_area = "person/" in collection.header.unit
    target_per_area = "person/" in target_unit

    if collection.header.unit == "people":
        if (area is None) or (area_unit is None):
            raise ValueError(
                "'area' and 'area_unit' must be provided if converting from an area normalised collection to a not normalised collection"
            )
        if area_unit not in Area().units:
            raise ValueError(
                f"'area_unit' must be provided in one of the following units: {Area().units}"
            )
        # convert area unit to SI (m2)
        area *= Area().to_unit(values=[1], unit="m2", from_unit=area_unit)[0]

    # define custom datatypes
    people_header = Header(
        data_type=GenericType(
            name="People",
            unit="people",
            min=0,
            abbreviation="ppl",
        ),
        unit="people",
        analysis_period=_ap,
        metadata=_meta,
    )
    person_per_area_header = Header(
        data_type=GenericType(
            name="PersonPerArea",
            unit="person/m2",
            min=0,
            abbreviation="ppa",
        ),
        unit="person/m2",
        analysis_period=_ap,
        metadata=_meta,
    )
    area_per_person_header = Header(
        data_type=GenericType(
            name="AreaPerPerson",
            unit="m2/person",
            min=0,
            abbreviation="app",
        ),
        unit="m2/person",
        analysis_period=_ap,
        metadata=_meta,
    )

    possible_units = [
        i.unit for i in [people_header, person_per_area_header, area_per_person_header]
    ]
    possible_dtype_names = [
        i.name
        for i in [
            people_header.data_type,
            person_per_area_header.data_type,
            area_per_person_header.data_type,
        ]
    ]

    if target_unit not in possible_units:
        raise ValueError(f"Target unit must be one of {possible_units}.")
    if collection.header.data_type.name not in possible_dtype_names:
        raise ValueError(f"Collection data type must be one of {possible_dtype_names}.")

    if collection.header.unit == target_unit:
        return collection

    # handle conversions where normalisation/aggregation not necessary
    match (collection.header.unit, target_unit):
        case ("m2/person", "person/m2"):
            vals = []
            for i in collection.values:
                if i == 0:
                    vals.append(0)
                else:
                    vals.append(1 / i)
            return HourlyContinuousCollection(
                header=person_per_area_header,
                values=vals,
            )
        case ("person/m2", "m2/person"):
            vals = []
            for i in collection.values:
                if i == 0:
                    vals.append(0)
                else:
                    vals.append(1 / i)
            return HourlyContinuousCollection(
                header=area_per_person_header,
                values=vals,
            )

    ## convert to base unit (non-normalised)
    if area is None:
        raise ValueError(
            f"Area must be provided to convert {collection.header.unit} to {target_unit}."
        )
    match collection.header.unit:
        case "m2/person":
            vals = []
            for i in collection.values:
                if i == 0:
                    vals.append(0)
                else:
                    vals.append(area / i)
            converted = HourlyContinuousCollection(
                header=people_header,
                values=vals,
            )
        case "person/m2":
            vals = []
            for i in collection.values:
                if i == 0:
                    vals.append(0)
                else:
                    vals.append(area * i)
            converted = HourlyContinuousCollection(
                header=people_header,
                values=vals,
            )
        case _:
            converted = collection.duplicate()

    ## convert to target unit
    match target_unit:
        case "people":
            return converted
        case "person/m2":
            vals = []
            for i in converted.values:
                if i == 0:
                    vals.append(0)
                else:
                    vals.append(i / area)
            return HourlyContinuousCollection(
                header=person_per_area_header,
                values=vals,
            )
        case "m2/person":
            vals = []
            for i in converted.values:
                if i == 0:
                    vals.append(0)
                else:
                    vals.append(area / i)
            return HourlyContinuousCollection(
                header=area_per_person_header,
                values=vals,
            )

    raise ValueError(f"Conversion to {target_unit} not supported. Try one of {possible_units}.")


def convert_volume_flow(
    collection: HourlyContinuousCollection,
    target_unit: str,
    area: float = None,
    volume: float = None,
    area_unit: str = None,
    volume_unit: str = None,
) -> HourlyContinuousCollection:
    """Convert a flow-rate or flow-rate-intensity to a different unit.

    Args:
        collection (HourlyContinuousCollection):
            The flow-rate collection.
        target_unit (str):
            The target unit.
        area (float, optional):
            The area to use for conversion. Optional if conversion is between non area-nor alised units, or like-units.
        volume (float, optional):
            The volume to use for conversion. Required if conversion is to/from ACH.
        area_unit (str, optional):
            The unit to use for area-included conversions. Must be provided if area conversion is happening.
        volume_unit (str, optional):
            The unit to use for volume-included conversions. Must be provided if volume conversion is included.

    Returns:
        HourlyContinuousCollection:
            The converted collection.
    """

    # handle non-conversions early-on
    if collection.header.unit == target_unit:
        return collection

    # ensure that input collection unit is one of the accepted types
    possible_units = list(VolumeFlowRate().units) + list(VolumeFlowRateIntensity().units) + ["ach"]
    if target_unit not in possible_units:
        raise NotImplementedError(
            f"Conversion between '{collection.header.unit}' and '{target_unit}' not currently supported. Conversions currently only possible between {possible_units}."
        )

    # check that conversion is to/from ach, and if so, ensure that volume is provided
    if (collection.header.unit == "ach") or (target_unit == "ach"):
        if (volume is None) or (volume_unit is None):
            raise ValueError(
                "'volume' and 'volume_unit' must be provided if converting to/from ACH"
            )
        if volume_unit not in Volume().units:
            raise ValueError(
                f"'volume_unit' must be provided in one of the following units: {Volume().units}"
            )
        # convert volume unit to SI (m3)
        volume *= Volume().to_unit(values=[1], unit="m3", from_unit=volume_unit)[0]
        # convert input ach to more typical LB datatype for downstream conversion
        if collection.header.unit == "ach":
            collection = HourlyContinuousCollection(
                header=Header(
                    data_type=VolumeFlowRate(),
                    unit="m3/s",
                    analysis_period=collection.header.analysis_period,
                    metadata=collection.header.metadata,
                ),
                values=[i * volume / 3600 for i in collection.values],
            )

    # convert input collection to SI units, to save on headaches
    collection = collection.to_si()

    input_normalised = collection.header.unit in VolumeFlowRateIntensity().units
    target_normalised = target_unit in VolumeFlowRateIntensity().units

    # convert to m3/s or m3/s-m2
    if input_normalised:
        collection = collection.to_unit(unit="m3/s-m2")
    else:
        collection = collection.to_unit(unit="m3/s")

    # if input collection and output collection are both normalised, or not normalised, then convert using standard method
    if (input_normalised and target_normalised) or (
        not input_normalised and not target_normalised and target_unit != "ach"
    ):
        return collection.to_unit(target_unit)

    # handle conversion to ACH, if input collection not normalised
    if not input_normalised and target_unit == "ach":
        return HourlyContinuousCollection(
            header=Header(
                data_type=GenericType(name="AirExchange", min=0, abbreviation="ACH", unit="ach"),
                unit="ach",
                analysis_period=collection.header.analysis_period,
                metadata=collection.header.metadata,
            ),
            values=[i * 3600 / volume for i in collection.values],
        )

    # handle input area and area_unit for area-included conversions
    if (area is None) or (area_unit is None):
        raise ValueError(
            "'area' and 'area_unit' must be provided if converting from an area normalised collection to a not normalised collection"
        )
    if area_unit not in Area().units:
        raise ValueError(
            f"'area_unit' must be provided in one of the following units: {Area().units}"
        )
    # convert area unit to SI (m2)
    area *= Area().to_unit(values=[1], unit="m2", from_unit=area_unit)[0]

    if input_normalised and (not target_normalised) and (target_unit != "ach"):
        return collection.aggregate_by_area(area=area, area_unit="m2").to_unit(unit=target_unit)

    if (not input_normalised) and target_normalised and target_unit != "ach":
        return collection.normalize_by_area(area=area, area_unit="m2").to_unit(unit=target_unit)

    # handle conversion to ach, if input collection normalised
    if input_normalised and target_unit == "ach":
        return HourlyContinuousCollection(
            header=Header(
                data_type=GenericType(name="AirExchange", min=0, abbreviation="ACH", unit="ach"),
                unit="ach",
                analysis_period=collection.header.analysis_period,
                metadata=collection.header.metadata,
            ),
            values=[i * area * 3600 / volume for i in collection.values],
        )

    raise ValueError("Conversion failed ... how did you get here?")


def convert_energy(
    collection: HourlyContinuousCollection,
    target_unit: str,
    area: float = None,
    area_unit: str = None,
) -> HourlyContinuousCollection:
    """Convert an Energy or EnergyIntensity collection to a different unit.

    Args:
        collection (HourlyContinuousCollection):
            The energy-related collection.
        target_unit (str):
            The target unit.
        area (float, optional):
            The area to use for conversion. Optional if conversion is between non area-nor alised units, or like-units.
        area_unit (str, optional):
            The unit to use for area-included conversions. Must be provided if area conversion is happening.

    Returns:
        HourlyContinuousCollection:
            The converted collection.
    """

    # handle non-conversions early-on
    if collection.header.unit == target_unit:
        return collection

    # ensure that input collection unit is one of the accepted types
    possible_units = list(Energy().units) + list(EnergyIntensity().units)
    if target_unit not in possible_units:
        raise NotImplementedError(
            f"Conversion between '{collection.header.unit}' and '{target_unit}' not currently supported. Conversions currently only possible between {possible_units}."
        )

    # convert input collection to SI units, to save on headaches
    collection = collection.to_si()

    input_normalised = collection.header.unit in EnergyIntensity().units
    target_normalised = target_unit in EnergyIntensity().units

    # convert to Wh or Wh/m2
    if input_normalised:
        collection = collection.to_unit(unit="Wh/m2")
    else:
        collection = collection.to_unit(unit="Wh")

    # if input collection and output collection are both normalised, or not normalised, then convert using standard method
    if (input_normalised and target_normalised) or (not input_normalised and not target_normalised):
        return collection.to_unit(target_unit)

    # handle input area and area_unit for area-included conversions
    if (area is None) or (area_unit is None):
        raise ValueError(
            "'area' and 'area_unit' must be provided if converting from an area normalised collection to a not normalised collection"
        )
    if area_unit not in Area().units:
        raise ValueError(
            f"'area_unit' must be provided in one of the following units: {Area().units}"
        )
    # convert area unit to SI (m2)
    area *= Area().to_unit(values=[1], unit="m2", from_unit=area_unit)[0]

    if input_normalised and (not target_normalised):
        return collection.aggregate_by_area(area=area, area_unit="m2").to_unit(unit=target_unit)

    if (not input_normalised) and target_normalised:
        return collection.normalize_by_area(area=area, area_unit="m2").to_unit(unit=target_unit)

    raise ValueError("Conversion failed ... how did you get here?")


def people_to_collection(
    people: People, target_unit: str = "m2/person", area: float = None, area_unit: str = None
) -> HourlyContinuousCollection:
    """Convert a People object into a collection representing hourly occupancy conditions.

    Args:
        people (People):
            The people object to convert.
        target_unit (str, optional):
            The target unit to convert to. Defaults to "m2/person".
        area (float, optional):
            The area to use for conversion.
        area_unit (str, optional):
            The unit to use for area-included conversions.

    Returns:
        HourlyContinuousCollection:
            A collection containing occupancy values.
    """
    if not isinstance(people, People):
        raise ValueError("Expected a People object.")

    # ensure target_unit is one of

    if (people.people_per_area is None) or (people.people_per_area == 0):
        pass


def lighting_to_collection(
    lighting: Lighting, room_or_area: Room | float, output_type: str = "total"
) -> HourlyContinuousCollection:
    """Convert a Lighting object into a collection representing hourly
    lighting gains.

    Args:
        lighting (Lighting):
            The lighting object to convert.
        room_or_area (Room | float):
            The space to be associated with the Lighting object.
            This can be a Room object or a float representing the floor area.
        output_type (str, optional):
            The type of lighting energy output to convert.
            Defaults to "total" and can be one of "total", "radiant",
            "convected", or "visible".

    Returns:
        HourlyContinuousCollection:
            A collection containing lighting values.
    """

    # validation
    if not isinstance(lighting, Lighting):
        raise ValueError("Expected a Lighting object.")

    if not isinstance(room_or_area, (Room, float)):
        raise ValueError("Expected a Room object or a float.")

    if isinstance(room_or_area, Room):
        room_or_area = room_or_area.floor_area

    if output_type not in ["total", "radiant", "convected", "visible"]:
        raise ValueError(
            "Output type must be one of 'total', 'radiant', 'convected', or 'visible'."
        )

    # get the occupancy values
    schedule = collection_to_series(lighting.schedule.data_collection)

    # TODO - handle non-hourly AnalysisPeriods here (adjust time period, Wh values, etc.)
    # TODO - add identifier to each of the colection metadatas

    # create the baseline w_m2
    w_m2 = np.array([0] * 8760).astype(float)

    # add the watts per area if available
    if lighting.watts_per_area != 0:
        match output_type:
            case "total":
                w_m2 += lighting.watts_per_area
            case "radiant":
                w_m2 += lighting.watts_per_area * lighting.radiant_fraction
            case "convected":
                w_m2 += lighting.watts_per_area * lighting.convected_fraction
            case "visible":
                w_m2 += lighting.watts_per_area * lighting.visible_fraction
            case _:
                raise ValueError("How did you get here?")

    # multiply by schedule
    w_m2 *= schedule.values

    # create the collection
    collection = HourlyContinuousCollection(
        header=Header(data_type=EnergyIntensity(), unit="Wh/m2", analysis_period=AnalysisPeriod()),
        values=w_m2.tolist(),
    )

    return collection


def electric_equipment_to_collection(
    electric_equipment: ElectricEquipment, room_or_area: Room | float, output_type: str = "total"
) -> HourlyContinuousCollection:
    """Convert an ElectricEquipment object into a collection representing
    hourly electric equipment gains.

    Args:
        electric_equipment (ElectricEquipment):
            The electric equipment object to convert.
        room_or_area (Room | float):
            The space to be associated with the ElectricEquipment object.
            This can be a Room object or a float representing the floor area.
        output_type (str, optional):
            The type of electric equipment energy output to convert.
            Defaults to "total" and can be one of "total", "radiant",
            "convected", or "latent".

    Returns:
        HourlyContinuousCollection:
            A collection containing equipment gains values.
    """

    # validation
    if not isinstance(electric_equipment, ElectricEquipment):
        raise ValueError("Expected a ElectricEquipment object.")

    if not isinstance(room_or_area, (Room, float)):
        raise ValueError("Expected a Room object or a float.")

    if isinstance(room_or_area, Room):
        room_or_area = room_or_area.floor_area

    # get the occupancy values
    schedule = collection_to_series(electric_equipment.schedule.data_collection)

    # TODO - handle non-hourly AnalysisPeriods here (adjust time period, Wh values, etc.)

    # create the baseline w_m2
    w_m2 = np.array([0] * 8760).astype(float)

    # add the watts per area if available
    if electric_equipment.watts_per_area != 0:
        match output_type:
            case "total":
                w_m2 += electric_equipment.watts_per_area
            case "radiant":
                w_m2 += electric_equipment.watts_per_area * electric_equipment.radiant_fraction
            case "convected":
                w_m2 += electric_equipment.watts_per_area * electric_equipment.convected_fraction
            case "latent":
                w_m2 += electric_equipment.watts_per_area * electric_equipment.latent
            case _:
                raise ValueError("How did you get here?")

    # multiply by schedule
    w_m2 *= schedule.values

    # create the collection
    collection = HourlyContinuousCollection(
        header=Header(data_type=EnergyIntensity(), unit="Wh/m2", analysis_period=AnalysisPeriod()),
        values=w_m2.tolist(),
    )

    return collection


def setpoint_to_collections(
    setpoint: Setpoint, output_type: str = "heating"
) -> HourlyContinuousCollection:
    """Create a collection representing the hourly setpoint conditions.

    Args:
        setpoint (Setpoint):
            The setpoint object to convert.
        output_type (str, optional):
            The type of setpoint to convert. Defaults to "heating" and can be
            one of "heating", "cooling", "humidification" or "dehumidification".

    Returns:
        HourlyContinuousCollection:
            A collection containing setpoint values.
    """

    if not isinstance(setpoint, Setpoint):
        raise ValueError("Expected a Setpoint object.")

    match output_type:
        case "heating":
            return HourlyContinuousCollection(
                values=setpoint.heating_schedule.data_collection.values,
                header=Header(data_type=Temperature(), unit="C", analysis_period=AnalysisPeriod()),
            )
        case "cooling":
            return HourlyContinuousCollection(
                values=setpoint.cooling_schedule.data_collection.values,
                header=Header(data_type=Temperature(), unit="C", analysis_period=AnalysisPeriod()),
            )
        case "humidification":
            return HourlyContinuousCollection(
                values=setpoint.humidifying_schedule.data_collection.values,
                header=Header(
                    data_type=RelativeHumidity(), unit="%", analysis_period=AnalysisPeriod()
                ),
            )
        case "dehumidification":
            return HourlyContinuousCollection(
                values=setpoint.dehumidifying_schedule.data_collection.values,
                header=Header(
                    data_type=RelativeHumidity(), unit="%", analysis_period=AnalysisPeriod()
                ),
            )
        case _:
            raise ValueError(
                "Output type must be one of 'heating', 'cooling', 'humidification' or 'dehumidification'."
            )


def collection_to_unit(
    datacollection: HourlyContinuousCollection,
    target_unit: str = None,
    **kwargs,
) -> HourlyContinuousCollection:
    """A universal converter function to help convert data collections from one unit to another, including area/volume aggregations.

    Args:
        datacollection (HourlyContinuousCollection):
            The data collection to convert.
        target_unit (str, optional):
            The target unit to convert to. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the conversion functions.
            These can include the following to normalise/aggregate collections:
            - area (in m)
            - volume (in m3)

    Returns:
        HourlyContinuousCollection: The converted data collection.
    """
    # TODO - continue here

    if not isinstance(datacollection, HourlyContinuousCollection):
        raise ValueError("Collection must be a HourlyContinuousCollection.")

    original_unit = datacollection.header.unit
    if (target_unit is None) or (target_unit == original_unit):
        return datacollection

    # handle conversions here
    # NOTE - if units are identical, this case is already handled above
    match target_unit:
        case u if target_unit in ["C", "K", "F"]:
            if not isinstance(datacollection.header.data_type, Temperature):
                raise ValueError("Conversion to temperature units requires a Temperature datatype.")
            converted = datacollection.to_unit(u)
        case u if target_unit in ["people", "m2/person", "person/m2"]:
            assert isinstance(
                datacollection.header.data_type, GenericType
            ), 'Conversions between "people" units expect a GenericType as it\'s a custom implementation.'
            if "area" not in kwargs:
                raise ValueError("Area must be provided to convert people units .")
            if u == "m2/person":
                print(u)
                converted = HourlyContinuousCollection(
                    header=Header(
                        data_type=PEOPLE_DENSITY,
                        unit=u,
                        analysis_period=datacollection.header.analysis_period,
                        metadata=datacollection.header.metadata,
                    ),
                    values=[
                        _numpeople_to_areaperperson(numpeople=i, area=kwargs["area"])
                        for i in datacollection.values
                    ],
                )
            elif u == "people":
                print(u)
                converted = HourlyContinuousCollection(
                    header=Header(
                        data_type=NUMBER_OF_PEOPLE,
                        unit=u,
                        analysis_period=datacollection.header.analysis_period,
                        metadata=datacollection.header.metadata,
                    ),
                    values=[
                        _peopleperarea_to_numpeople(peopleperarea=i, area=kwargs["area"])
                        for i in datacollection.values
                    ],
                )
            else:
                raise ValueError("Conversion to/from people units is a custom implementation.")
        case _:
            raise ValueError(
                f"Conversion between {original_unit} and {target_unit} not currently supported. Try implementing manually."
            )

    return converted

    # if isinstance(datacollection.header.data_type, Temperature):
    #     return _convert_energy_collection(datacollection, target_unit, original_unit, **kwargs)

    # # convert collection to SI units
    # datacollection = datacollection.to_si()

    # # de-normalise collection
    # input_unit_normalised = (
    #     original_unit.endswith("-m2")
    #     or original_unit.endswith("/m2")
    #     or original_unit.endswith("-ft2")
    #     or original_unit.endswith("/ft2")
    # )
    # if input_unit_normalised:
    #     if "area" not in kwargs:
    #         raise ValueError(
    #             "Cannot convert between normalised and non-normalised units without area."
    #         )
    #     datacollection = datacollection.aggregate_by_area(area=kwargs["area"], area_unit="m2")

    # # normalise collection, if requested
    # if target_unit.endswith("-m2") or target_unit.endswith("/m2"):
    #     if "area" not in kwargs:
    #         raise ValueError("Area must be provided to convert to a normalised unit.")
    #     datacollection = datacollection.normalize_by_area(area=kwargs["area"], area_unit="m2")
    # if target_unit.endswith("-ft2") or target_unit.endswith("/ft2"):
    #     if "area" not in kwargs:
    #         raise ValueError("Area must be provided to convert to a normalised unit.")
    #     datacollection = datacollection.normalize_by_area(
    #         area=Area()._m2_to_ft2(kwargs["area"]), area_unit="ft2"
    #     )

    # # attempt conversion using standard method
    # return datacollection.to_unit(target_unit)
