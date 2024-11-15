# from honeybee_energy.load.equipment import ElectricEquipment

# from ...ladybug_extension.datacollection import (
#     _areaperperson_to_peopleperarea,
#     _numpeople_to_peopleperarea,
#     _peopleperarea_to_areaperperson,
#     _peopleperarea_to_numpeople,
# )


# def equipment_to_collection(
#     equipment: ElectricEquipment, room_area: Room | float, unit: str = "kWh"
# ) -> HourlyContinuousCollection:
#     """Convert an ElectricEquipment object into a collection representing the hourly equipment conditions."""

#     if isinstance(room_area, Room):
#         room_area = room_area.floor_area

#     # get the occupancy values
#     schedule = collection_to_series(equipment.schedule.data_collection)

#     # create the baseline w_m2
#     w_m2 = np.array([0] * 8760).astype(float)

#     # add the watts per area if available
#     if equipment.watts_per_area != 0:
#         w_m2 += equipment.watts_per_area

#     # multiply by schedule
#     w_m2 *= schedule.values

#     # create the collection
#     collection = HourlyContinuousCollection(
#         header=Header(data_type=EnergyIntensity(), unit="Wh/m2", analysis_period=AnalysisPeriod()),
#         values=w_m2.tolist(),
#     )

#     # convert to unit
#     collection = collection_to_unit(collection, unit, room_area)

#     return collection


# def people_to_collection(
#     people: People, unit: str = "people", area: float = None
# ) -> HourlyContinuousCollection:
#     """Convert a People object into a collection representing the hourly peoiple conditions."""

#     if not isinstance(people, People):
#         raise ValueError("Expected a People object.")

#     match unit:
#         case "people":
#             if area is None:
#                 raise ValueError("Area must be provided when target unit is 'people'.")
#             collection = _people_obj_to_people(people, area)
#         case "m2/person":
#             collection = _people_obj_to_people_density(people)
#         case "Wh" | "kWh" | "Wh/m2" | "kWh/m2":
#             if area is None:
#                 raise ValueError("Area must be provided when target unit is 'an energy type'.")
#             collection = collection_to_unit(
#                 _people_obj_to_wh(people, area), target_unit=unit, area=area
#             )
#         case _:
#             raise ValueError(
#                 f"Invalid unit. Should be one of [people, m2/person, Wh, kWh]. Got {unit}."
#             )

#     return collection


# def lighting_to_collection(
#     lighting: Lighting, room_or_area: Room | float, unit: str = "kWh"
# ) -> HourlyContinuousCollection:
#     """Convert an ElectricEquipment object into a collection representing the hourly lighting conditions."""

#     if not isinstance(lighting, Lighting):
#         raise ValueError("Expected a Lighting object.")

#     if isinstance(room_or_area, Room):
#         room_or_area = room_or_area.floor_area

#     # get the occupancy values
#     schedule = collection_to_series(lighting.schedule.data_collection)

#     # create the baseline w_m2
#     w_m2 = np.array([0] * 8760).astype(float)

#     # add the watts per area if available
#     if lighting.watts_per_area != 0:
#         w_m2 += lighting.watts_per_area

#     # multiply by schedule
#     w_m2 *= schedule.values

#     # create the collection
#     collection = HourlyContinuousCollection(
#         header=Header(data_type=EnergyIntensity(), unit="Wh/m2", analysis_period=AnalysisPeriod()),
#         values=w_m2.tolist(),
#     )

#     # convert to unit
#     collection = collection_to_unit(collection, unit, room_or_area)

#     return collection


# def setpoint_to_collections(
#     setpoint: Setpoint,
# ) -> dict[str, HourlyContinuousCollection]:
#     """Create collections representing the hourly heating and cooling setpoint conditions."""

#     if not isinstance(setpoint, Setpoint):
#         raise ValueError("Expected a Setpoint object.")

#     htg = HourlyContinuousCollection(
#         values=setpoint.heating_schedule.data_collection.values,
#         header=Header(data_type=Temperature(), unit="C", analysis_period=AnalysisPeriod()),
#     )
#     clg = HourlyContinuousCollection(
#         values=setpoint.cooling_schedule.data_collection.values,
#         header=Header(data_type=Temperature(), unit="C", analysis_period=AnalysisPeriod()),
#     )

#     return {
#         "heating": htg,
#         "cooling": clg,
#     }


# def ventilation_to_collection(
#     ventilation: Ventilation, room: Room, unit: str = "m3/s"
# ) -> HourlyContinuousCollection:
#     """Convert a Ventilation object into a collection representing the hourly ventilation conditions."""

#     if not isinstance(ventilation, Ventilation):
#         raise ValueError("Expected a Ventilation object.")

#     # get the schedule values
#     try:
#         schedule = collection_to_series(ventilation.schedule.data_collection)
#     except AttributeError:
#         schedule = collection_to_series(ventilation.schedule.data_collection())

#     # get space metrics
#     number_of_people = collection_to_series(
#         people_to_collection(
#             people=room.properties.energy.program_type.people, area=room.floor_area, unit="people"
#         )
#     ).values

#     # create the baseline m3/s flow rate for the zone
#     m3_s = np.array([0] * 8760).astype(float)

#     # add flow per person if available
#     if ventilation.flow_per_person != 0:
#         m3_s += number_of_people * ventilation.flow_per_person

#     # add flow per area if available
#     if ventilation.flow_per_area != 0:
#         m3_s += area * ventilation.flow_per_area

#     # add flow per zone if available
#     if ventilation.flow_per_zone != 0:
#         m3_s += ventilation.flow_per_zone

#     # add air changes per hour if available
#     if ventilation.air_changes_per_hour != 0:
#         m3_s += ventilation.air_changes_per_hour * room.volume / 3600

#     # multiply by schedule
#     m3_s *= schedule.values

#     # create collection from values
#     collection = HourlyContinuousCollection(
#         header=Header(data_type=VolumeFlowRate(), unit="m3/s", analysis_period=AnalysisPeriod()),
#         values=m3_s.tolist(),
#     )

#     # convert to target unit
#     if unit != "ach":
#         collection = collection_to_unit(collection, unit, room.floor_area)
#     else:
#         collection = HourlyContinuousCollection(
#             values=collection * 3600 / room.volume,
#             header=Header(
#                 data_type=AIR_EXCHANGE,
#                 unit="ach",
#                 analysis_period=AnalysisPeriod(),
#                 metadata=collection.header.metadata,
#             ),
#         )

#     return collection


# def infiltration_to_collection(
#     infiltration: Infiltration, room: Room, unit: str = "m3/s"
# ) -> HourlyContinuousCollection:
#     """Convert an Infiltration object into a collection representing the hourly infiltration conditions."""

#     if not isinstance(infiltration, Infiltration):
#         raise ValueError("Expected an Infiltration object.")

#     # get the schedule values
#     schedule = collection_to_series(infiltration.schedule.data_collection())

#     # create the baseline m3/s flow rate for the zone
#     m3_s = np.array([0] * 8760).astype(float)

#     # add flow per area if available
#     if infiltration.flow_per_exterior_area != 0:
#         m3_s += room.exposed_area * infiltration.flow_per_exterior_area

#     # multiply by schedule
#     m3_s *= schedule.values

#     # create collection from values
#     collection = HourlyContinuousCollection(
#         header=Header(data_type=VolumeFlowRate(), unit="m3/s", analysis_period=AnalysisPeriod()),
#         values=m3_s.tolist(),
#     )

#     # convert to target unit
#     if unit != "ach":
#         collection = collection_to_unit(collection, unit, room.floor_area)
#     else:
#         collection = HourlyContinuousCollection(
#             values=collection * 3600 / room.volume,
#             header=Header(
#                 data_type=AIR_EXCHANGE,
#                 unit="ach",
#                 analysis_period=AnalysisPeriod(),
#                 metadata=collection.header.metadata,
#             ),
#         )

#     return collection
