from dataclasses import dataclass, field
from typing import List

from honeybee.model import Model
from honeybee_energy.programtype import (ElectricEquipment, Infiltration,
                                         Lighting, People, Setpoint)
from honeybee_energy.schedule.ruleset import ScheduleRuleset


def schedule_weekday_weekend(identifier: str, weekday_values: list, weekend_values: list,
                             summer_design_values: list, winter_design_values: list):
    schedule_ruleset = ScheduleRuleset.from_week_daily_values(identifier= identifier,
                                                              monday_values= weekday_values,
                                                              tuesday_values= weekday_values,
                                                              wednesday_values= weekday_values,
                                                              thursday_values= weekday_values,
                                                              friday_values= weekday_values,
                                                              saturday_values= weekend_values,
                                                              sunday_values= weekend_values,
                                                              holiday_values= weekend_values,
                                                              summer_designday_values= summer_design_values,
                                                              winter_designday_values= winter_design_values)
    return schedule_ruleset

# TODO these have a lot of repition and could be coded much more nicely, maybe with a base class?
# Each of these is currently hard coded to have 0 gain in the winter condition and weekeday load for summer condition

@dataclass
class PeopleLoad:
    m2_per_person: float = field(init=True, default= 8)
    people_schedule_week_values: List[float] = field(init=True, default_factory=lambda: [0,0,0,0,0,0,0,0,0.5,1,1,1,1,1,1,1,1,1,0.5,0,0,0,0,0])
    people_schedule_weekend_values: List[float] = field(init=True, default_factory=lambda: [0,0,0,0,0,0,0,0,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0,0,0,0,0])
    people_gain_pp: int = field(init=True, default=140) # CIBSE Guide A Table 6.3 Seated, Moderate Work Office W/person (140 man, 130 avg men, women, children)

    def __post_init__(self):
        people_schedule = schedule_weekday_weekend(identifier='BM_people',
                                                   weekday_values= self.people_schedule_week_values,
                                                   weekend_values= self.people_schedule_weekend_values,
                                                   summer_design_values= self.people_schedule_week_values,
                                                   winter_design_values= [0]*24)
        self._load = People(identifier="BM_people",
                             people_per_area= 1/self.m2_per_person,
                             occupancy_schedule= people_schedule,
                             activity_schedule= ScheduleRuleset.from_constant_value('BM_people_activity', value = self.people_gain_pp))
        
    @property
    def load(self) -> People:
        return self._load

@dataclass
class LightingLoad:
    w_per_m2: float = field(init=True, default= 5)
    lighting_schedule_week_values: List[float] = field(init=True, default_factory=lambda: [0,0,0,0,0,0,0,0,0.5,1,1,1,1,1,1,1,1,1,0.5,0,0,0,0,0])
    lighting_schedule_weekend_values: List[float] = field(init=True, default_factory=lambda: [0,0,0,0,0,0,0,0,0.5,1,1,1,1,1,1,1,1,1,0.5,0,0,0,0,0])

    def __post_init__(self):
        lighting_schedule = schedule_weekday_weekend(identifier='BM_lighting',
                                                   weekday_values= self.lighting_schedule_week_values,
                                                   weekend_values= self.lighting_schedule_weekend_values,
                                                   summer_design_values= self.lighting_schedule_week_values,
                                                   winter_design_values= [0]*24)
        self._load = Lighting(identifier="BM_lighting", watts_per_area= self.w_per_m2, schedule= lighting_schedule)
        
    @property
    def load(self) -> Lighting:
        return self._load

# TODO add diversities? would be a mutliplier for the schedules
@dataclass
class ElectricEquipmentLoad:
    w_per_m2: float = field(init=True, default= 12)
    equipment_schedule_week_values: List[float] = field(init=True, default_factory=lambda: [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.5,1,1,1,1,1,1,1,1,1,0.5,0.05,0.05,0.05,0.05,0.05])
    equipment_schedule_weekend_values: List[float] = field(init=True, default_factory=lambda: [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05])

    def __post_init__(self):
        equipment_schedule = schedule_weekday_weekend(identifier='BM_equipment',
                                                   weekday_values= self.equipment_schedule_week_values,
                                                   weekend_values= self.equipment_schedule_weekend_values,
                                                   summer_design_values= self.equipment_schedule_week_values,
                                                   winter_design_values= [0]*24)
        self._load = ElectricEquipment(identifier="BM_equipment", watts_per_area= self.w_per_m2, schedule= equipment_schedule)
        
    @property
    def load(self) -> ElectricEquipment:
        return self._load

# TODO need to work out how to handle this better, infiltration needs a model, becomes circular as this is applied to a model
@dataclass
class InfiltrationLoad:
    """If no model is passed, then a default value of infiltration will be provided of 0.0001 m3/m2 facade area per second
    """
    hb_model: Model = field(init=True, default= None)
    ach: float = field(init=True, default = 0.15) # TODO need to provide a reasonable refernce, potentially CIBSE Guide A Fig 4.15

    def __post_init__(self):
        if self.hb_model == None:
            self.flow_per_exterior_area = 0.0001
        else:
            volumes = 0 # m3 building volume
            areas = 0 # m2 facade
            for room in self.hb_model.rooms:
                volumes += room.volume
                areas += room.exterior_wall_area
            # HB expects infiltration in m3/s per m2 facade
            m3_per_hour = self.ach*volumes
            m3_per_second = m3_per_hour/(60*60)
            m3_per_m2_per_second = m3_per_second / areas
            self.flow_per_exterior_area = m3_per_m2_per_second     
        self._load = Infiltration(identifier='BM_infiltration', flow_per_exterior_area= self.flow_per_exterior_area,
                                  schedule= ScheduleRuleset.from_constant_value('BM_infiltration', value = 1))
        
    @property
    def load(self) -> Infiltration:
        return self._load

@dataclass
class SetpointProgram:
    heating_setpoint: float = field(init=True, default = 21)
    heating_setback: float = field(init=True, default = 14)
    heating_on_off: List[float] = field(init=True, default_factory=lambda: [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0])

    cooling_setpoint: float = field(init=True, default = 24)
    cooling_setback: float = field(init=True, default = 30)
    cooling_on_off: List[float] = field(init=True, default_factory=lambda: [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0])

    def __post_init__(self):
        self.heating_schedule = self.bool_to_schedule(setpoint = self.heating_setpoint, setback= self.heating_setback,
                                                      bool_schedule= self.heating_on_off, indentifier= 'BM_heating_schedule')
        self.cooling_schedule = self.bool_to_schedule(setpoint= self.cooling_setpoint, setback= self.cooling_setback,
                                                      bool_schedule=self.cooling_on_off, indentifier='BM_cooling_schedule')
        self._setpoint = Setpoint(identifier='BM_setpoints', heating_schedule= self.heating_schedule, cooling_schedule= self.cooling_schedule)
    
    @property
    def setpoint(self):
        return self._setpoint

    @staticmethod
    def bool_to_schedule(setpoint, setback, bool_schedule, indentifier):
        list_schedule = []
        for hour in bool_schedule:
            if hour == 0:
                list_schedule.append(setback)
            elif hour == 1:
                list_schedule.append(setpoint)
            else:
                raise ValueError('Must be 0 or 1 for setback or setpoint')
        
        return ScheduleRuleset.from_daily_values(identifier=indentifier, daily_values=list_schedule)
