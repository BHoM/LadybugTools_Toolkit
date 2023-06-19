import json
import os
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import honeybee.config as hb_config
import honeybee_energy.dictutil as energy_dict_util
import matplotlib.pyplot as plt
import pandas as pd
from honeybee.boundarycondition import boundary_conditions
from honeybee.config import folders as hb_folders
from honeybee.face import Face
from honeybee.facetype import Wall
from honeybee.model import Model, Room
from honeybee_energy.config import folders as energy_folders
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.window import WindowConstruction
from honeybee_energy.constructionset import ConstructionSet, ShadeConstruction
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.lib.constructionsets import construction_set_by_identifier
from honeybee_energy.material.gas import EnergyWindowMaterialGas
from honeybee_energy.material.glazing import EnergyWindowMaterialSimpleGlazSys
from honeybee_energy.programtype import (ElectricEquipment, Infiltration,
                                         Lighting, People, ProgramType,
                                         Setpoint, Ventilation)
from honeybee_energy.result.err import Err
from honeybee_energy.result.loadbalance import LoadBalance
from honeybee_energy.result.osw import OSW
from honeybee_energy.run import (output_energyplus_files, run_idf, run_osw,
                                 to_openstudio_osw)
from honeybee_energy.schedule.day import ScheduleDay
from honeybee_energy.schedule.ruleset import (ScheduleRule, ScheduleRuleset,
                                              ScheduleTypeLimit)
from honeybee_energy.simulation.output import SimulationOutput
from honeybee_energy.simulation.parameter import SimulationParameter
from honeybee_energy.simulation.runperiod import RunPeriod
from honeybee_radiance.modifierset import ModifierSet
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_radiance_command.options.rfluxmtx import RfluxmtxOptions
from honeybee_vtk.model import Model as VTKModel
from ladybug.epw import EPW
from ladybug.futil import nukedir, preparedir
from ladybug.sql import SQLiteResult
from ladybug_geometry.geometry3d.face import Face3D
from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
from lbt_recipes.settings import RecipeSettings

"""
Within class BoxModel:
- Create the model: Box_Model
Room: Box_Room

"""

@dataclass
class BoxModel:
    epw: Union[EPW, Path, str] # post init converts Path or str to EPW obj
    name: str = field(init=True, default=None)
    construct_set: ConstructionSet = field(init=True, default=ConstructionSet(identifier='generic_constructions'))
    modifier_set: ModifierSet = field(init=True, default=ModifierSet(identifier='generic_modifiers'))
    # TODO add other system types
    ideal_air_system: IdealAirSystem = field(init=True, default = IdealAirSystem(identifier='BM_Ideal_Air',
                                                                                 economizer_type= 'NoEconomizer',
                                                                                 demand_controlled_ventilation= False))
    program_type: ProgramType = field(init=True, default = ProgramType(identifier='generic_program_type'))

    # geometry - room
    bay_width: float = field(init=True, default=3)
    count_bays: int = field(init = True, default=3)
    height: float = field(init=True, default=3)
    depth: float = field(init=True, default=10) 
    facade_azimuth_angle: float = field(init= True, default = 180)

    # geometry - glazing
    glazing_ratio: float = field(init= True, default = 0.4)
    # targets may not be achieved, LBT will overide if needed to meet glazing ratio - TODO raise warning if not met?  
    target_window_height: float = field(init=True, default=2)
    target_sill_height: float = field(init=True, default=0.8)

    # geometry - daylight
    wall_thickness: float = field(init=True, default = 0.5)
    sensor_wall_offset: float = field(init=True, default = 0.5)
    sensor_grid_size: float = field(init=True, default = 0.2)
    sensor_grid_offset: float = field(init=True, default = 0.8)
    sensor_grid_bay_count: int = field(init=True, default = 2)

    def __post_init__(self):
        # generate UUID
        self.identifier = str(uuid4())
        
        # Converts str or Path to EPW obj
        if isinstance(self.epw, (str, Path)):
            self.epw = EPW(self.epw)

        # Calculated variables
        # total box model width
        self.width = self.bay_width*self.count_bays
        # origin at center
        self.origin = Point3D(x=-self.width/2, y=-self.depth/2, z=0)

        # Error checking
        # check that height is between 1 and 10
        if not 1 <= self.height <= 10:
            raise ValueError("Height must be between 1 and 10.")
        # check glazing ratio is between 0 and 0.95
        if not self.glazing_ratio <= 0.95:
            raise ValueError("Glazing ratio must be less than 0.95")
        if self.sensor_grid_bay_count > self.count_bays:
            raise ValueError("sensor_grid_bay_count must be less than or equal to count_bays")       

        # automated naming if None given
        if self.name is None:
            self.name = f"Generic_Box_Model_W{self.width:0.0f}"

    def _create_room_geometry(self) -> Room:
        room = Room.from_box(identifier = self.name, width = self.width, depth=self.depth, height = self.height, origin=self.origin)
        # set all faces to adiabatic
        for face in room.faces:
            face.boundary_condition = boundary_conditions.adiabatic
        # set north face (face 1) to outdoors, enables apertures to be added to this face 
        room.faces[1].boundary_condition = boundary_conditions.outdoors
        room.faces[1].apertures_by_ratio_rectangle(ratio =self.glazing_ratio, aperture_height =self.target_window_height,
                                                  sill_height =self.target_sill_height, horizontal_separation= self.bay_width)
        if self.wall_thickness is not None and self.wall_thickness > 0:
            for aperture in room.faces[1].apertures:
                aperture.extruded_border(self.wall_thickness, True)
        else:
            pass 
        return room
    
    def _create_sensor_grid(self) -> SensorGrid:
        # generate vertices
        def make_square(x, y, z):
            """Returns four points that form a square with sides of length abs(x) and abs(y), 
            with a constant z coordinate."""
            return [Point3D(x, y-self.wall_thickness, z), Point3D(-x, y-self.wall_thickness, z), Point3D(-x, -y, z), Point3D(x, -y, z)]
        vertices = make_square(x = (self.bay_width/2)*self.sensor_grid_bay_count, y = (self.depth/2)-self.sensor_wall_offset, z = 0)
        face = Face3D(vertices, enforce_right_hand=False)
        mesh = face.mesh_grid(x_dim = self.sensor_grid_size, y_dim = self.sensor_grid_size, offset=self.sensor_grid_offset)
        sensor_grid = SensorGrid.from_planar_positions(identifier= (self.name + str("_sensor_grid")), positions=mesh.face_centroids,
                                                             plane_normal = (0,0,1)) 
        return sensor_grid

    def _create_hb_model(self) -> Model:
        room = self._create_room_geometry()
        room.properties.energy.construction_set = self.construct_set
        room.properties.radiance.modifier_set = self.modifier_set
        room.properties.energy.hvac = self.ideal_air_system
        room.properties.energy.program_type = self.program_type
        model = Model(identifier = self.name, rooms = [room])
        sensor_grid = self._create_sensor_grid()
        model.properties.radiance.add_sensor_grids([sensor_grid])
        model.rotate_xy(angle = 360-self.facade_azimuth_angle, origin=Point3D())
        return model
    
    @property
    def model(self) -> Model:
        return self._create_hb_model()
    
    def to_hbjson(self, name = 'BM_HBJSON', folder = None):
        model = self._create_hb_model()
        return model.to_hbjson(name = name, folder = folder)

    def to_html(self, name = None, show = False, folder = None):
        # Creates a .html file of the Box Model that can be opened in browser
        if name == None:
            name = self.name
        model = self._create_hb_model()
        if folder == None:
            vtk_model = VTKModel(model).to_html(name = name, show = show)
        else:
            vtk_model = VTKModel(model).to_html(name = name, show = show, folder = folder)
        return vtk_model
    
    # TODO add default save location
    def to_vtkjs(self):
        model = self._create_hb_model()
        return VTKModel(model).to_vtkjs(name = self.name)
    
    def to_IES_gem(self, filepath):
        # Adds surrounding zones to be made adjacent to capture the same boundary conditions as in the HB Model version
        model = self._create_hb_model()
        top_room=Room.from_box(identifier = 'top_room', width = self.width, depth=self.depth, height = self.height, origin=self.origin.move(Vector3D(x=0, y=0, z=self.height)))
        bottom_room=Room.from_box(identifier = 'bottom_room', width = self.width, depth=self.depth, height = self.height, origin=self.origin.move(Vector3D(x=0, y=0, z=-self.height)))
        back_room=Room.from_box(identifier = 'back_room', width = self.width*3, depth=self.depth, height = self.height*3, origin=self.origin.move(Vector3D(x=-self.width, y=-self.depth, z=-self.height)))
        left_room=Room.from_box(identifier = 'left_room', width = self.width, depth=self.depth, height = self.height*3, origin=self.origin.move(Vector3D(x=-self.width, y=0, z=-self.height)))
        right_room=Room.from_box(identifier = 'right_room', width = self.width, depth=self.depth, height = self.height*3, origin=self.origin.move(Vector3D(x=self.width, y=0, z=-self.height)))
        model.add_rooms([top_room,bottom_room,back_room,left_room,right_room])
        return model.to_gem(filepath, name="BM_Gem")

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]):
        raise NotImplementedError("create BoxModel from dict")
    
    @classmethod
    def from_json(cls, json_string: str):
        """Create BoxModel from a JSON string.
        
        Args:
            json_string (str):
                This is a string containing stuff for making a HB Model.

        Returns:
            BoxModel:
                Returns a BoxModel
        """
        
        dictionary = json.loads(json_string)

        return cls.from_dict(dictionary)
    
###########################
# Applying fabric properties to model

# Methods - should these be static methods?
def constr_set_from_base_climate_vintage(new_set_identifier: str, epw: EPW, vintage: str, construction_type: str):
    # Returns a contruction set based on climate (EPW), ASHRAE vintage, and ASHRAE construction type
    climate_zone = epw.ashrae_climate_zone
    climate_zone = climate_zone[0]  # strip out any qualifiers like A, C, or C
    constr_set = '{}::{}{}::{}'.format(vintage, 'ClimateZone', climate_zone, construction_type)
    base_constr_set = construction_set_by_identifier(constr_set)
    new_constr_set = base_constr_set.duplicate()
    new_constr_set.identifier = new_set_identifier
    new_constr_set.unlock()
    return new_constr_set

def adjust_vis_reflectance(construction: OpaqueConstruction, inside_reflectance: Optional[float] = None, outside_reflectance: Optional[float] = None):
    # Changes a constructions visible reflectance
    construction.unlock()
    if inside_reflectance is not None:
        construction.materials[-1].visible_reflectance = inside_reflectance
    if outside_reflectance is not None:
        construction.materials[0].visible_reflectance = outside_reflectance
    return construction

def adjust_sol_absorptance(construction: OpaqueConstruction, inside_absorptance: Optional[float] = None, outside_absorptance: Optional[float] = None):
    # Changes a constructions solar absorptance
    construction.unlock()
    if inside_absorptance is not None:
        construction.materials[-1].solar_absorptance = inside_absorptance
    if outside_absorptance is not None:
        construction.materials[0].solar_absorptance = outside_absorptance
    return construction

def adjust_thermal_absorptance(construction: OpaqueConstruction, inside_absorptance: Optional[float] = None, outside_absorptance: Optional[float] = None):
    # Changes a constructions thermal absorptance
    construction.unlock()
    if inside_absorptance is not None:
        construction.materials[-1].thermal_absorptance = inside_absorptance
    if outside_absorptance is not None:
        construction.materials[0].thermal_absorptance = outside_absorptance
    return construction

@dataclass
class BoxModelFabricProperties:
    epw: Union[EPW, Path, str]
    base_vintage: str = field(init=True, default = '2019')
    base_construction_type: str = field(init=True, default = 'SteelFramed')
    name: str = field(init=True, default = 'BM_Constr_Set')

    # TODO each of these could be a subset instead of doing it to the ConstructionSet

    # Shade properties 
    shade_sol_reflectance: float = field(init=True, default = 0.35)
    shade_vis_reflectance: float = field(init=True, default = 0.3)

    # Ext Wall properties
    ext_wall_u_factor: float = field(init=True, default = 0.18) # TODO debate whether to match this to base set (ASHRAE) or set our own default
    ext_wall_int_sol_absorptance: float = field(init=True, default = 0.55) # Matches IES default
    ext_wall_ext_sol_absorptance: float = field(init=True, default = 0.7) # Matches IES default
    ext_wall_int_therm_emissivity: float = field(init=True, default = 0.9) # Matches IES default
    ext_wall_ext_therm_emissivity: float = field(init=True, default = 0.9) # Matches IES default
    ext_wall_int_vis_reflectance: float = field(init=True, default = 0.7)
    ext_wall_ext_vis_reflectance: float = field(init=True, default = 0.3)

    # Int Wall properties
    int_wall_int_sol_absorptance: float = field(init=True, default = 0.55) # Matches IES default
    int_wall_int_therm_emissivity: float = field(init=True, default = 0.9) # Matches IES default
    int_wall_int_vis_reflectance: float = field(init=True, default = 0.7)

    # Floor properties - note that floor inside is [0] and is floor finish TODO need to watch out if this isn't adiabatic and becomes ground floor
    floor_int_sol_absorptance: float = field(init=True, default = 0.55) # Matches IES default
    floor_int_therm_emissivity: float = field(init=True, default = 0.9) # Matches IES default
    floor_int_vis_reflectance: float = field(init=True, default = 0.7)

    # Ceiling properties - note that ceiling inside is [0] and is ceiling finish.
    ceiling_int_sol_absorptance: float = field(init=True, default = 0.55) # Matches IES default
    ceiling_int_therm_emissivity: float = field(init=True, default = 0.9) # Matches IES default
    ceiling_int_vis_reflectance: float = field(init=True, default = 0.7)

    # Window TODO could add option for detailed window construction
    ext_win_u_factor: float = field(init=True, default = 1.4) # TODO debate whether to match this to base set (ASHRAE) or set our own default
    ext_win_g_value: float = field(init=True, default = 0.3) # TODO debate whether to match this to base set (ASHRAE) or set our own default
    ext_win_vlt: float = field(init=True, default = 0.6) # TODO debate how to align this with g-value to give realistic inputs
    # TODO consider adding frame

    def __post_init__(self):
        # Create unique identifier
        self.identifier = str(uuid4())

        # Pass epw input to EPW obj
        if isinstance(self.epw, (str, Path)):
            self.epw = EPW(self.epw)
        
        # Get base construction
        base_constr_set = constr_set_from_base_climate_vintage(new_set_identifier= self.name+ "_"+ self.identifier,
                                                epw = self.epw, vintage= self.base_vintage,
                                                construction_type= self.base_construction_type)
        # Duplicate base construction so original isn't edited
        new_constr_set = base_constr_set.duplicate()
        new_constr_set.unlock()
        # Assign this construction set to the obj
        self._constr_set: ConstructionSet = new_constr_set

        # Shade property assignment
        shade_constr = ShadeConstruction(identifier='BM_Shade', solar_reflectance=self.shade_sol_reflectance, visible_reflectance=self.shade_vis_reflectance)
        self._constr_set.shade_construction = shade_constr
        # Ext wall property assignment
        ext_wall = self._constr_set.wall_set.exterior_construction
        self.adjust_u_factor(construction = ext_wall, target_u_factor = self.ext_wall_u_factor)
        adjust_vis_reflectance(construction = ext_wall, inside_reflectance= self.ext_wall_int_vis_reflectance, outside_reflectance= self.ext_wall_ext_vis_reflectance)
        adjust_thermal_absorptance(construction= ext_wall, inside_absorptance= self.ext_wall_int_therm_emissivity, outside_absorptance= self.ext_wall_ext_therm_emissivity)
        adjust_sol_absorptance(construction= ext_wall, inside_absorptance= self.ext_wall_int_sol_absorptance, outside_absorptance= self.ext_wall_ext_sol_absorptance)
        # Int wall property assignment
        int_wall = self._constr_set.wall_set.interior_construction
        adjust_vis_reflectance(construction = int_wall, inside_reflectance= self.int_wall_int_vis_reflectance)
        adjust_thermal_absorptance(construction= int_wall, inside_absorptance= self.int_wall_int_therm_emissivity)
        adjust_sol_absorptance(construction= int_wall, inside_absorptance= self.int_wall_int_sol_absorptance)
        # Floor property assignment
        floor = self._constr_set.floor_set.interior_construction
        adjust_vis_reflectance(construction = floor, inside_reflectance= self.floor_int_vis_reflectance)
        adjust_thermal_absorptance(construction= floor, inside_absorptance= self.floor_int_therm_emissivity)
        adjust_sol_absorptance(construction= floor, inside_absorptance= self.floor_int_sol_absorptance)
        # Ceiling property assignment
        ceiling = self._constr_set.roof_ceiling_set.interior_construction
        adjust_vis_reflectance(construction = ceiling, inside_reflectance= self.ceiling_int_vis_reflectance)
        adjust_thermal_absorptance(construction= ceiling, inside_absorptance= self.ceiling_int_therm_emissivity)
        adjust_sol_absorptance(construction= ceiling, inside_absorptance= self.ceiling_int_sol_absorptance)
        # Window property assignment
        window_constr = WindowConstruction(identifier='BM_Window', materials=[EnergyWindowMaterialSimpleGlazSys(identifier='BM_Simple_Window_Material',
                                                                                                                     u_factor= self.ext_win_u_factor,
                                                                                                                     shgc= self.ext_win_g_value,
                                                                                                                     vt= self.ext_win_vlt)])
        self._constr_set.aperture_set.window_construction = window_constr # TODO openable windows
    
    def internal_visible_modifier_set(self) -> ModifierSet:
        return self._constr_set.to_radiance_visible_interior()
    
    @property
    def construction_set(self) -> ConstructionSet:
        return self._constr_set

    @staticmethod
    def adjust_u_factor(construction: OpaqueConstruction, target_u_factor: float):
        # TODO this is pretty bad, requires insulation to be the [2] material to be correct 
        construction.materials[2].unlock()
        r1 = construction.r_factor
        r2 = construction.materials[2].r_value
        r_remainder = r1 - r2
        new_r = 1/target_u_factor - r_remainder
        construction.materials[2].r_value = new_r
        return construction

########################

# Applying thermal template to model

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

###########################

# Energy Simulation including settings

# TODO I'd like to turn off the sizing simulation from E+, might make sims quicker?

@dataclass
class EnergySettings:
    reporting_frequency: str = field(init=True, default='Hourly')
    timestep: int = field(init=True, default=10)
    terrain_type: str = field(init=True, default = 'Suburbs') # TODO check matches IES
    load_type: str = field(init=True, default='Sensible')
    zone_energy_use: str = field(init=True, default=True) # TODO there's the potential to refine this to only report values we're interested in and reduce database size
    hvac_energy_use: bool = field(init=True, default = False) # TODO needed iff detailed HVAC systems are made accessible
    gains_and_losses: bool = field(init=True, default= True)
    comfort_metrics: bool = field(init=True, default=True)
    surface_temperature: bool = field(init=True, default=False) # TODO could add this for comfort analysis
    surface_energy_flow: bool = field(init=True, default= True)

    def __post_init__(self):
        self._simulation_output = SimulationOutput(reporting_frequency=self.reporting_frequency)
        if self.zone_energy_use:
            self._simulation_output.add_zone_energy_use(self.load_type)
        if self.hvac_energy_use:
            self._simulation_output.add_hvac_energy_use()
        if self.gains_and_losses:
            load_type = self.load_type if self.load_type != 'All' else 'Total'
            self._simulation_output.add_gains_and_losses(load_type)
        if self.comfort_metrics:
            self._simulation_output.add_comfort_metrics()
        if self.surface_temperature:
            self._simulation_output.add_surface_temperature()
        if self.surface_energy_flow:
            self._simulation_output.add_surface_energy_flow()

        self._simulation_parameters = SimulationParameter(output= self._simulation_output,
                                                          run_period= RunPeriod(),
                                                          timestep=self.timestep,
                                                          simulation_control=None,
                                                          shadow_calculation=None,
                                                          sizing_parameter=None,
                                                          terrain_type= self.terrain_type)

    def simulation_output(self):
        return self._simulation_output
    
    def simulation_parameters(self):
        return self._simulation_parameters

@dataclass
class EnergySimulation:
    model: Union[Model, BoxModel]
    epw: Union[EPW, Path, str]
    folder: str = field(init=True, default= None)
    simulation_parameters: SimulationParameter = field(init=True, default= EnergySettings().simulation_parameters())

    def __post_init__(self):
        ### Set up
        # Converts str or Path to EPW obj
        if isinstance(self.epw, (str, Path)):
            self.epw = EPW(self.epw)
                # Set unique identifier (shared with Box Model if that's given)
        
        if isinstance(self.model, BoxModel):
            self.identifier = self.model.identifier
            self.model = self.model.model # TODO this code looks ropey 
        else:
            self.identifier = str(uuid4())

        # duplicate input to avoid making changes to original model
        self.model = self.model.duplicate()
        self.simulation_parameters = self.simulation_parameters.duplicate() # ensure input is not edited

        # Set simulation folder
        if self.folder is None:
            self.folder = hb_config.folders.default_simulation_folder # TODO add the id to this? Needs to be checked

    def simulate(self):
        # TODO review this, copied some old code
        # get epw file path
        epw_file = self.epw.file_path
        hb_model = self.model
        clean_name = clean_name = re.sub(r'[^.A-Za-z0-9_-]', '_', hb_model.display_name)
        directory = os.path.join(self.folder, clean_name, 'openstudio')
        epw_obj = EPW(epw_file)
        silent = False

        # Prepare folders
        nukedir(directory, True)
        preparedir(directory)
        sch_directory = os.path.join(directory, 'schedules')
        preparedir(sch_directory)

        # Produce energy settings JSON
        # get energy settings and suplicate so original is unchanged
        energy_parameters = EnergySettings().simulation_parameters()
        energy_parameters = energy_parameters.duplicate()

        # add autocalculated design days
        des_days = [epw_obj.approximate_design_day('WinterDesignDay'),
                    epw_obj.approximate_design_day('SummerDesignDay')]
        energy_parameters.sizing_parameter.design_days = des_days

        # create json of energy parameters for iput to simulation
        energy_param_dict = energy_parameters.to_dict()
        energy_param_json = os.path.join(directory, 'simulation_parameter.json')
        with open(energy_param_json, 'w') as fp:
            json.dump(energy_param_dict, fp)

        # Produce model parameters JSON
        model_dict = hb_model.to_dict(triangulate_sub_faces=True)
        hb_model.properties.energy.add_autocal_properties_to_dict(model_dict)
        model_json = os.path.join(directory, '{}.hbjson'.format(clean_name))
        with open(model_json, 'w') as fp:
            json.dump(model_dict, fp)

        # Run simulation
        osw_path = to_openstudio_osw(model_path = model_json, osw_directory = directory, sim_par_json_path = energy_param_json)
        osm, idf = run_osw(osw_json= osw_path, silent=silent)
        sql, zsz, rdd, html, err = run_idf(idf_file_path = idf, epw_file_path = epw_file, silent=silent)

        return sql

##########
# Radiance

RFLUXMTX = {
    'ab': [3, 5, 6],
    'ad': [5000, 15000, 25000],
    'as_': [128, 2048, 4096],
    'ds': [.5, .25, .05],
    'dt': [.5, .25, .15],
    'dc': [.25, .5, .75],
    'dr': [0, 1, 3],
    'dp': [64, 256, 512],
    'st': [.85, .5, .15],
    'lr': [4, 6, 8],
    'lw': [0.000002, 6.67E-07, 4E-07],
    'ss': [0, .7, 1],
    'c': [1, 1, 1]
}

DETAIL_LEVELS = {
    '0': 0,
    'low': 0,
    '1': 1,
    'medium': 1,
    '2': 2,
    'high': 2
}

@dataclass
class AnnualRadianceSettings:
    additional_parameters: str = field(init=True, default=None)
    detail_level: Union[int, str] = field(init=True, default=1)
    option_dict = RFLUXMTX
    option_obj = RfluxmtxOptions()

    def __post_init__(self):
        self.detail_level = DETAIL_LEVELS[self.detail_level.lower()]

        for opt_name, opt_val in self.option_dict.items():
            setattr(self.option_obj, opt_name, opt_val[self.detail_level])

        if self.additional_parameters:
            self.option_obj.update_from_string(self.additional_parameters)

        self._radiance_parameters = self.option_obj.to_radiance()

    @property
    def radiance_parameters(self):
        return self._radiance_parameters

    @staticmethod
    def ambient_resolution(hb_model, aa = 0.25, detail_dimension = 0.05):
        min = hb_model.min
        max = hb_model.max
        x_dim = max.x - min.x
        y_dim = max.y - min.y
        longest_dimension = max((x_dim, y_dim))
        ambient_resolution = int((longest_dimension * aa) / detail_dimension)
        return '-ar {}'.format(ambient_resolution)



###########################

@dataclass
class Results:
    sql_file: Union[str, Path]
    hb_model: Model

    def __post_init__(self):
        # Convert BoxModel to Model
        load_balance = LoadBalance.from_sql_file(model = self.hb_model,
                                                 sql_path= self.sql_file)
        load_balance.conduction



@dataclass
class Results_Old:
    """Base class for results from a simulation run.

    Args:
        results_file: Ladybug sim folder for object simuialted
    """

    results_file: Path

@dataclass
class EnergyResults(Results):
    heating_load: pd.Series = field(init=True, repr=False, compare=False)
    cooling_load: pd.Series = field(init=True, repr=False, compare=False)
    solar_gain: pd.Series = field(init=True, repr=False, compare=False)

    def __post_init__(self):
        if not self.results_file.exists():
            raise ValueError("Results directory does not exist!")
        # e+ results exist
        if not (self.results_file / "eplusout.sql").exists():
            raise ValueError("Results directory does not contain an eplusout.sql file!")

        # load in the results - EXAMPLE!
        raise NotImplementedError("Load Energy results from an SQL file.")

    @property
    def peak_heating_load(self):
        return self.heating_load.max()

    @property
    def peak_cooling_load(self):
        return self.cooling_load.max()

    @property
    def total_load(self):
        return self.heating_load + self.cooling_load

    @property
    def peak_total_load(self):
        return self.total_load.max()

    @property
    def peak_solar_gain(self):
        return self.solar_gain.max()

    def plot_load_balance(self) -> plt.Figure:
        raise NotImplementedError()
        fig, ax = plt.subplots()
        ax.plot(self.heating_load, label="Heating")
        ax.plot(self.cooling_load, label="Cooling")
        ax.plot(self.total_load, label="Total")
        ax.legend()
        return fig


@dataclass
class DaylightResults(Results):
    illuminance: pd.DataFrame = field(init=True, repr=False, compare=False)

    def __post_init__(self):
        if not self.results_file.exists():
            raise ValueError("Results directory does not exist!")
        # annual illuminace results exist
        if not (self.results_file / "annual_illuminance.csv").exists():
            raise ValueError(
                "Results directory does not contain an annual_illuminance.csv file!"
            )

        # load in the results - EXAMPLE!
        raise NotImplementedError("Load illumiancne results from an ILL file.")

    def plot_daylight_autonomy(self) -> plt.Figure:
        raise NotImplementedError()

    def get_illuminance(self) -> pd.DataFrame:
        """Returns a time-indexed DF with point-columns ILL values."""
        raise NotImplementedError()


###########################
# SIMULATION #


class SimulationType(Enum):
    ENERGY = auto()
    DAYLIGHT = auto()
    THERMAL_COMFORT = auto()

@dataclass
class BoxModelSimulation:
    box_model: Model = field(init=True, repr=True, compare=True)
    epw_file: Path = field(init=True, repr=True, compare=True)

    simulation_options: List[SimulationType] = field(
        init=True,
        repr=True,
        compare=True,
        default_factory=lambda: [SimulationType.ENERGY, SimulationType.DAYLIGHT],
    )

    _eplus_results_file: Path = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not Path(self.epw_file).exists():
            raise ValueError("EPW file does not exist!")
        if not isinstance(self.box_model, Model):
            raise ValueError("Box model must be a honeybee model!")

        _eplus_results_file = (
            Path(hb_folders.default_simulation_folder)
            / self.box_model.identifier
            / "run"
            / "eplusout.sql"
        )

    def _already_run(self, sim_type: SimulationType) -> bool:
        # check if the simulation has already been run
        # return True or False
        raise NotImplementedError()

    def run_energy(self) -> EnergyResults:
        # Run an energy simulation using Honeybee energy

        # check if it's alresady run, and retrun existign results if True
        if self._already_run(SimulationType.ENERGY):
            return EnergyResults(results_file=self._eplus_results_file)

        # get the sql file output
        # load the sql file
        # return the results
        # return EnergyResults(results_file=sql_file)
        raise NotImplementedError()

    def run_daylight(self) -> EnergyResults:
        # Run a daylight simulation using Honeybee radiance

        raise NotImplementedError()

    def run_thermal_comfort(self) -> EnergyResults:
        # Run a daylight simulation using Honeybee radiance
        raise NotImplementedError()

    def run(self) -> Results:
        energy_results = self.run_energy()
        daylight_results = self.run_daylight()


def main():
    epw = r"C:\Program Files\IES\Shared Content\Weather\GBR_ENG_London.Wea.Ctr-St.James.Park.037700_TMYx.2004-2018.epw"
    folder = r"C:\Users\cbrooker\Desktop\Box_Model\Code_Testing"
    recipe_settings = RecipeSettings(folder = folder)
    bm_construct_set = BoxModelFabricProperties(epw = epw).construction_set
    box_model = BoxModel(epw = epw, name = "Test",
                         target_sill_height= 0, target_window_height= 2.5, bay_width= 3, count_bays= 10, depth= 10,
                         facade_azimuth_angle= 90,
                         sensor_grid_bay_count= 2,
                         construct_set= bm_construct_set,
                         modifier_set= bm_construct_set.to_radiance_visible_interior())
    bm_program_type = ProgramType(identifier='BM_program_type',
                                people = PeopleLoad().load,
                                lighting= LightingLoad().load,
                                electric_equipment= ElectricEquipmentLoad().load,
                                infiltration= InfiltrationLoad(hb_model = box_model.model, ach = 0.1).load,
                                setpoint= SetpointProgram().setpoint,
                                ventilation= None
                                )
    box_model.program_type = bm_program_type

    simulation = EnergySimulation(model = box_model, epw = box_model.epw, folder = folder,
                                  simulation_parameters= EnergySettings().simulation_parameters())
    
    results = simulation.simulate()

    sql_obj = SQLiteResult(results)
    load_balance = LoadBalance.from_sql_file(box_model.model, results)

    print(box_model.to_html(show = True))
    print("This is it's ID:", box_model.identifier)
    hb_model = box_model._create_hb_model()

    box_model.to_IES_gem(filepath=r"C:\Users\cbrooker\Desktop\IES")
    box_model.to_hbjson(folder=r"C:\Users\cbrooker\Desktop\IES")
    print('success')

if __name__ == "__main__":
    main()
