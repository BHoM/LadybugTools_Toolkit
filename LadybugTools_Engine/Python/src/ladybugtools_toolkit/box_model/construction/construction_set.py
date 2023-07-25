from dataclasses import dataclass, field
from typing import Optional

from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.shade import ShadeConstruction
from honeybee_energy.construction.window import (
    EnergyWindowMaterialSimpleGlazSys, WindowConstruction)
from honeybee_energy.constructionset import ConstructionSet
from honeybee_energy.lib.constructionsets import construction_set_by_identifier
from honeybee_radiance.modifierset import ModifierSet
from ladybug.epw import EPW


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
    epw: EPW
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
        window_materials = [EnergyWindowMaterialSimpleGlazSys(identifier='BM_Simple_Window_Material',
                                                              u_factor= self.ext_win_u_factor,
                                                              shgc= self.ext_win_g_value,
                                                              vt= self.ext_win_vlt)]
        window_constr = WindowConstruction(identifier='BM_Window', materials=window_materials)
        self._constr_set.aperture_set.window_construction = window_constr
    
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
