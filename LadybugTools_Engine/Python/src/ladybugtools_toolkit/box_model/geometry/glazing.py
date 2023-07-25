from dataclasses import dataclass, field

from honeybee.boundarycondition import Outdoors
from honeybee.facetype import Wall
from honeybee.room import Room
from honeybee.shade import Shade
from honeybee.typing import clean_and_id_string


@dataclass
class BoxModelGlazing:
    """Class containing all the data required for BoxModel workflow"""
    glazing_ratio: float = field(init= True, default = 0.4)
    # targets may not be achieved, LBT will overide if needed to meet glazing ratio - TODO raise warning if not met?  
    target_window_height: float = field(init=True, default=2)
    target_sill_height: float = field(init=True, default=0.8)
    wall_thickness: float = field(init=True, default = 0.5)
    bay_width: float = field(init=True, default=3)

def assign_glazing_parameters(glazing_parameters: BoxModelGlazing, room: Room):
    """Returns a room with BoxModelGlazing parameters assigned, including adding wall thickness (reveal and internal wall finish)"""
    room = room.duplicate()
    for face in room.faces:
        if can_host_aperture(face):
            face.apertures_by_ratio_rectangle(ratio = glazing_parameters.glazing_ratio,
                                              aperture_height = glazing_parameters.target_window_height,
                                              sill_height = glazing_parameters.target_sill_height,
                                              horizontal_separation = glazing_parameters.bay_width,
                                              vertical_separation = 0)            

            if glazing_parameters.wall_thickness:
                for aperture in face.apertures:
                    assign_border_shades(aperture,
                                        depth = glazing_parameters.wall_thickness,
                                        indoor = True)
                    
                shade_identifier = clean_and_id_string('Internal_Face')
                internal_wall = Shade(identifier = shade_identifier,
                                    geometry= face.punched_geometry,
                                    is_detached = False)
                movement_vector = -face.normal*glazing_parameters.wall_thickness
                internal_wall.move(movement_vector)
                room.add_indoor_shade(internal_wall)
    return room

def can_host_aperture(face):
    """Test if a face is intended to host apertures (type:Wall & bc:Outdoors)"""
    return isinstance(face.boundary_condition, Outdoors) and \
        isinstance(face.type, Wall)

def assign_border_shades(aperture, depth, indoor):
    """Assign window border shades to an Aperture based on a set of inputs."""
    if isinstance(aperture.boundary_condition, Outdoors):
        aperture.extruded_border(depth, indoor)