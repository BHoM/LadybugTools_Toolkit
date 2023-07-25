from dataclasses import dataclass, field

from honeybee.boundarycondition import boundary_conditions
from honeybee.room import Room
from honeybee.typing import clean_and_id_string
from ladybug_geometry.geometry3d import Point3D

# TODO - Add adiabatic toggle for roof / floor
# TODO - Add generation method for surrounding zones (good for IES export)

@dataclass
class BoxModelRoom:
    """Class for a typical BoxModel room with one external wall and remaining surfaces set to adiabatic"""
    name: str = field(init=True, default='BoxModel_Room')
    bay_width: float = field(init=True, default=3)
    count_bays: int = field(init = True, default=3)
    height: float = field(init=True, default=3)
    depth: float = field(init=True, default=10) 

    def __post_init__(self):
        # generate UUID
        self.identifier = clean_and_id_string(self.name)

        # Calculated variables
        # total box model width
        self.width = self.bay_width*self.count_bays
        # origin at center
        self.origin = Point3D(x=-self.width/2, y=-self.depth/2, z=0)
    
    @property
    def room(self) -> Room:
        """Returns a room from the BoxModel room geometry parameters"""
        room = Room.from_box(identifier = self.identifier,
                             width = self.width,
                             depth = self.depth,
                             height = self.height,
                             origin= self.origin)
        # set all faces to adiabatic
        for face in room.faces:
            face.boundary_condition = boundary_conditions.adiabatic
        # set north face (face 1) to outdoors, enables apertures to be added to this face 
        room.faces[1].boundary_condition = boundary_conditions.outdoors
        return room
