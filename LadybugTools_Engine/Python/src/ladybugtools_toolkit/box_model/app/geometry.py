# Ladybug geometry imports
from honeybee.boundarycondition import Outdoors
from honeybee.face import Face
from honeybee.facetype import Wall
from honeybee.room import Room
from honeybee_energy.boundarycondition import Adiabatic
from ladybug_geometry.geometry3d.pointvector import Point3D

Face.TYPES

def create_room(room_width, room_depth, room_height):
    
    # Calculate origin at the center
    origin_x = -room_width / 2
    origin_y = -room_depth / 2
    origin = Point3D(origin_x, origin_y, 0)

    # Create room from box
    room = Room.from_box(room_width, room_depth, room_height, origin)

    # Set all faces to adiabatic by default
    for face in room.faces:
        face.boundary_condition = Adiabatic()

    # Set one wall to non-adiabatic
    room.faces[1].boundary_condition = Outdoors()

    return room

def add_glazing(room, sill_height, glazing_ratio, window_height, separation):

  # Loop through faces
  for face in room.faces:

    if face.type == Wall():
      
      # Add glazing aperture directly to face
      face.apertures_by_ratio_rectangle(
        glazing_ratio, sill_height, window_height, separation)

  return room