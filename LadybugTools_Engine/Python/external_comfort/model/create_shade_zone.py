import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")


from honeybee.boundarycondition import boundary_conditions
from honeybee.facetype import face_types
from honeybee.model import Face, Room
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug_geometry.geometry3d import Point3D

from external_comfort.model import _WIDTH, _DEPTH, _SHADE_HEIGHT, _SHADE_HEIGHT_ABOVE_GROUND

def create_shade_zone(material: _EnergyMaterialOpaqueBase) -> Room:
    """Create a shade zone with boundary conditions and face identifiers named per external comfort workflow.

    Args:
        material (_EnergyMaterialOpaqueBase): A surface material for the shade zones faces.

    Returns:
        Room: A zone representing a massive shade above a person.
    """

    shade_zone = Room.from_box(
        identifier=f"SHADE_ZONE",
        width=_WIDTH,
        depth=_DEPTH,
        height=_SHADE_HEIGHT,
        origin=Point3D(-_WIDTH / 2, -_DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
    )

    shade_construction = OpaqueConstruction(
        identifier=f"SHADE_CONSTRUCTION", materials=[material]
    )

    for face in shade_zone.faces:
        face: Face
        face.boundary_condition = boundary_conditions.outdoors
        face.properties.energy.construction = shade_construction
        if face.normal.z == 1:
            face.identifier = f"SHADE_ZONE_UP"
            face.type = face_types.roof_ceiling
        elif face.normal.z == -1:
            face.identifier = f"SHADE_ZONE_DOWN"
            face.type = face_types.floor
        else:
            face.identifier = f"SHADE_ZONE_{face.cardinal_direction().upper()}"
            face.type = face_types.wall

    return shade_zone

