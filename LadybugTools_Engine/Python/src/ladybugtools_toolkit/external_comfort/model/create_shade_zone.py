from honeybee.boundarycondition import boundary_conditions
from honeybee.facetype import face_types
from honeybee.model import Face, Room
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug_geometry.geometry3d import Point3D


from ladybugtools_toolkit import analytics


@analytics
def create_shade_zone(
    material: _EnergyMaterialOpaqueBase,
    width: float = 10,
    depth: float = 10,
    shade_height: float = 3,
    shade_thickness: float = 0.2,
) -> Room:
    """Create a shade zone with boundary conditions and face identifiers named per external
        comfort workflow.

    Args:
        material (_EnergyMaterialOpaqueBase):
            A surface material for the shade zones faces.
        width (float, optional):
            The width (x-dimension) of the shaded zone. Defaults to 10m.
        depth (float, optional):
            The depth (y-dimension) of the shaded zone. Defaults to 10m.
        shade_height (float, optional):
            The height of the shade. Default is 3m.
        shade_thickness (float, optional):
            The thickness of the shade. Default is 0.2m.

    Returns:
        Room:
            A zone representing a massive shade above a person.
    """

    shade_zone = Room.from_box(
        identifier="SHADE_ZONE",
        width=width,
        depth=depth,
        height=shade_thickness,
        origin=Point3D(-width / 2, -depth / 2, shade_height),
    )

    shade_construction = OpaqueConstruction(
        identifier="SHADE_CONSTRUCTION", materials=[material]
    )

    for face in shade_zone.faces:
        face: Face
        if face.normal.z == 1:
            face.identifier = "SHADE_ZONE_UP"
            face.boundary_condition = boundary_conditions.outdoors
            face.type = face_types.roof_ceiling
            face.properties.energy.construction = shade_construction
        elif face.normal.z == -1:
            face.identifier = "SHADE_ZONE_DOWN"
            face.boundary_condition = boundary_conditions.outdoors
            face.type = face_types.floor
            face.properties.energy.construction = shade_construction
        else:
            face.identifier = f"SHADE_ZONE_{face.cardinal_direction().upper()}"
            face.boundary_condition = boundary_conditions.outdoors
            face.type = face_types.wall
            face.properties.energy.construction = shade_construction

    return shade_zone
