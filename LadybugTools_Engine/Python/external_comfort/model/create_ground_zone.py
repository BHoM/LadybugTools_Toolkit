import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from external_comfort.model import _DEPTH, _GROUND_HEIGHT, _WIDTH
from honeybee.boundarycondition import boundary_conditions
from honeybee.facetype import face_types
from honeybee.model import Face, Room
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.material.opaque import EnergyMaterial, _EnergyMaterialOpaqueBase
from ladybug_geometry.geometry3d import Point3D


def create_ground_zone(
    material: _EnergyMaterialOpaqueBase, shaded: bool = False
) -> Room:
    """Create a ground zone with boudnary conditions and face identifiers named per external comfort workflow.

    Args:
        material (_EnergyMaterialOpaqueBase): A surface material for the ground zones topmost face.
        shaded (bool, optional): A flag to describe whether this zone is shaded. Defaults to False.

    Returns:
        Room: A zone representing the ground beneath a person.
    """

    shade_id = "SHADED" if shaded else "UNSHADED"

    ground_zone = Room.from_box(
        identifier=f"GROUND_ZONE_{shade_id}",
        width=_WIDTH,
        depth=_DEPTH,
        height=_GROUND_HEIGHT,
        origin=Point3D(-_WIDTH / 2, -_DEPTH / 2, -_GROUND_HEIGHT),
    )

    ground_top_construction = OpaqueConstruction(
        identifier=f"GROUND_CONSTRUCTION_TOP", materials=[material]
    )
    ground_interface_construction = OpaqueConstruction(
        identifier=f"GROUND_CONSTRUCTION_INTERFACE",
        materials=[
            EnergyMaterial(
                identifier="GROUND_MATERIAL_INTERFACE",
                roughness="Rough",
                thickness=0.5,
                conductivity=3.0,
                density=1250.0,
                specific_heat=1250.0,
                thermal_absorptance=0.9,
                solar_absorptance=0.7,
                visible_absorptance=0.7,
            )
        ],
    )

    for face in ground_zone.faces:
        face: Face
        if face.normal.z == 1:
            face.identifier = f"GROUND_ZONE_UP_{shade_id}"
            face.boundary_condition = boundary_conditions.outdoors
            face.type = face_types.roof_ceiling
            face.properties.energy.construction = ground_top_construction
        elif face.normal.z == -1:
            face.identifier = f"GROUND_ZONE_DOWN_{shade_id}"
            face.boundary_condition = boundary_conditions.ground
            face.type = face_types.floor
            face.properties.energy.construction = ground_interface_construction
        else:
            face.identifier = (
                f"GROUND_ZONE_{face.cardinal_direction().upper()}_{shade_id}"
            )
            face.boundary_condition = boundary_conditions.ground
            face.type = face_types.wall
            face.properties.energy.construction = ground_interface_construction

    return ground_zone
