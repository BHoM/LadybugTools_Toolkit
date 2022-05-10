_WIDTH = 10
_DEPTH = 10
_GROUND_HEIGHT = 1
_SHADE_HEIGHT = 0.2
_SHADE_HEIGHT_ABOVE_GROUND = 3
_SENSOR_HEIGHT_ABOVE_GROUND = 1.2

import uuid
from typing import List

from honeybee.boundarycondition import boundary_conditions
from honeybee.facetype import face_types
from honeybee.model import Face, Model, Room, Shade
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.material.opaque import EnergyMaterial, _EnergyMaterialOpaqueBase
from honeybee_radiance.sensorgrid import Sensor, SensorGrid
from ladybug_geometry.geometry3d import Point3D, Vector3D


def create_ground_zone(
    material: _EnergyMaterialOpaqueBase, shaded: bool = False
) -> Room:
    """Create a ground zone with boundary conditions and face identifiers named per external comfort workflow.

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


def create_model(
    ground_material: _EnergyMaterialOpaqueBase,
    shade_material: _EnergyMaterialOpaqueBase,
    identifier: str = None,
) -> Model:
    """Create a model containing geometry describing a shaded and unshaded external comfort scenario, including sensor grids for simulation.

    Args:
        ground_material (_EnergyMaterialOpaqueBase): A surface material for the ground zones topmost face.
        shade_material (_EnergyMaterialOpaqueBase): A surface material for the shade zones faces.
        identifier (str, optional): A unique identifier for the model. Defaults to None which will generate a unique identifier. This is useful for testing purposes!

    Returns:
        Model: A model containing geometry describing a shaded and unshaded external comfort scenario, including sensor grids for simulation.
    """
    displacement_vector = Vector3D(0, 500, 0)

    sensor_grid = SensorGrid(
        identifier="_",
        sensors=[
            Sensor(
                pos=Point3D(0, 0, _SENSOR_HEIGHT_ABOVE_GROUND), dir=Point3D(0, 0, 1)
            ),
            Sensor(
                pos=Point3D(0, 0, _SENSOR_HEIGHT_ABOVE_GROUND), dir=Point3D(0, 0, -1)
            ),
        ],
    )

    # unshaded case
    ground_zone_unshaded = create_ground_zone(ground_material, shaded=False)

    unshaded_grid = sensor_grid.duplicate()
    unshaded_grid.identifier = "UNSHADED"

    # shaded case
    ground_zone_shaded = create_ground_zone(ground_material, shaded=True)
    ground_zone_shaded.move(displacement_vector)

    shade_zone = create_shade_zone(shade_material)
    shade_zone.move(displacement_vector)

    shades = create_shade_valence()
    [i.move(displacement_vector) for i in shades]

    if identifier is None:
        identifier = f"{str(uuid.uuid4())[:8]}"

    model = Model(
        identifier=identifier,
        rooms=[ground_zone_unshaded, ground_zone_shaded, shade_zone],
        orphaned_shades=shades,
    )

    model.properties.radiance.sensor_grids = [unshaded_grid]

    return model


def create_shade_valence() -> List[Shade]:
    """Create a massless shade around the location being assessed for a shaded external comfort condition.

    Returns:
        List[Shade]: A list of shading surfaces.
    """

    shades = [
        Shade.from_vertices(
            identifier=f"SHADE_VALENCE_SOUTH",
            vertices=[
                Point3D(-_WIDTH / 2, -_DEPTH / 2, 0),
                Point3D(-_WIDTH / 2, -_DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(_WIDTH / 2, -_DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(_WIDTH / 2, -_DEPTH / 2, 0),
            ],
        ),
        Shade.from_vertices(
            identifier=f"SHADE_VALENCE_NORTH",
            vertices=[
                Point3D(_WIDTH / 2, _DEPTH / 2, 0),
                Point3D(_WIDTH / 2, _DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(-_WIDTH / 2, _DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(-_WIDTH / 2, _DEPTH / 2, 0),
            ],
        ),
        Shade.from_vertices(
            identifier=f"SHADE_VALENCE_WEST",
            vertices=[
                Point3D(-_WIDTH / 2, _DEPTH / 2, 0),
                Point3D(-_WIDTH / 2, _DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(-_WIDTH / 2, -_DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(-_WIDTH / 2, -_DEPTH / 2, 0),
            ],
        ),
        Shade.from_vertices(
            identifier=f"SHADE_VALENCE_EAST",
            vertices=[
                Point3D(_WIDTH / 2, -_DEPTH / 2, 0),
                Point3D(_WIDTH / 2, -_DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(_WIDTH / 2, _DEPTH / 2, _SHADE_HEIGHT_ABOVE_GROUND),
                Point3D(_WIDTH / 2, _DEPTH / 2, 0),
            ],
        ),
    ]

    return shades


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
        if face.normal.z == 1:
            face.identifier = f"SHADE_ZONE_UP"
            face.boundary_condition = boundary_conditions.outdoors
            face.type = face_types.roof_ceiling
            face.properties.energy.construction = shade_construction
        elif face.normal.z == -1:
            face.identifier = f"SHADE_ZONE_DOWN"
            face.boundary_condition = boundary_conditions.ground
            face.type = face_types.floor
            face.properties.energy.construction = shade_construction
        else:
            face.identifier = f"SHADE_ZONE_{face.cardinal_direction().upper()}"
            face.boundary_condition = boundary_conditions.ground
            face.type = face_types.wall
            face.properties.energy.construction = shade_construction

    return shade_zone
