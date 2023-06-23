import uuid
from typing import List

from honeybee.boundarycondition import boundary_conditions
from honeybee.facetype import face_types
from honeybee.model import Face, Model, Room, Shade
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.shade import ShadeConstruction
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from honeybee_radiance.sensorgrid import Sensor, SensorGrid
from ladybug_geometry.geometry3d import Point3D, Vector3D


def opaque_to_shade(construction: OpaqueConstruction) -> ShadeConstruction:
    """Convert an opaque construction to a shade construction."""
    return ShadeConstruction(
        identifier="{construction.identifier}_shade",
        solar_reflectance=construction.outside_solar_reflectance,
        visible_reflectance=construction.outside_visible_reflectance,
        is_specular=False,
    )


def equality(model0: Model, model1: Model, include_identifier: bool = False) -> bool:
    """Check for equality between two models, with regards to their material
        properties.

    Args:
        model0 (Model):
            A honeybee model.
        model1 (Model):
            A honeybee model.
        include_identifier (bool, optional):
            Include the identifier (name) of the model in the quality check.
                Defaults to False.

    Returns:
        bool:
            True if models are equal.
    """

    if not isinstance(model0, Model) or not isinstance(model1, Model):
        raise TypeError("Both inputs must be of type Model.")

    if include_identifier:
        if model0.identifier != model1.identifier:
            return False

    # Check ground material properties
    gnd0_material = model0.faces[5].properties.energy.construction.materials[0]
    gnd1_material = model1.faces[5].properties.energy.construction.materials[0]
    gnd_materials_match: bool = str(gnd0_material) == str(gnd1_material)

    # Check shade material properties
    shd0_material = model0.faces[-6].properties.energy.construction.materials[0]
    shd1_material = model1.faces[-6].properties.energy.construction.materials[0]
    shd_materials_match: bool = str(shd0_material) == str(shd1_material)

    return gnd_materials_match and shd_materials_match


def _create_ground_zone(
    construction: OpaqueConstruction,
    shaded: bool = False,
    width: float = 10,
    depth: float = 10,
    thickness: float = 1,
) -> Room:
    """Create a ground zone with boundary conditions and face identifiers named
         per external comfort workflow.

    Args:
        construction (OpaqueConstruction):
            A construction for the ground zones topmost face.
        shaded (bool, optional):
            A flag to describe whether this zone is shaded. Defaults to False.
        width (float, optional):
            The width (x-dimension) of the ground zone. Defaults to 10m.
        depth (float, optional):
            The depth (y-dimension) of the ground zone. Defaults to 10m.
        thickness (float, optional):
            The thickness (z-dimension) of the ground zone. Defaults to 1m.

    Returns:
        Room:
            A zone representing the ground beneath a person.
    """

    shade_id = "SHADED" if shaded else "UNSHADED"

    ground_zone = Room.from_box(
        identifier=f"GROUND_ZONE_{shade_id}",
        width=width,
        depth=depth,
        height=thickness,
        origin=Point3D(-width / 2, -depth / 2, -thickness),
    )

    # apply ground construction
    ground_zone.properties.energy.make_ground(construction)

    for face in ground_zone.faces:
        face: Face
        if face.normal.z == 1:
            face.identifier = f"GROUND_ZONE_UP_{shade_id}"
            face.properties.radiance.modifier = (
                construction.to_radiance_solar_exterior()
            )
        elif face.normal.z == -1:
            face.identifier = f"GROUND_ZONE_DOWN_{shade_id}"
        else:
            face.identifier = (
                f"GROUND_ZONE_{face.cardinal_direction().upper()}_{shade_id}"
            )
            face.boundary_condition = boundary_conditions.ground
    return ground_zone


def _create_shade_zone(
    construction: OpaqueConstruction,
    width: float = 10,
    depth: float = 10,
    shade_height: float = 3,
    shade_thickness: float = 0.2,
) -> Room:
    """Create a shade zone with boundary conditions and face identifiers named
        per external comfort workflow.

    Args:
        construction (OpaqueConstruction):
            A construction for the shade zones faces.
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
    # convert zone into a plenum
    shade_zone.properties.energy.make_plenum(False, True, False)
    # apply high infiltration rate
    shade_zone.properties.energy.absolute_infiltration_ach(30)

    for face in shade_zone.faces:
        face: Face
        if face.normal.z == 1:
            face.identifier = "SHADE_ZONE_UP"
            face.boundary_condition = boundary_conditions.outdoors
            face.type = face_types.roof_ceiling
            face.properties.energy.construction = construction
            face.properties.radiance.modifier = (
                construction.to_radiance_solar_exterior()
            )
        elif face.normal.z == -1:
            face.identifier = "SHADE_ZONE_DOWN"
            face.boundary_condition = boundary_conditions.outdoors
            face.type = face_types.floor
            face.properties.energy.construction = construction
            face.properties.radiance.modifier = (
                construction.to_radiance_solar_exterior()
            )
        else:
            face.identifier = f"SHADE_ZONE_{face.cardinal_direction().upper()}"
            face.boundary_condition = boundary_conditions.outdoors
            face.type = face_types.wall
            face.properties.energy.construction = construction
            face.properties.radiance.modifier = (
                construction.to_radiance_solar_exterior()
            )

    # ventilate the shade zone

    return shade_zone


def _create_shade_valence(
    construction: OpaqueConstruction,
    width: float = 10,
    depth: float = 10,
    shade_height: float = 3,
) -> List[Shade]:
    """Create a massless shade around the location being assessed for a shaded
        external comfort condition.

    Args:
        construction (OpaqueConstruction):
            A construction for the shade faces.
        width (float, optional):
            The width (x-dimension) of the shaded zone. Defaults to 10m.
        depth (float, optional):
            The depth (y-dimension) of the shaded zone. Defaults to 10m.
        shade_height (float, optional):
            The height of the shade. Default is 3m.

    Returns:
        List[Shade]:
            A list of shading surfaces.
    """

    shade_construction = opaque_to_shade(construction)

    shades = [
        Shade.from_vertices(
            identifier="SHADE_VALENCE_SOUTH",
            vertices=[
                Point3D(-width / 2, -depth / 2, 0),
                Point3D(-width / 2, -depth / 2, shade_height),
                Point3D(width / 2, -depth / 2, shade_height),
                Point3D(width / 2, -depth / 2, 0),
            ],
            is_detached=True,
        ),
        Shade.from_vertices(
            identifier="SHADE_VALENCE_NORTH",
            vertices=[
                Point3D(width / 2, depth / 2, 0),
                Point3D(width / 2, depth / 2, shade_height),
                Point3D(-width / 2, depth / 2, shade_height),
                Point3D(-width / 2, depth / 2, 0),
            ],
            is_detached=True,
        ),
        Shade.from_vertices(
            identifier="SHADE_VALENCE_WEST",
            vertices=[
                Point3D(-width / 2, depth / 2, 0),
                Point3D(-width / 2, depth / 2, shade_height),
                Point3D(-width / 2, -depth / 2, shade_height),
                Point3D(-width / 2, -depth / 2, 0),
            ],
            is_detached=True,
        ),
        Shade.from_vertices(
            identifier="SHADE_VALENCE_EAST",
            vertices=[
                Point3D(width / 2, -depth / 2, 0),
                Point3D(width / 2, -depth / 2, shade_height),
                Point3D(width / 2, depth / 2, shade_height),
                Point3D(width / 2, depth / 2, 0),
            ],
            is_detached=True,
        ),
    ]
    for shade in shades:
        shade.properties.energy.construction = shade_construction

    return shades


def create_model(
    ground_material: _EnergyMaterialOpaqueBase,
    shade_material: _EnergyMaterialOpaqueBase,
    identifier: str = None,
) -> Model:
    """
    Create a model containing geometry describing a shaded and unshaded
        external comfort scenario, including sensor grids for simulation.

    Args:
        ground_material (_EnergyMaterialOpaqueBase):
            A surface material for the ground zones topmost face.
        shade_material (_EnergyMaterialOpaqueBase):
            A surface material for the shade zones faces.
        identifier (str, optional):
            A unique identifier for the model. Defaults to None which will
            generate a unique identifier. This is useful for testing purposes!

    Returns:
        Model:
            A model containing geometry describing a shaded and unshaded
            external comfort scenario, including sensor grids for simulation.
    """
    displacement_vector = Vector3D(y=200)

    # convert materials to single-leayer constructions
    ground_construction = OpaqueConstruction(
        ground_material.identifier, [ground_material]
    )
    shade_construction = OpaqueConstruction(shade_material.identifier, [shade_material])

    # unshaded case
    ground_zone_unshaded = _create_ground_zone(ground_construction, shaded=False)

    # shaded case
    ground_zone_shaded = _create_ground_zone(ground_construction, shaded=True)
    ground_zone_shaded.move(displacement_vector)

    shade_zone = _create_shade_zone(shade_construction)
    shade_zone.move(displacement_vector)

    shades = _create_shade_valence(shade_construction)
    for shade in shades:
        shade.move(displacement_vector)

    # create grids
    sensor_grids = [
        SensorGrid(
            identifier="UNSHADED_UP",
            sensors=[
                Sensor(pos=Point3D(0, 0, 1.2), dir=Point3D(0, 0, 1)),
            ],
        ),
        SensorGrid(
            identifier="UNSHADED_DOWN",
            sensors=[
                Sensor(pos=Point3D(0, 0, 1.2), dir=Point3D(0, 0, -1)),
            ],
        ),
        SensorGrid(
            identifier="SHADED_UP",
            sensors=[
                Sensor(
                    pos=Point3D(0, 0, 1.2).move(displacement_vector),
                    dir=Point3D(0, 0, 1),
                ),
            ],
        ),
        SensorGrid(
            identifier="SHADED_DOWN",
            sensors=[
                Sensor(
                    pos=Point3D(0, 0, 1.2).move(displacement_vector),
                    dir=Point3D(0, 0, -1),
                ),
            ],
        ),
    ]

    if identifier is None:
        identifier = str(uuid.uuid4())

    model = Model(
        identifier=identifier,
        rooms=[ground_zone_unshaded, ground_zone_shaded, shade_zone],
        orphaned_shades=shades,
    )

    # assign grids
    model.properties.radiance.sensor_grids = sensor_grids  # pylint: disable=no-member

    return model
