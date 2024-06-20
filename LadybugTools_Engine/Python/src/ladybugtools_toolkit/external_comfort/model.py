"""Methods for creating the generic external-comfort model.
"""

from honeybee.boundarycondition import boundary_conditions
from honeybee.facetype import face_types
from honeybee.model import Face, Model, Room, Shade
from honeybee_energy.internalmass import InternalMass
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.shade import ShadeConstruction
from honeybee_energy.material.opaque import (
    EnergyMaterial,
    EnergyMaterialVegetation,
    _EnergyMaterialOpaqueBase,
)
from honeybee_radiance.sensorgrid import Sensor, SensorGrid
from ladybug_geometry.geometry3d import Point3D, Vector3D

from .material import _material_equality

_ZONE_WIDTH = 10
_ZONE_DEPTH = 10
_GROUND_THICKNESS = 1
_SHADE_HEIGHT_ABOVE_GROUND = 3
_SHADE_THICKNESS = 0.2


def opaque_to_shade(construction: OpaqueConstruction) -> ShadeConstruction:
    """Convert a Honeybee OpaqueConstruction to a Honeybee ShadeConstruction.

    Args:
        construction (OpaqueConstruction):
            A Honeybee OpaqueConstruction.

    Returns:
        ShadeConstruction:
            A Honeybee ShadeConstruction.

    Raises:
        TypeError: If construction is not of type OpaqueConstruction.
    """

    if not isinstance(construction, OpaqueConstruction):
        raise TypeError("construction must be of type OpaqueConstruction.")

    return ShadeConstruction(
        identifier=f"{construction.identifier}_shade",
        solar_reflectance=construction.outside_solar_reflectance,
        visible_reflectance=construction.outside_visible_reflectance,
        is_specular=False,
    )


def single_layer_construction(
    material: _EnergyMaterialOpaqueBase,
) -> OpaqueConstruction:
    """Create a single layer Honeybee OpaqueConstruction from a Honeybee _EnergyMaterialOpaqueBase object.

    Args:
        material (_EnergyMaterialOpaqueBase):
            A Honeybee _EnergyMaterialOpaqueBase object.

    Returns:
        OpaqueConstruction:
            A single layer Honeybee OpaqueConstruction.

    Raises:
        TypeError: If material is not of type _EnergyMaterialOpaqueBase.
    """

    if not isinstance(material, _EnergyMaterialOpaqueBase):
        raise TypeError("material must be of type _EnergyMaterialOpaqueBase.")

    return OpaqueConstruction(material.identifier, [material])


def _ground_zone(
    construction: OpaqueConstruction,
    shaded: bool = False,
    width: float = _ZONE_WIDTH,
    depth: float = _ZONE_DEPTH,
    thickness: float = _GROUND_THICKNESS,
) -> Room:
    """Create a ground zone with boundary conditions and face identifiers named
         per external comfort workflow.

    Args:
        construction (OpaqueConstruction):
            A construction for the ground zones topmost face.
        shaded (bool, optional):
            A flag to describe whether this zone is shaded. Defaults to False.
        width (float, optional):
            The width (x-dimension) of the ground zone.
        depth (float, optional):
            The depth (y-dimension) of the ground zone.
        thickness (float, optional):
            The thickness (z-dimension) of the ground zone.

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

    internal_mass = InternalMass(
        identifier=f"{ground_zone.identifier}_internal_mass",
        area=ground_zone.exposed_area,
        construction=ground_zone.properties.energy.construction_set.wall_set.exterior_construction,
    )

    ground_zone.properties.energy.add_internal_mass(internal_mass)

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


def _shade_zone(
    construction: OpaqueConstruction,
    width: float = _ZONE_WIDTH,
    depth: float = _ZONE_DEPTH,
    shade_height: float = _SHADE_HEIGHT_ABOVE_GROUND,
    shade_thickness: float = _SHADE_THICKNESS,
) -> Room:
    """Create a shade zone with boundary conditions and face identifiers named
        per external comfort workflow.

    Args:
        construction (OpaqueConstruction):
            A construction for the shade zones faces.
        width (float, optional):
            The width (x-dimension) of the shaded zone.
        depth (float, optional):
            The depth (y-dimension) of the shaded zone.
        shade_height (float, optional):
            The height of the shade.
        shade_thickness (float, optional):
            The thickness of the shade.

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
    # pylint: disable=no-member
    shade_zone.properties.energy.make_plenum(False, True, False)
    # apply high infiltration rate
    shade_zone.properties.energy.absolute_infiltration_ach(30)
    # pylint: enable=no-member

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


def _shade_valence(
    construction: OpaqueConstruction,
    width: float = _ZONE_WIDTH,
    depth: float = _ZONE_DEPTH,
    shade_height: float = _SHADE_HEIGHT_ABOVE_GROUND,
) -> list[Shade]:
    """Create a massless shade around the location being assessed for a shaded
        external comfort condition.

    Args:
        construction (OpaqueConstruction):
            A construction for the shade faces.
        width (float, optional):
            The width (x-dimension) of the shaded zone.
        depth (float, optional):
            The depth (y-dimension) of the shaded zone.
        shade_height (float, optional):
            The height of the shade.

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
        # pylint: disable=no-member
        shade.properties.energy.construction = shade_construction
        # pylint: enable=no-member

    return shades


def create_model(
    identifier: str,
    ground_material: _EnergyMaterialOpaqueBase,
    shade_material: _EnergyMaterialOpaqueBase,
) -> Model:
    """
    Create a model containing geometry describing a shaded and unshaded
        external comfort scenario, including sensor grids for simulation.

    Args:
        identifier (str):
            A unique identifier for the model.
        ground_material (_EnergyMaterialOpaqueBase):
            A surface material for the ground zones topmost face.
        shade_material (_EnergyMaterialOpaqueBase):
            A surface material for the shade zones faces.

    Returns:
        Model:
            A model containing geometry describing a shaded and unshaded
            external comfort scenario, including sensor grids for simulation.
    """

    displacement_vector = Vector3D(y=200)

    # convert materials to single-layer constructions
    ground_construction = single_layer_construction(ground_material)
    shade_construction = single_layer_construction(shade_material)

    # unshaded case
    ground_zone_unshaded = _ground_zone(ground_construction, shaded=False)

    # shaded case
    ground_zone_shaded = _ground_zone(ground_construction, shaded=True)
    ground_zone_shaded.move(displacement_vector)

    shade_zone = _shade_zone(shade_construction)
    shade_zone.move(displacement_vector)

    shades = _shade_valence(shade_construction)
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

    model = Model(
        identifier=identifier,
        rooms=[ground_zone_unshaded, ground_zone_shaded, shade_zone],
        orphaned_shades=shades,
    )

    # assign grids
    model.properties.radiance.sensor_grids = sensor_grids  # pylint: disable=no-member

    return model


def get_ground_material(model: Model) -> _EnergyMaterialOpaqueBase:
    """Get the ground material from a model.

    Args:
        model (Model):
            A honeybee model.

    Returns:
        _EnergyMaterialOpaqueBase:
            The ground material.
    """

    if not isinstance(model, Model):
        raise TypeError("model must be of type Model.")

    return model.faces_by_identifier(["GROUND_ZONE_UP_UNSHADED"])[
        0
    ].properties.energy.construction.materials[0]


def get_shade_material(model: Model) -> _EnergyMaterialOpaqueBase:
    """Get the shade material from a model.

    Args:
        model (Model):
            A honeybee model.

    Returns:
        _EnergyMaterialOpaqueBase:
            The shade material.
    """

    if not isinstance(model, Model):
        raise TypeError("model must be of type Model.")

    return model.faces_by_identifier(["SHADE_ZONE_DOWN"])[
        0
    ].properties.energy.construction.materials[0]


def get_ground_reflectance(model: Model) -> float:
    """Get the ground/floor reflectance from a model.

    Args:
        model (Model):
            A honeybee model.

    Returns:
        float:
            The ground/floor reflectance.
    """

    if not isinstance(model, Model):
        raise TypeError("model must be of type Model.")

    _material: EnergyMaterial | EnergyMaterialVegetation = get_ground_material(model)
    return _material.solar_reflectance


def model_equality(model0: Model, model1: Model) -> bool:
    """Check for equality between two models, with regards to their material
        properties.

    Args:
        model0 (Model):
            A honeybee model.
        model1 (Model):
            A honeybee model.

    Returns:
        bool:
            True if models are equal.
    """

    if not isinstance(model0, Model) or not isinstance(model1, Model):
        raise TypeError("Both inputs must be of type Model.")

    # Check identifiers match
    if model0.identifier != model1.identifier:
        return False

    # Check ground materials match
    gnd_materials_match = _material_equality(
        *[get_ground_material(model) for model in [model0, model1]]
    )

    # Check shade materials match
    shd_materials_match = _material_equality(
        *[get_shade_material(model) for model in [model0, model1]]
    )

    return gnd_materials_match and shd_materials_match
