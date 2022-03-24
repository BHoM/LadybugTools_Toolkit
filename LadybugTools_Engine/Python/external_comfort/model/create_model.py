import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import uuid

from external_comfort.model import _SENSOR_HEIGHT_ABOVE_GROUND
from external_comfort.model.create_ground_zone import create_ground_zone
from external_comfort.model.create_shade_valence import create_shade_valence
from external_comfort.model.create_shade_zone import create_shade_zone
from honeybee.model import Model
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from honeybee_radiance.sensorgrid import Sensor, SensorGrid
from ladybug_geometry.geometry3d import Point3D, Vector3D


def create_model(
    ground_material: _EnergyMaterialOpaqueBase,
    shade_material: _EnergyMaterialOpaqueBase,
    identifier: str = None,
) -> Model:
    """Create a model containing geometry describing a shaded and unshaded external comfort scenario, including sensor grids for simulation.

    Args:
        ground_material (_EnergyMaterialOpaqueBase): A surface material for the ground zones topmost face.
        shade_material (_EnergyMaterialOpaqueBase): A surface material for the shade zones faces.
        identifier (str, optional): A unique identifier for the model. Defaults to None which will generate a unique identifier. This is usefuil for testing purposes!

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
        identifier = f"external_comfort_{uuid.uuid4()}"

    model = Model(
        identifier=identifier,
        rooms=[ground_zone_unshaded, ground_zone_shaded, shade_zone],
        orphaned_shades=shades,
    )

    model.properties.radiance.sensor_grids = [unshaded_grid]

    return model
