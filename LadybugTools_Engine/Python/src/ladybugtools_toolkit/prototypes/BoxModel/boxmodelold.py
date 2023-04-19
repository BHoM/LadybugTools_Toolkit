from honeybee.model import Model, Room
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_vtk.model import Model as md
from ladybug_geometry.geometry3d import Vector3D

IDENTIFIER_BASE = "HB_BoxModelTest"
_SENSOR_GRID_FLOOR_OFFSET = 0.8

_ORIENTATION = 128
_BAY_WIDTH = 4
_BAY_DEPTH = 10
_BAY_HEIGHT = 3
_ZONE_WIDTH = 20
_GLAZING_RATIO = 0.5
_APERTURE_HEIGHT = 2.2
_SILL_HEIGHT = 0.5
_APERTURE_SEPARATION = 3
_SENSOR_GRID_SPACING = 0.5
_WALL_THICKNESS = 0.4
_LOUVER_COUNT = 1
_LOUVER_DEPTH = 0.5

_ZONE_DEPTH = _BAY_DEPTH
_ZONE_HEIGHT = _BAY_HEIGHT


def main():

    # create facade zone
    zone = Room.from_box(
        identifier="{0:}_Zone".format(IDENTIFIER_BASE),
        width=_ZONE_WIDTH,
        height=_ZONE_HEIGHT,
        depth=_ZONE_DEPTH,
    )
    zone.move(Vector3D(-_ZONE_WIDTH / 2, -_ZONE_DEPTH / 2, 0))

    # create analysis bay within facade zone
    bay = Room.from_box(
        identifier="{0:}_Bay".format(IDENTIFIER_BASE),
        width=_BAY_WIDTH,
        height=_ZONE_HEIGHT,
        depth=_BAY_DEPTH,
    )
    bay.move(Vector3D(-_BAY_WIDTH / 2, -_BAY_DEPTH / 2, 0))
    bay.scale(
        factor=1 - (_WALL_THICKNESS / _BAY_DEPTH * 2),
        origin=bay.geometry.faces[0].center,
    )  # scale the bay based on the wall depth - to remove points from within a certain distance of the bounding walls

    # add glazing to "north" face of room (it will be rotated later to face the right direction)
    zone.faces[1].apertures_by_ratio_rectangle(
        ratio=_GLAZING_RATIO,
        aperture_height=_APERTURE_HEIGHT,
        sill_height=_SILL_HEIGHT,
        horizontal_separation=_APERTURE_SEPARATION,
    )

    # add wall thickness
    wall_depth_shades = []
    for aperture in zone.faces[1].apertures:
        wall_depth_shades.append(aperture.extruded_border(_WALL_THICKNESS, True))

    # add external shades
    louver_shades = []
    for aperture in zone.faces[1].apertures:
        louver_shades.append(
            aperture.louvers_by_count(
                louver_count=_LOUVER_COUNT, depth=_LOUVER_DEPTH, offset=0, angle=0
            )
        )

    # create sensor grid for bay
    floor_face = bay.geometry.faces[0].flip()
    sensor_grid = SensorGrid.from_face3d(
        identifier="{0:}_SensorGrid".format(IDENTIFIER_BASE),
        faces=[floor_face],
        x_dim=_SENSOR_GRID_SPACING,
        y_dim=_SENSOR_GRID_SPACING,
        offset=_SENSOR_GRID_FLOOR_OFFSET,
    )

    model = Model.from_objects(
        identifier="{0:}_Model".format(IDENTIFIER_BASE), objects=[zone]
    )

    # add sensor grid to model
    model.properties.radiance.add_sensor_grids([sensor_grid])

    # rotate model to requested orientation
    model.rotate_xy(angle=-_ORIENTATION, origin=zone.center)

    # write model to disk
    #    print(model.to_hbjson(IDENTIFIER_BASE, indent=4))

    return model


model = main()

test_web = md.from_hbjson(model.to_hbjson(name="Test"))
test_web.to_html(folder=".", name="two-rooms", show=True)
test_web.to_vtkjs(folder=".", name="two-rooms")
