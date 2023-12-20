"""Methods for handling honeybee models."""
# pylint: disable=R0911
from enum import Enum, auto  # pylint: disable=E0401

import matplotlib.pyplot as plt
from honeybee.model import AirBoundary, Face3D, Floor, Model, RoofCeiling, Wall
from ladybug_geometry.geometry3d import LineSegment3D, Plane
from matplotlib.collections import PolyCollection
import numpy as np


class HbModelGeometry(Enum):
    """Honeybee model boundary types."""

    WALL = auto()
    FLOOR = auto()
    ROOFCEILING = auto()
    AIRBOUNDARY = auto()
    SHADE = auto()
    APERTURE = auto()
    DOOR = auto()

    def get_geometry(self, model: Model) -> list[object]:
        """Return the objects of a model that correspond to the type of object."""
        match self:
            case HbModelGeometry.WALL:
                return [i.geometry for i in model.faces if isinstance(i.type, Wall)]
            case HbModelGeometry.FLOOR:
                return [i.geometry for i in model.faces if isinstance(i.type, Floor)]
            case HbModelGeometry.ROOFCEILING:
                return [
                    i.geometry for i in model.faces if isinstance(i.type, RoofCeiling)
                ]
            case HbModelGeometry.AIRBOUNDARY:
                return [
                    i.geometry for i in model.faces if isinstance(i.type, AirBoundary)
                ]
            case HbModelGeometry.SHADE:
                return [i.geometry for i in model.shades]
            case HbModelGeometry.APERTURE:
                return [i.geometry for i in model.apertures]
            case HbModelGeometry.DOOR:
                return [i.geometry for i in model.doors]
            case _:
                raise ValueError(f"Invalid SliceColoring: {self}")

    def slice_polycollection(self, model: Model, plane: Plane) -> PolyCollection:
        """Slice a model with a plane and return a matplotlib PolyCollection.

        Args:
            model (Model): A Honeybee model to slice.
            plane (Plane): A Ladybug 3D Plane object to slice the model with.

        Returns:
            PolyCollection: A matplotlib PolyCollection object.
        """

        geometries = self.get_geometry(model)
        line_segments = plane_intersections(geometries, plane)

        return linesegments_to_polycollection(
            line_segments=line_segments,
            facecolor=self.facecolor,
            edgecolor=self.edgecolor,
            alpha=self.alpha,
            linewidth=self.linewidth,
            zorder=self.zorder,
        )

    @property
    def facecolor(self) -> str:
        """Return the facecolor for the geometry type."""
        match self:
            case HbModelGeometry.WALL:
                return "#E6B43C"
            case HbModelGeometry.FLOOR:
                return "#808080"
            case HbModelGeometry.ROOFCEILING:
                return "#801414"
            case HbModelGeometry.AIRBOUNDARY:
                return "#FFFFC8"
            case HbModelGeometry.SHADE:
                return "#784BBE"
            case HbModelGeometry.APERTURE:
                return "#40B4FF"
            case HbModelGeometry.DOOR:
                return "#A09664"
            case _:
                raise ValueError(f"Invalid SliceColoring: {self}")

    @property
    def edgecolor(self) -> str:
        """Return the edgecolor for the geometry type."""
        match self:
            case HbModelGeometry.WALL:
                return "#E6B43C"
            case HbModelGeometry.FLOOR:
                return "#808080"
            case HbModelGeometry.ROOFCEILING:
                return "#801414"
            case HbModelGeometry.AIRBOUNDARY:
                return "#FFFFC8"
            case HbModelGeometry.SHADE:
                return "#784BBE"
            case HbModelGeometry.APERTURE:
                return "#40B4FF"
            case HbModelGeometry.DOOR:
                return "#A09664"
            case _:
                raise ValueError(f"Invalid SliceColoring: {self}")

    @property
    def alpha(self) -> float:
        """Return the alpha for the geometry type."""
        match self:
            case HbModelGeometry.WALL:
                return 1
            case HbModelGeometry.FLOOR:
                return 1
            case HbModelGeometry.ROOFCEILING:
                return 1
            case HbModelGeometry.AIRBOUNDARY:
                return 1
            case HbModelGeometry.SHADE:
                return 1
            case HbModelGeometry.APERTURE:
                return 1
            case HbModelGeometry.DOOR:
                return 1
            case _:
                raise ValueError(f"Invalid SliceColoring: {self}")

    @property
    def linewidth(self) -> float:
        """Return the linewidth for the geometry type."""
        match self:
            case HbModelGeometry.WALL:
                return 1
            case HbModelGeometry.FLOOR:
                return 1
            case HbModelGeometry.ROOFCEILING:
                return 1
            case HbModelGeometry.AIRBOUNDARY:
                return 1
            case HbModelGeometry.SHADE:
                return 1
            case HbModelGeometry.APERTURE:
                return 1
            case HbModelGeometry.DOOR:
                return 1
            case _:
                raise ValueError(f"Invalid SliceColoring: {self}")

    @property
    def zorder(self) -> int:
        """Set the default zorder of the geometry type."""
        match self:
            case HbModelGeometry.WALL:
                return 6
            case HbModelGeometry.FLOOR:
                return 5
            case HbModelGeometry.ROOFCEILING:
                return 5
            case HbModelGeometry.AIRBOUNDARY:
                return 4
            case HbModelGeometry.SHADE:
                return 5
            case HbModelGeometry.APERTURE:
                return 7
            case HbModelGeometry.DOOR:
                return 7
            case _:
                raise ValueError(f"Invalid SliceColoring: {self}")


def plane_intersections(
    geometries: list[Face3D],
    plane: Plane,
) -> list[LineSegment3D]:
    """Return the LineSegment3D objects that result from slicing a set of geometries with a plane.

    Args:
        geometries (list[Face3D]): A list of Face3D objects to slice.
        plane (Plane): A Ladybug 3D Plane object to slice the model with.

    Returns:
        list[LineSegment3D]: A list of LineSegment3D objects resulting from the slice.
    """

    line_segments = []
    for geometry in geometries:
        try:
            line_segments.extend(geometry.intersect_plane(plane))
        except TypeError:
            continue
    return line_segments


def linesegments_to_polycollection(
    line_segments: list[LineSegment3D], plane: Plane = None, **kwargs
) -> PolyCollection:
    """Convert a list of LineSegment3D objects to a matplotlib PolyCollection.

    Args:
        line_segments (list[LineSegment3D]):
            A list of LineSegment3D objects.
        plane (Plane, optional):
            A Ladybug 3D Plane object to project the segments onto. Defaults to None which does not project the
            segments and instead asssumes the line segments are in an XY plane.
        **kwargs:
            Additional keyword arguments to pass to the matplotlib PolyCollection constructor.

    Returns:
        PolyCollection:
            A matplotlib PolyCollection object.
    """

    plane = Plane() if plane is None else plane

    _vertices = []
    for segment in line_segments:
        _vertices.append(
            [
                plane.xyz_to_xy(segment.p1).to_array(),
                plane.xyz_to_xy(segment.p2).to_array(),
            ]
        )
    return PolyCollection(verts=_vertices, closed=False, **kwargs)


def slice_geometry(
    hb_objects: list[object], plane: Plane, ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    """Slice a set of Honeybee objects with a plane and plot their intersections.

    Args:
        hb_objects (list[object]):
            A set of Honeybee objects to slice. These must have a geometry attribute.
        plane (Plane):
            A Ladybug 3D Plane object to slice the objects with.
        ax (plt.Axes, optional):
            The matplotlib axes to plot the intersections on. Defaults to None which uses the current axes.
        **kwargs:
            Additional keyword arguments to pass to the matplotlib PolyCollection constructor.

    Returns:
        plt.Axes:
            A matplotlib axes with the intersections plotted.
    """

    if ax is None:
        ax = plt.gca()

    _vertices = []
    for obj in hb_objects:
        segments: list[LineSegment3D] = obj.geometry.intersect_plane(plane)
        if segments is None:
            continue
        _vertices.extend(
            [
                [plane.xyz_to_xy(pt).to_array() for pt in segment.vertices]
                for segment in segments
            ]
        )
    _vertices = np.array(_vertices)
    ax.add_artist(
        PolyCollection(
            verts=_vertices,
            closed=False,
            fc=None,
            **kwargs,
        ),
    )

    ax.autoscale_view()
    ax.set_aspect("equal")

    return ax


def slice_model(model: Model, plane: Plane, ax: plt.Axes = None) -> plt.Axes:
    """Slice a Honeybee model with a plane and plot the intersections.

    Args:
        model (Model): A Honeybee model to slice.
        plane (Plane): A Ladybug 3D Plane object to slice the model with.
        ax (plt.Axes, optional): The matplotlib axes to plot the intersections on.
        Defaults to None which uses the current axes.

    Returns:
        plt.Axes: A matplotlib axes with the intersections plotted.
    """

    if ax is None:
        ax = plt.gca()

    meta = {
        "wall": {
            "objects": [i for i in model.faces if isinstance(i.type, Wall)],
            "poly_kwargs": {
                "color": "grey",
                "zorder": 4,
                "alpha": 1,
                "lw": 0.5,
            },
        },
        "floor": {
            "objects": [i for i in model.faces if isinstance(i.type, Floor)],
            "poly_kwargs": {
                "color": "black",
                "zorder": 5,
                "alpha": 1,
                "lw": 0.5,
            },
        },
        "roofceiling": {
            "objects": [i for i in model.faces if isinstance(i.type, RoofCeiling)],
            "poly_kwargs": {
                "color": "brown",
                "zorder": 5,
                "alpha": 1,
                "lw": 0.5,
            },
        },
        "airboundary": {
            "objects": [i for i in model.faces if isinstance(i.type, AirBoundary)],
            "poly_kwargs": {
                "color": "pink",
                "zorder": 3,
                "alpha": 1,
                "lw": 0.5,
            },
        },
        "shade": {
            "objects": model.shades,
            "poly_kwargs": {
                "color": "green",
                "zorder": 3,
                "alpha": 1,
                "lw": 0.5,
            },
        },
        "aperture": {
            "objects": model.apertures,
            "poly_kwargs": {
                "color": "blue",
                "zorder": 6,
                "alpha": 1,
                "lw": 0.5,
            },
        },
    }

    for _, v in meta.items():
        slice_geometry(hb_objects=v["objects"], plane=plane, ax=ax, **v["poly_kwargs"])

    ax.autoscale_view()
    ax.set_aspect("equal")

    return ax
