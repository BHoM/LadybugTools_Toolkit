"""Methods for plotting Ladybug Geometry objects."""
# TODO - add more to this module
import warnings  # pylint: disable=E0401

import matplotlib.pyplot as plt
import numpy as np
from ladybug_geometry.geometry2d import (
    Arc2D,
    LineSegment2D,
    Mesh2D,
    Point2D,
    Polygon2D,
    Polyline2D,
    Ray2D,
    Vector2D,
)

from python_toolkit.bhom.analytics import bhom_analytics


@bhom_analytics()
def plot_lb_geo_2d(
    lb_geometry: tuple[
        Mesh2D
        | Polygon2D
        | Point2D
        | Vector2D
        | LineSegment2D
        | Arc2D
        | Polyline2D
        | Ray2D
    ],
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Example method for creating 2D geometry from LB Geometry objects.

    Args:
        lb_geometry (tuple[Mesh2D | Polygon2D | Point2D | Vector2D | LineSegment2D | Arc2D | Polyline2D | Ray2D]):
            A tuple of LB Geometry objects.
        ax (plt.Axes, optional):
            A matplotlib Axes object to plot on. Defaults to None.
        **kwargs:
            Keyword arguments to pass to the matplotlib plotting methods.

    Returns:
        plt.Axes:
            The matplotlib Axes object.
    """
    warnings.warn(
        "This method is undeveloped and needs splitting into multiple methods."
    )
    if ax is None:
        ax = plt.gca()

    for geo in lb_geometry:
        if isinstance(geo, (Point2D, Vector2D)):
            ax.scatter(geo.x, geo.y)
            continue
        if isinstance(geo, LineSegment2D):
            x_1, y_1 = geo.endpoints[0].to_array()
            x_2, y_2 = geo.endpoints[1].to_array()
            ax.plot([x_1, x_2], [y_1, y_2])
            continue
        if isinstance(geo, Ray2D):
            x_1, y_1 = geo.p.to_array()
            x_2, y_2 = geo.v.to_array()
            ax.plot([x_1, x_2], [y_1, y_2])
            continue
        if isinstance(geo, Polyline2D):
            xs, ys = list(zip(*geo.to_array()))
            ax.plot(xs, ys)
            continue
        if isinstance(geo, Polygon2D):
            xs, ys = [list(i) for i in zip(*geo.to_array())]
            xs.append(xs[0])
            ys.append(ys[0])
            ax.plot(xs, ys)
            continue
        if isinstance(geo, Arc2D):
            xs, ys = list(
                zip(
                    *[
                        geo.reflect(origin=geo.c, normal=Vector2D(1, 0))
                        .rotate(origin=geo.c, angle=np.pi * 1.5)
                        .point_at(i)
                        .to_array()
                        for i in np.linspace(0, 1, 100)
                    ]
                )
            )
            ax.plot(xs, ys)
            continue
        if isinstance(geo, Mesh2D):
            polygons = [Polygon2D.from_array(i) for i in geo.face_vertices]
            for polygon in polygons:
                xs, ys = [list(i) for i in zip(*polygon.to_array())]
                xs.append(xs[0])
                ys.append(ys[0])
                ax.plot(xs, ys)
        else:
            print(f"{type(geo)} not yet supported")

    return ax
