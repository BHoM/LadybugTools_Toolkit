from typing import List

import matplotlib.pyplot as plt
import numpy as np
from ladybug.psychchart import PsychrometricChart
from ladybug_geometry.geometry2d import Mesh2D
from ladybugtools_toolkit.ladybug_extension.epw import EPW, to_dataframe
from ladybugtools_toolkit.ladybug_extension.location import to_string
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap
from matplotlib.patches import Polygon


def psychrometric(epw: EPW, cmap: Colormap = "viridis") -> plt.Figure:
    """Create a psychrometric chart using a LB backend.

    Args:
        epw (EPW):
            An EPW object.
        cmap (Colormap, optional):
            A colormap to color things with!. Defaults to "viridis".

    Returns:
        plt.Figure:
            A Figure object.
    """

    df = to_dataframe(epw, include_additional=True).droplevel([0, 1], axis=1)

    # create mesh for rendering on chart
    psychart = PsychrometricChart(
        temperature=epw.dry_bulb_temperature,
        relative_humidity=epw.relative_humidity,
        average_pressure=epw.atmospheric_station_pressure.average,
    )

    def lb_mesh_to_patch_collection(
        lb_mesh: Mesh2D, values: List[float] = None, cmap=cmap
    ) -> PatchCollection:
        patch_collection = PatchCollection(
            [Polygon([i.to_array() for i in j]) for j in lb_mesh.face_vertices],
            cmap=cmap,
            zorder=8,
        )
        patch_collection.set_array(values)
        return patch_collection

    p = lb_mesh_to_patch_collection(psychart.colored_mesh, psychart.hour_values)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ll = ax.add_collection(p)

    # add enthalpy lines
    for i in psychart.enthalpy_lines:
        ax.plot(*np.array(i.to_array()).T, c="k", ls=":", lw=0.5, alpha=0.5)
    for pt, txt in list(
        zip(*[psychart.enthalpy_label_points, psychart.enthalpy_labels])
    ):
        _x, _y = pt.to_array()
        ax.text(_x, _y, txt, ha="right", va="bottom", fontsize="x-small")

    # add hr lines
    for i in psychart.hr_lines:
        ax.plot(*np.array(i.to_array()).T, c="k", ls=":", lw=0.5, alpha=0.5)
    for pt, txt in list(zip(*[psychart.hr_label_points, psychart.hr_labels])):
        _x, _y = pt.to_array()
        ax.text(_x, _y, txt, ha="left", va="center", fontsize="small")

    # add rh lines
    for i in psychart.rh_lines:
        ax.plot(*np.array(i.to_array()).T, c="k", ls=":", lw=0.5, alpha=0.5)
    for n, (pt, txt) in enumerate(
        list(zip(*[psychart.rh_label_points, psychart.rh_labels]))
    ):
        if n % 2 == 0:
            continue
        _x, _y = pt.to_array()
        ax.text(_x, _y, txt, ha="right", va="center", fontsize="x-small")

    # add dbt lines
    for i in psychart.temperature_lines:
        ax.plot(*np.array(i.to_array()).T, c="k", ls=":", lw=0.5, alpha=0.5)
    for pt, txt in list(
        zip(*[psychart.temperature_label_points, psychart.temperature_labels])
    ):
        _x, _y = pt.to_array()
        ax.text(_x, _y, txt, ha="center", va="top", fontsize="small")

    # add x axis label
    _x, _y = psychart.x_axis_location.to_array()
    ax.text(_x, _y - 1, psychart.x_axis_text, ha="left", va="top", fontsize="large")

    # add y axis label
    _x, _y = psychart.y_axis_location.to_array()
    ax.text(
        _x + 2,
        _y,
        psychart.y_axis_text,
        ha="right",
        va="top",
        fontsize="large",
        rotation=90,
    )

    # set limits to align
    ax.set_xlim(0, 76)
    ax.set_ylim(0, 50)

    ax.axis("off")

    ax.set_title(to_string(epw.location))

    # Generate peak cooling summary
    clg_vals = df.loc[df.idxmax()["Dry Bulb Temperature (C)"]]
    max_dbt_table = f'Peak cooling {clg_vals.name:%b %d %H:%M}\nWS:  {clg_vals["Wind Speed (m/s)"]:>6.1f} m/s\nWD:  {clg_vals["Wind Direction (degrees)"]:>6.1f} deg\nDBT: {clg_vals["Dry Bulb Temperature (C)"]:>6.1f} °C\nWBT: {clg_vals["Wet Bulb Temperature (C)"]:>6.1f} °C\nRH:  {clg_vals["Relative Humidity (%)"]:>6.1f} %\nDPT: {clg_vals["Dew Point Temperature (C)"]:>6.1f} °C\nh:   {clg_vals["Enthalpy (kJ/kg)"]:>6.1f} kJ/kg\nHR:  {clg_vals["Humidity Ratio (fraction)"]:<5.4f} kg/kg'
    ax.text(
        0,
        0.98,
        max_dbt_table,
        transform=ax.transAxes,
        ha="left",
        va="top",
        zorder=8,
        fontsize="x-small",
        color="#555555",
        **{"fontname": "monospace"},
    )

    # Generate peak heating summary
    htg_vals = df.loc[df.idxmin()["Dry Bulb Temperature (C)"]]
    min_dbt_table = f'Peak heating {htg_vals.name:%b %d %H:%M}\nWS:  {htg_vals["Wind Speed (m/s)"]:>6.1f} m/s\nWD:  {htg_vals["Wind Direction (degrees)"]:>6.1f} deg\nDBT: {htg_vals["Dry Bulb Temperature (C)"]:>6.1f} °C\nWBT: {htg_vals["Wet Bulb Temperature (C)"]:>6.1f} °C\nRH:  {htg_vals["Relative Humidity (%)"]:>6.1f} %\nDPT: {htg_vals["Dew Point Temperature (C)"]:>6.1f} °C\nh:   {htg_vals["Enthalpy (kJ/kg)"]:>6.1f} kJ/kg\nHR:  {htg_vals["Humidity Ratio (fraction)"]:<5.4f} kg/kg'
    ax.text(
        0,
        0.8,
        min_dbt_table,
        transform=ax.transAxes,
        ha="left",
        va="top",
        zorder=8,
        fontsize="x-small",
        color="#555555",
        **{"fontname": "monospace"},
    )

    # Generate max HumidityRatio summary
    hr_vals = df.loc[df.idxmin()["Humidity Ratio (fraction)"]]
    max_hr_table = f'Peak humidity ratio {hr_vals.name:%b %d %H:%M}\nWS:  {hr_vals["Wind Speed (m/s)"]:>6.1f} m/s\nWD:  {hr_vals["Wind Direction (degrees)"]:>6.1f} deg\nDBT: {hr_vals["Dry Bulb Temperature (C)"]:>6.1f} °C\nWBT: {hr_vals["Wet Bulb Temperature (C)"]:>6.1f} °C\nRH:  {hr_vals["Relative Humidity (%)"]:>6.1f} %\nDPT: {hr_vals["Dew Point Temperature (C)"]:>6.1f} °C\nh:   {hr_vals["Enthalpy (kJ/kg)"]:>6.1f} kJ/kg\nHR:  {hr_vals["Humidity Ratio (fraction)"]:<5.4f} kg/kg'
    ax.text(
        0,
        0.62,
        max_hr_table,
        transform=ax.transAxes,
        ha="left",
        va="top",
        zorder=8,
        fontsize="x-small",
        color="#555555",
        **{"fontname": "monospace"},
    )

    # Generate max enthalpy summary
    enth_vals = df.loc[df.idxmin()["Enthalpy (kJ/kg)"]]
    max_enthalpy_table = f'Peak enthalpy {enth_vals.name:%b %d %H:%M}\nWS:  {enth_vals["Wind Speed (m/s)"]:>6.1f} m/s\nWD:  {enth_vals["Wind Direction (degrees)"]:>6.1f} deg\nDBT: {enth_vals["Dry Bulb Temperature (C)"]:>6.1f} °C\nWBT: {enth_vals["Wet Bulb Temperature (C)"]:>6.1f} °C\nRH:  {enth_vals["Relative Humidity (%)"]:>6.1f} %\nDPT: {enth_vals["Dew Point Temperature (C)"]:>6.1f} °C\nh:   {enth_vals["Enthalpy (kJ/kg)"]:>6.1f} kJ/kg\nHR:  {enth_vals["Humidity Ratio (fraction)"]:<5.4f} kg/kg'
    ax.text(
        0,
        0.44,
        max_enthalpy_table,
        transform=ax.transAxes,
        ha="left",
        va="top",
        zorder=8,
        fontsize="x-small",
        color="#555555",
        **{"fontname": "monospace"},
    )

    # add legend
    keys = "WS: Wind speed | WD: Wind direction | DBT: Dry-bulb temperature | WBT: Wet-bulb temperature\nRH: Relative humidity | DPT: Dew-point temperature | h: Enthalpy | HR: Humidity ratio"
    te = ax.text(
        1,
        -0.05,
        keys,
        transform=ax.transAxes,
        ha="right",
        va="top",
        zorder=8,
        fontsize="xx-small",
        color="#555555",
        **{"fontname": "monospace"},
    )

    cbar = plt.colorbar(ll)
    cbar.outline.set_visible(False)
    cbar.set_label("Hours")

    plt.tight_layout()

    return fig
