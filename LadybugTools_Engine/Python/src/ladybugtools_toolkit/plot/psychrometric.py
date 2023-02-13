import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import BaseCollection
from ladybug.psychchart import PsychrometricChart
from ladybug_comfort.chart.polygonpmv import PMVParameter, PolygonPMV
from ladybug_geometry.geometry2d import LineSegment2D, Mesh2D, Polyline2D
from ladybugtools_toolkit.ladybug_extension.analysis_period import (
    describe,
    to_datetimes,
)
from ladybugtools_toolkit.ladybug_extension.epw import EPW, to_dataframe
from ladybugtools_toolkit.ladybug_extension.location import to_string
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap
from matplotlib.patches import Polygon


def strategy_warning(polygon_name):
    """Give a warning about a polygon not fitting on the chart."""
    msg = f'Polygon "{polygon_name}" could not fit on the chart.\nTry moving the comfort polygon(s) by changingits criteria.'
    warn(msg)


@dataclass(init=True, repr=True)
class PassiveStrategyParameters:
    """Parameters for passive strategies for reducing thermal discomfort.

    Args:
        day_above_comfort: The number of degrees above the comfort temperature that
            the day temperature is allowed to be. (Default: 12)
        night_below_comfort: The number of degrees below the comfort temperature that
            the night temperature is allowed to be. (Default: 3)
        fan_air_speed: The air speed in m/s that the fan is assumed to be blowing
            at. (Default: 1)
        balance_temperature: The temperature in degrees Celcius at which the
            balance point between sensible and latent heat transfer occurs.
            (Default: 12.8)
        solar_heat_capacity: The heat capacity of the solar radiation in J/m2/K.
            (Default: 50)
        time_constant: The time constant of the thermal mass in hours.
            (Default: 8)
    """

    day_above_comfort: float = field(init=True, default=12)
    night_below_comfort: float = field(init=True, default=3)
    fan_air_speed: float = field(init=True, default=1)
    balance_temperature: float = field(init=True, default=12.8)
    solar_heat_capacity: float = field(init=True, default=50)
    time_constant: float = field(init=True, default=8)

    def __post_init__(self):
        if not 0 <= self.day_above_comfort <= 30:
            warn(
                "day_above_comfort must be between 0 and 30 (inclusive) - reverting to default of 12."
            )
            self.day_above_comfort = 12
        if not 0 <= self.night_below_comfort <= 15:
            warn(
                "night_below_comfort must be between 0 and 15 (inclusive) - reverting to default of 3."
            )
        if not 0.1 <= self.fan_air_speed <= 10:
            warn(
                "fan_air_speed must be between 0.1 and 10 (inclusive) - reverting to default of 1."
            )
        if not 5 <= self.balance_temperature <= 20:
            warn(
                "balance_temperature must be between 0 and 15 (inclusive) - reverting to default of 12.8."
            )
        if not 1 <= self.solar_heat_capacity <= 1000:
            warn(
                "solar_heat_capacity must be between 1 and 1000 (inclusive) - reverting to default of 50."
            )
        if not 1 <= self.time_constant <= 48:
            warn(
                "time_constant must be between 1 and 48 (inclusive) - reverting to default of 8."
            )


class PassiveStrategy(Enum):
    """Passive strategies for reducing thermal discomfort."""

    EVAPORATIVE_COOLING = "Evaporative Cooling"
    MASS_NIGHT_VENTILATION = "Mass + Night Vent"
    OCCUPANT_FAN_USE = "Occupant Use of Fans"
    INTERNAL_HEAT_CAPTURE = "Capture Internal Heat"
    PASSIVE_SOLAR_HEATING = "Passive Solar Heating"


@dataclass(init=True, repr=True)
class PsychrometricPolygons:
    """A collection of polygons for use in a psychrometric chart.

    Args:
        strategies (List[PassiveStrategy], optional):
            A list of passive strategies to include in the chart. Default is None.
        strategy_parameters (PassiveStrategyParameters, optional):
            A PassiveStrategyParameters object to use for the passive strategies.
        pmv_parameter (PMVParameter, optional):
            A PMVParameter object to use for the comfort polygon.
        mean_radiant_temperature (float, optional):
            A mean radiant temperature to use for the comfort polygon. Default is None.
        air_speed (float, optional):
            An air speed to use for the comfort polygon. Default is 0.1.
        metabolic_rate (float, optional):
            A metabolic rate to use for the comfort polygon. Default is 1.1.
        clo_value (float, optional):
            A clothing value to use for the comfort polygon. Default is 0.7.

    Returns:
        PsychrometricPolygons: A collection of polygons for use in a psychrometric chart.
    """

    strategies: List[PassiveStrategy] = field(init=True, default_factory=list)
    strategy_parameters: PassiveStrategyParameters = field(
        init=True, default=PassiveStrategyParameters()
    )
    pmv_parameter: PMVParameter = field(init=True, default=PMVParameter())
    mean_radiant_temperature: float = field(init=True, default=None)
    air_speed: float = field(init=True, default=0.1)
    metabolic_rate: float = field(init=True, default=1.1)
    clo_value: float = field(init=True, default=0.7)

    def __post_init__(self):
        if len(self.strategies) >= 1:
            assert all(
                isinstance(i, PassiveStrategy) for i in self.strategies
            ), f"PassiveStrategy not of correct type. Use {PassiveStrategy}"


def psychrometric(
    epw: EPW,
    cmap: Colormap = "viridis",
    analysis_period: AnalysisPeriod = None,
    wet_bulb: bool = False,
    psychro_polygons: PsychrometricPolygons = None,
) -> plt.Figure:
    """Create a psychrometric chart using a LB backend.

    Args:
        epw (EPW):
            An EPW object.
        cmap (Colormap, optional):
            A colormap to color things with!. Defaults to "viridis".
        analysis_period (AnalysisPeriod, optional):
            An analysis period to filter values by. Default is whole year.
        wet_bulb (bool, optional):
            Plot wet-bulb temperature constant lines instead of enthalpy. Default is False.

    Returns:
        plt.Figure:
            A Figure object.
    """

    if analysis_period is None:
        analysis_period = AnalysisPeriod()

    df = (
        to_dataframe(epw, include_additional=True)
        .droplevel([0, 1], axis=1)
        .loc[to_datetimes(analysis_period)]
    )

    # create mesh for rendering on chart
    psychart = PsychrometricChart(
        temperature=epw.dry_bulb_temperature.filter_by_analysis_period(analysis_period),
        relative_humidity=epw.relative_humidity.filter_by_analysis_period(
            analysis_period
        ),
        average_pressure=epw.atmospheric_station_pressure.filter_by_analysis_period(
            analysis_period
        ).average,
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

    if wet_bulb:
        # add wet-bulb lines
        for i in psychart.wb_lines:
            ax.plot(*np.array(i.to_array()).T, c="k", ls=":", lw=0.5, alpha=0.5)
        for pt, txt in list(zip(*[psychart.wb_label_points, psychart.wb_labels])):
            _x, _y = pt.to_array()
            ax.text(_x, _y, txt, ha="right", va="bottom", fontsize="x-small")
    else:
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
    ax.set_ylim(-0.01, 50)

    ax.axis("off")

    ax.set_title(f"{to_string(epw.location)}\n{describe(analysis_period)}")

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
    ax.text(
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

    # add polygon if polgyon passed
    if psychro_polygons is not None:

        polygon_data = []
        polygon_names = []

        def line_objs_to_vertices(
            lines: List[Union[Polyline2D, LineSegment2D]]
        ) -> List[List[float]]:

            # ensure input list is flat
            lines = [
                v
                for item in lines
                for v in (item if isinstance(item, list) else [item])
            ]

            # iterate list to obtain point2d objects
            point_2ds = []
            for line_obj in lines:
                if isinstance(line_obj, LineSegment2D):
                    point_2ds.extend(i.to_array() for i in line_obj.vertices)
                if isinstance(line_obj, Polyline2D):
                    point_2ds.extend(i.to_array() for i in line_obj.vertices)

            # remove duplicates
            vvertices = []
            for v in point_2ds:
                if v not in vvertices:
                    vvertices.append(v)

            # obtain any LineSegment2D objects
            return vvertices

        def process_polygon(polygon_name, polygon):
            """Process a strategy polygon that does not require any special treatment."""

            if polygon is not None:
                strategy_poly = line_objs_to_vertices(polygon)
                dat = poly_obj.evaluate_polygon(polygon, 0.01)
                dat = (
                    dat[0]
                    if len(dat) == 1
                    else poly_obj.create_collection(dat, polygon_name)
                )
            else:
                strategy_warning(polygon_name)
                return None, None, None

            return polygon_name, strategy_poly, dat

        def polygon_area(xs, ys):
            """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
            # https://stackoverflow.com/a/30408825/7128154
            return 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))

        def polygon_centroid(xs, ys):
            """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
            xy = np.array([xs, ys])
            c = np.dot(
                xy + np.roll(xy, 1, axis=1), xs * np.roll(ys, 1) - np.roll(xs, 1) * ys
            ) / (6 * polygon_area(xs, ys))
            return c

        def merge_polygon_data(poly_data):
            """Merge an array of polygon comfort conditions into a single data list."""
            val_mtx = [dat.values for dat in poly_data]
            merged_values = []
            for hr_data in zip(*val_mtx):
                hr_val = 1 if 1 in hr_data else 0
                merged_values.append(hr_val)
            return merged_values

        poly_obj = PolygonPMV(
            psychart,
            rad_temperature=[psychro_polygons.mean_radiant_temperature],
            air_speed=[psychro_polygons.air_speed],
            met_rate=[psychro_polygons.metabolic_rate],
            clo_value=[psychro_polygons.clo_value],
        )

        # add generic comfort polygon
        poly = line_objs_to_vertices(poly_obj.merged_comfort_polygon)
        dat = poly_obj.merged_comfort_data
        name = "Comfort"
        polygon_names.append(name)
        polygon_data.append(dat)
        ax.add_collection(
            PatchCollection(
                [Polygon(poly)],
                fc="none",
                ec="black",
                zorder=8,
                lw=1,
                alpha=0.5,
                ls="--",
            )
        )
        xx, yy = polygon_centroid(*np.array(poly).T)
        ax.text(
            xx,
            yy,
            "\n".join(textwrap.wrap("Comfort", 12)),
            fontsize="xx-small",
            c="black",
            ha="center",
            va="center",
            zorder=8,
        )

        # add strategy polygons
        if PassiveStrategy.EVAPORATIVE_COOLING in psychro_polygons.strategies:
            ec_poly = poly_obj.evaporative_cooling_polygon()
            name, poly, dat = process_polygon(
                PassiveStrategy.EVAPORATIVE_COOLING.value, ec_poly
            )
            if not all(i is None for i in [name, poly, dat]):
                polygon_data.append(dat)
                polygon_names.append(name)
                ax.add_collection(
                    PatchCollection(
                        [Polygon(poly)],
                        fc="none",
                        ec="blue",
                        zorder=8,
                        lw=1,
                        alpha=0.5,
                        ls="--",
                    )
                )
                xx, yy = polygon_centroid(*np.array(poly).T)
                ax.text(
                    xx,
                    yy,
                    "\n".join(textwrap.wrap(name, 12)),
                    fontsize="xx-small",
                    c="blue",
                    ha="center",
                    va="center",
                    zorder=8,
                )

        if PassiveStrategy.MASS_NIGHT_VENTILATION in psychro_polygons.strategies:
            nf_poly = poly_obj.night_flush_polygon(
                psychro_polygons.strategy_parameters.day_above_comfort
            )
            if nf_poly is not None:
                name = PassiveStrategy.MASS_NIGHT_VENTILATION.value
                poly = line_objs_to_vertices(nf_poly)
                dat = poly_obj.evaluate_night_flush_polygon(
                    nf_poly,
                    epw.dry_bulb_temperature,
                    psychro_polygons.strategy_parameters.night_below_comfort,
                    psychro_polygons.strategy_parameters.time_constant,
                    0.01,
                )
                dat = dat[0] if len(dat) == 1 else poly_obj.create_collection(dat, name)
                polygon_data.append(dat)
                polygon_names.append(name)
                ax.add_collection(
                    PatchCollection(
                        [Polygon(poly)],
                        fc="none",
                        ec="purple",
                        zorder=8,
                        lw=1,
                        alpha=0.5,
                        ls="--",
                    )
                )
                xx, yy = polygon_centroid(*np.array(poly).T)
                ax.text(
                    xx,
                    yy,
                    "\n".join(textwrap.wrap(name, 12)),
                    fontsize="xx-small",
                    c="purple",
                    ha="center",
                    va="center",
                    zorder=8,
                )
            else:
                strategy_warning(name)

        if PassiveStrategy.INTERNAL_HEAT_CAPTURE in psychro_polygons.strategies:
            iht_poly = poly_obj.internal_heat_polygon(
                psychro_polygons.strategy_parameters.balance_temperature
            )
            name, poly, dat = process_polygon(
                PassiveStrategy.INTERNAL_HEAT_CAPTURE.value, iht_poly
            )
            if not all(i is None for i in [name, poly, dat]):
                polygon_data.append(dat)
                polygon_names.append(name)
                ax.add_collection(
                    PatchCollection(
                        [Polygon(poly)],
                        fc="none",
                        ec="orange",
                        zorder=8,
                        lw=1,
                        alpha=0.5,
                        ls="--",
                    )
                )
                xx, yy = polygon_centroid(*np.array(poly).T)
                ax.text(
                    xx,
                    yy,
                    "\n".join(textwrap.wrap(name, 12)),
                    fontsize="xx-small",
                    c="orange",
                    ha="center",
                    va="center",
                    zorder=8,
                )

        if PassiveStrategy.OCCUPANT_FAN_USE in psychro_polygons.strategies:
            fan_poly = poly_obj.fan_use_polygon(
                psychro_polygons.strategy_parameters.balance_temperature
            )
            name, poly, dat = process_polygon(
                PassiveStrategy.OCCUPANT_FAN_USE.value, fan_poly
            )
            if not all(i is None for i in [name, poly, dat]):
                polygon_data.append(dat)
                polygon_names.append(name)
                ax.add_collection(
                    PatchCollection(
                        [Polygon(poly)],
                        fc="none",
                        ec="cyan",
                        zorder=8,
                        lw=1,
                        alpha=0.5,
                        ls="--",
                    )
                )
                xx, yy = polygon_centroid(*np.array(poly).T)
                ax.text(
                    xx,
                    yy,
                    "\n".join(textwrap.wrap(name, 12)),
                    fontsize="xx-small",
                    c="cyan",
                    ha="center",
                    va="center",
                    zorder=8,
                )

        if PassiveStrategy.PASSIVE_SOLAR_HEATING in psychro_polygons.strategies:
            warn(
                f"{PassiveStrategy.PASSIVE_SOLAR_HEATING} assumes radiation from skylights only, using global horizontal radiation."
            )
            bal_t = (
                psychro_polygons.strategy_parameters.balance_temperature
                if PassiveStrategy.INTERNAL_HEAT_CAPTURE in psychro_polygons.strategies
                else None
            )
            dat, delta = poly_obj.evaluate_passive_solar(
                epw.global_horizontal_radiation,
                psychro_polygons.strategy_parameters.solar_heat_capacity,
                psychro_polygons.strategy_parameters.time_constant,
                bal_t,
            )
            sol_poly = poly_obj.passive_solar_polygon(delta, bal_t)
            if sol_poly is not None:
                name = PassiveStrategy.PASSIVE_SOLAR_HEATING.value
                poly = line_objs_to_vertices(sol_poly)
                dat = dat[0] if len(dat) == 1 else poly_obj.create_collection(dat, name)
                polygon_data.append(dat)
                polygon_names.append(name)
                ax.add_collection(
                    PatchCollection(
                        [Polygon(poly)],
                        fc="none",
                        ec="red",
                        zorder=8,
                        lw=1,
                        alpha=0.5,
                        ls="--",
                    )
                )
                xx, yy = polygon_centroid(*np.array(poly).T)
                ax.text(
                    xx,
                    yy,
                    "\n".join(textwrap.wrap(name, 12)),
                    fontsize="xx-small",
                    c="red",
                    ha="center",
                    va="center",
                    zorder=8,
                )
            else:
                strategy_warning(name)

        # compute total comfort values
        polygon_comfort = (
            [dat.average for dat in polygon_data]
            if isinstance(polygon_data[0], BaseCollection)
            else polygon_data
        )
        if isinstance(polygon_data[0], BaseCollection):
            merged_vals = merge_polygon_data(polygon_data)
            total_comf_data = poly_obj.create_collection(merged_vals, "Total Comfort")
            total_comfort = total_comf_data.average
        else:
            total_comf_data = 1 if sum(polygon_data) > 0 else 0
            total_comfort = total_comf_data
        polygon_names.insert(0, "Total Comfort")
        polygon_comfort.insert(0, total_comfort)

        # add total comfort to chart
        comfort_text = []
        for strat, val in list(zip(*[polygon_names, polygon_comfort])):
            comfort_text.append(f"{strat+':':<22} {val:>6.1%}")
        comfort_text = "\n".join(comfort_text)
        settings_text = "\n".join(
            [
                f'MRT: {"DBT" if psychro_polygons.mean_radiant_temperature is None else psychro_polygons.mean_radiant_temperature}',
                f" WS: {psychro_polygons.air_speed}m/s",
                f"CLO: {psychro_polygons.clo_value}",
                f"MET: {psychro_polygons.metabolic_rate}",
            ]
        )
        ax.text(
            0.3,
            0.98,
            "\n\n".join([settings_text, comfort_text]),
            transform=ax.transAxes,
            ha="left",
            va="top",
            zorder=8,
            fontsize="x-small",
            color="#555555",
            **{"fontname": "monospace"},
        )

    plt.tight_layout()

    return fig
