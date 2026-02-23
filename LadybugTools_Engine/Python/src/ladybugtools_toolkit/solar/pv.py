from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.model import Model
from honeybee.shade import Shade
from honeybee_energy.generator.loadcenter import ElectricLoadCenter
from honeybee_energy.generator.pv import PVProperties
from honeybee_energy.lib.constructions import opaque_construction_by_identifier
from honeybee_energy.result.err import Err
from honeybee_energy.run import run_idf
from honeybee_energy.simulation.parameter import SimulationParameter
from honeybee_energy.writer import energyplus_idf_version
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.energy import Energy
from ladybug.epw import EPW
from ladybug.futil import nukedir, write_to_file_by_name
from ladybug.sql import SQLiteResult
from ladybug_geometry.geometry3d import Face3D, Plane, Point3D
from matplotlib import pyplot as plt

from ..convert.to_pandas import to_pandas
from ..ladybug_geometry_extension.util import azimuth_altitude_to_vector
from ..bhom.logger import CONSOLE_LOGGER


def pv_yield(
    epw_file: Path,
    azimuth: float = 180,
    altitude: float = 0,
    context_shades: Optional[list[Shade]] = [],
    rated_efficiency: float = 0.15,
    active_area_fraction: float = 0.9,
    module_type: Optional[Any] = None,
    mounting_type: str = "FixedOpenRack",
    system_loss_fraction: float = 0.14,
    tracking_ground_coverage_ratio: float = 0.4,
    inverter_efficiency: float = 0.96,
    inverter_dc_to_ac_size_ratio: float = 1.1,
    output_directory: Optional[Path] = None,
    ylim: Union[list[float], None] = None,
) -> pd.DataFrame:
    """Estimate PV yield for a given azimuth and tilt, and PV configuration. This
    method does not include effects from overshading or self-shading.

    Args:
        epw_file: The path to the EPW file to use for the simulation. This file is used
            to determine the solar radiation and temperature conditions at the site.
        azimuth: A number between 0 and 360 for the azimuth of the plane of the
            photovoltaic array. The azimuth is the angle between the plane normal and
            true north, which is 0 degrees. The azimuth is positive in the clockwise
            direction from true north. (Default: 180 degrees, which is south).
        altitude: A number between 0 and 90 for the tilt of the plane of the photovoltaic
            array. The tilt is the angle between the plane normal and the horizontal
            plane. A tilt of 0 degrees is horizontal and a tilt of 90 degrees is
            vertical. (Default: 0 degrees).
        rated_efficiency: A number between 0 and 1 for the rated nameplate efficiency
            of the photovoltaic solar cells under standard test conditions (STC).
            Standard test conditions are 1,000 Watts per square meter solar
            irradiance, 25 degrees C cell temperature, and ASTM G173-03 standard
            spectrum. Nameplate efficiencies reported by manufacturers are typically
            under STC. Standard poly- or mono-crystalline silicon modules tend to have
            rated efficiencies in the range of 14-17%. Premium high efficiency
            mono-crystalline silicon modules with anti-reflective coatings can have
            efficiencies in the range of 18-20%. Thin film photovoltaic modules
            typically have efficiencies of 11% or less. (Default: 0.15 for standard
            silicon solar cells).
        active_area_fraction: The fraction of the parent Shade geometry that is
            covered in active solar cells. This fraction includes the difference
            between the PV panel (aka. PV module) area and the active cells within
            the panel as well as any losses for how the (typically rectangular) panels
            can be arranged on the Shade geometry. When the parent Shade geometry
            represents just the solar panels, this fraction is typically around 0.9
            given that the metal framing elements of the panel reduce the overall
            active area. (Default: 0.9, assuming parent Shade geometry represents
            only the PV panel geometry).
        module_type: Text to indicate the type of solar module. This is used to
            determine the temperature coefficients used in the simulation of the
            photovoltaic modules. Choose from the three options below. If None,
            the module_type will be inferred from the rated_efficiency of these
            PVProperties using the rated efficiencies listed below. (Default: None).

            * Standard - 12% <= rated_efficiency < 18%
            * Premium - rated_efficiency >= 18%
            * ThinFilm - rated_efficiency < 12%

        mounting_type: Text to indicate the type of mounting and/or tracking used
            for the photovoltaic array. Note that the OneAxis options have an axis
            of rotation that is determined by the azimuth of the parent Shade
            geometry. Also note that, in the case of one or two axis tracking,
            shadows on the (static) parent Shade geometry still reduce the
            electrical output, enabling the simulation to account for large
            context geometry casting shadows on the array. However, the effects
            of smaller detailed shading may be improperly accounted for and self
            shading of the dynamic panel geometry is only accounted for via the
            tracking_ground_coverage_ratio property on this object. Choose from
            the following. (Default: FixedOpenRack).
            * FixedOpenRack - ground or roof mounting where the air flows freely
            * FixedRoofMounted - mounting flush with the roof with limited air flow
            * OneAxis - a fixed tilt and azimuth, which define an axis of rotation
            * OneAxisBacktracking - same as OneAxis but with controls to reduce self-shade
            * TwoAxis - a dynamic tilt and azimuth that track the sun
        system_loss_fraction: A number between 0 and 1 for the fraction of the
            electricity output lost due to factors other than EPW climate conditions,
            panel efficiency/type, active area, mounting, and inverter conversion from
            DC to AC. Factors that should be accounted for in this input include
            soiling, snow, wiring losses, electrical connection losses, manufacturer
            defects/tolerances/mismatch in cell characteristics, losses from power
            grid availability, and losses due to age or light-induced degradation.
            Losses from these factors tend to be between 10-20% but can vary widely
            depending on the installation, maintenance and the grid to which the
            panels are connected. The loss_fraction_from_components staticmethod
            on this class can be used to estimate this value from the various
            factors that it is intended to account for. (Default: 0.14).
        tracking_ground_coverage_ratio: A number between 0 and 1 that only applies to
            arrays with one-axis tracking mounting_type. The ground coverage ratio (GCR)
            is the ratio of module surface area to the area of the ground beneath
            the array, which is used to account for self shading of single-axis panels
            as they move to track the sun. A GCR of 0.5 means that, when the modules
            are horizontal, half of the surface below the array is occupied by
            the array. An array with wider spacing between rows of modules has a
            lower GCR than one with narrower spacing. A GCR of 1 would be for an
            array with no space between modules, and a GCR of 0 for infinite spacing
            between rows. Typical values range from 0.3 to 0.6. (Default: 0.4).
        inverter_efficiency: A number between 0 and 1 for the load centers's
            inverter nominal rated DC-to-AC conversion efficiency. An inverter
            converts DC power, such as that output by photovoltaic panels, to
            AC power, such as that distributed by the electrical grid and is available
            from standard electrical outlets. Inverter efficiency is defined
            as the inverter's rated AC power output divided by its rated DC power
            output. (Default: 0.96).
        inverter_dc_to_ac_size_ratio: A positive number (typically greater than 1) for
            the ratio of the inverter's DC rated size to its AC rated size. Typically,
            inverters are not sized to convert the full DC output under standard
            test conditions (STC) as such conditions rarely occur in reality and
            therefore unnecessarily add to the size/cost of the inverter. For a
            system with a high DC to AC size ratio, during times when the
            DC power output exceeds the inverter's rated DC input size, the inverter
            limits the array's power output by increasing the DC operating voltage,
            which moves the arrays operating point down its current-voltage (I-V)
            curve. The default value of 1.1 is reasonable for most systems. A
            typical range is 1.1 to 1.25, although some large-scale systems have
            ratios of as high as 1.5. The optimal value depends on the system's
            location, array orientation, and module cost. (Default: 1.1).
        output_directory: The directory where the output CSV and images will be
            saved. If None, the output will be saved in the simulation directory.
        ylim: A list of two floats indicating the y-axis limits for the monthly
            PV yield bar chart. If None, the y-axis limits will be determined
            automatically. (Default: None).

    Returns:
        pd.DataFrame: A pandas DataFrame containing the PV yield data and EPW metrics.

    """
    OUTPUTS = [
        "Generator Produced DC Electricity Energy",
        "Generator PV Cell Temperature",
        "Plane of Array Irradiance",
        "Shaded Percent",
        "Inverter DC to AC Efficiency",
        "Inverter DC Input Electricity Energy",
        "Inverter AC Output Electricity Energy",
        "Inverter Conversion Loss Energy",
        "Inverter Conversion Loss Decrement Energy",
        "Inverter Thermal Loss Energy",
        "Inverter Ancillary AC Electricity Energy",
        "Electric Load Center Produced Electricity Energy",
        "Electric Load Center Produced Thermal Energy",
        "Facility Net Purchased Electricity Energy",
        "Facility Total Produced Electricity Energy",
    ]

    if altitude < 0:
        raise ValueError("Altitude must be greater than or equal to 0.")

    epw_file = Path(epw_file)
    epw = EPW(epw_file)

    normal = azimuth_altitude_to_vector(azimuth=azimuth, altitude=altitude)
    plane = Plane(n=normal, o=Point3D())
    face = Face3D.from_regular_polygon(
        side_count=4, radius=np.sqrt(2) / 2, base_plane=plane
    )
    shade = Shade.from_vertices(
        identifier="pv_panel_geometry", vertices=face.vertices, is_detached=True
    )

    # create PV properties and assign to shade
    pv_props = PVProperties(
        identifier="pv_panel",
        rated_efficiency=rated_efficiency,
        active_area_fraction=active_area_fraction,
        module_type=module_type,
        mounting_type=mounting_type,
        system_loss_fraction=system_loss_fraction,
        tracking_ground_coverage_ratio=tracking_ground_coverage_ratio,
    )
    module_type = pv_props.module_type
    shade.properties.energy.pv_properties = pv_props

    # create the model to simulate
    model = Model.from_objects("Generation_Loads", [shade])
    model.rooms_to_orphaned()

    # add ground
    soil_construction = opaque_construction_by_identifier("Mud")
    model.properties.energy.generate_ground_room(soil_construction)

    # add inverter efficiency and size
    energy_load_center = ElectricLoadCenter(
        inverter_efficiency=inverter_efficiency,
        inverter_dc_to_ac_size_ratio=inverter_dc_to_ac_size_ratio,
    )
    model.properties.energy.electric_load_center = energy_load_center  # type: ignore

    # process the simulation folder name and the directory
    directory: Path = Path(hb_folders.default_simulation_folder) / model.identifier
    sch_directory: Path = directory / "schedules"
    nukedir(directory.as_posix())

    # create simulation parameters for the coarsest/fastest E+ sim possible
    _sim_par_ = SimulationParameter()
    _sim_par_.timestep = 6
    _sim_par_.shadow_calculation.solar_distribution = "FullExteriorWithReflections"
    _sim_par_.output.reporting_frequency = "Hourly"
    for output in OUTPUTS:
        _sim_par_.output.add_output(output)
    _sim_par_.output.include_html = False
    _sim_par_.simulation_control.do_zone_sizing = False
    _sim_par_.simulation_control.do_system_sizing = False
    _sim_par_.simulation_control.do_plant_sizing = False

    # create the strings for simulation parameters and model
    ver_str = energyplus_idf_version()
    sim_par_str = _sim_par_.to_idf()
    model_str = model.to.idf(
        model,
        schedule_directory=sch_directory.as_posix(),
        patch_missing_adjacencies=True,
    )
    idf_str = "\n\n".join([ver_str, sim_par_str, model_str])

    # write the final string into an IDF
    idf = directory / "in.idf"
    write_to_file_by_name(directory.as_posix(), "in.idf", idf_str, True)

    CONSOLE_LOGGER.info(
        f"Calculating PV yield for {epw_file.name} ({altitude=}, {azimuth=})"
    )
    # run the IDF through EnergyPlus
    sql, _, _, _, err = run_idf(idf.as_posix(), epw_file.as_posix(), silent=True)
    if sql is None and err is not None:  # something went wrong; parse the errors
        err_obj = Err(err)
        print(err_obj.file_contents)
        for error in err_obj.fatal_errors:
            raise Exception(error)

    # parse the result sql and get the monthly data collections
    sql_obj = SQLiteResult(sql)
    collections = []
    for output in OUTPUTS:
        for col in sql_obj.data_collections_by_output_name(output):
            col: HourlyContinuousCollection
            if isinstance(col.header.data_type, Energy):
                col = col.to_unit("Wh").normalize_by_area(
                    area=shade.area, area_unit="m2"
                )
            col.header.metadata["time-zone"] = epw.location.time_zone
            collections.append(col)

    # add metadata to collections
    for col in collections:
        col.header.metadata["pv_rated_efficiency"] = rated_efficiency
        col.header.metadata["pv_active_area_fraction"] = active_area_fraction
        col.header.metadata["pv_module_type"] = module_type
        col.header.metadata["pv_mounting_type"] = mounting_type
        col.header.metadata["pv_system_loss_fraction"] = system_loss_fraction
        col.header.metadata["pv_tracking_ground_coverage_ratio"] = (
            tracking_ground_coverage_ratio
        )
        col.header.metadata["pv_inverter_efficiency"] = inverter_efficiency
        col.header.metadata["pv_inverter_dc_to_ac_size_ratio"] = (
            inverter_dc_to_ac_size_ratio
        )
        col.header.metadata["pv_azimuth"] = azimuth
        col.header.metadata["pv_altitude"] = altitude

    # add epw metrics
    collections.extend(
        [
            epw.dry_bulb_temperature,
            epw.global_horizontal_radiation,
            epw.direct_normal_radiation,
            epw.diffuse_horizontal_radiation,
        ]
    )

    # convert collections to pandas DataFrame
    df = pd.concat([to_pandas(i) for i in collections], axis=1).sort_index(axis=1)

    if output_directory is None:
        output_directory = directory
    else:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_directory / "pv_data.csv", index=True, header=True)

    # create some summary outputs to make reporting this nice and easy
    image_dir = output_directory / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    FIGSIZE = (12, 5)
    TITLE_STR = (
        f"{epw_file.name} - Azimuth: {azimuth}$\degree$ - Altitude: {altitude}$\degree$"  # type: ignore
    )

    # Annual hourly PV yield (normalised by area, with indication of panel orientation)
    total_ac_produced = df.filter(
        regex="Facility Total Produced Electricity Energy Intensity"
    ).squeeze()  # Wh/m2
    total_dc_produced = df.filter(
        regex="Generator Produced DC Electricity Energy Intensity"
    ).squeeze()  # Wh/m2
    insolation = df.filter(regex="Plane of Array Irradiance").squeeze()  # W/m2

    pv_description = (
        f"Total AC: {total_ac_produced.sum() / 1000:.1f}kWh/m2\n"
        f"Total DC: {total_dc_produced.sum() / 1000:.1f}kWh/m2\n"
        f"PV azimuth: {azimuth}$\degree$\n"  # type: ignore
        f"PV altitude: {altitude}$\degree$\n"  # type: ignore
        f"PV efficiency: {rated_efficiency:.1%}\n"
        f"PV active area: {active_area_fraction:.1%}\n"
        f"PV module type: {module_type}\n"
        f"PV mounting type: {mounting_type}\n"
        f"PV system loss: {system_loss_fraction:0.1%}\n"
        f"Inverter Efficiency: {inverter_efficiency:.1%}\n"
    )

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.plot(insolation, label="Insolation", c="#EE7837", zorder=2)
    ax.plot(total_ac_produced, label="AC", c="#6D104E", zorder=4)
    ax.plot(total_dc_produced, label="DC", c="#006DA8", zorder=3)
    ax.text(
        0.01,
        1,
        pv_description,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize="xx-small",
    )
    ax.set_title(TITLE_STR)
    ax.set_xlim(df.index.min(), df.index.max())
    ax.set_ylim(0, None)
    ax.set_ylabel(f"{total_dc_produced.name[0]} ({total_dc_produced.name[1]})")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(
        image_dir / "pv_yield.png", dpi=300, bbox_inches="tight", transparent=True
    )
    plt.close(fig)

    # yield, monthly totals (kWh/m2)
    monthly = (
        pd.concat([total_dc_produced, total_ac_produced], axis=1).resample("MS").sum()
        / 1000
    )  # kWh/m2
    monthly.columns = ["DC", "AC"]
    monthly.index = [i.strftime("%b") for i in monthly.index]
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    for col, color in zip(monthly.columns, ["#006DA8", "#6D104E"]):
        monthly[col].plot(
            kind="bar",
            ax=ax,
            fc=color,
            width=0.85,
            label=col,
            zorder=2,
        )
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ylim = ax.get_ylim()
    ax.text(
        0.01,
        1,
        pv_description,
        ha="left",
        transform=ax.transAxes,
        va="top",
        fontsize="xx-small",
    )
    ax.set_title(TITLE_STR)
    # ax.set_xlim(df.index.min(), df.index.max())
    ax.set_ylim(0, None)
    ax.set_ylabel(f"{total_dc_produced.name[0]} (kWh/m2)")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(
        image_dir / "pv_yield_monthly.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    # PV temperature and dbt
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    dbt_series = df.filter(regex="Dry Bulb Temperature").squeeze()
    pv_temperature_series = df.filter(regex="PV Cell Temperature").squeeze()

    ax.plot(pv_temperature_series, label="PV Cell Temperature", c="#EB671C", zorder=2)
    ax.plot(
        dbt_series,
        label="Dry Bulb Temperature",
        c="#BC204B",
    )
    # add text description
    ax.text(
        0.01,
        1,
        pv_description,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize="xx-small",
    )
    ax.set_title(TITLE_STR)
    ax.set_xlim(df.index.min(), df.index.max())
    ax.set_ylabel(f"{pv_temperature_series.name[0]} ({pv_temperature_series.name[1]})")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(
        image_dir / "pv_temperature.png", dpi=300, bbox_inches="tight", transparent=True
    )
    plt.close(fig)

    return df
