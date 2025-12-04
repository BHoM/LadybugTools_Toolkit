"""Module for the external comfort package, handling simulation of shaded/
unshaded surface temperatures in an abstract "openfield" condition."""
# pylint: disable=E0401
import json
import os
from pathlib import Path
from dataclasses import dataclass
import subprocess
from typing import Any, Optional
import re

# pylint: enable=E0401
from caseconverter import pascalcase
import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.model import Model
from honeybee_energy.config import folders as energy_folders
from honeybee_energy.dictutil import dict_to_material
from honeybee_energy.material.opaque import EnergyMaterial, EnergyMaterialVegetation
from honeybee_energy.measure import Measure
from honeybee_energy.run import (
    _parse_os_cli_failure,
    output_energyplus_files,
    run_idf,
    run_osw,
    to_openstudio_sim_folder,
)
from honeybee_energy.result.err import Err
from honeybee_energy.simulation.parameter import (
    ShadowCalculation,
    SimulationControl,
    SimulationOutput,
    SimulationParameter,
)
from honeybee_openstudio.openstudio import OSModel
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug.futil import copy_file_tree, nukedir, preparedir
from ladybug.stat import STAT
from ladybug_comfort.collection.solarcal import OutdoorSolarCal, SolarCalParameter
from lbt_recipes.version import check_openstudio_version

from ..bhom.logger import CONSOLE_LOGGER
from ..bhom.to_bhom import (
    hourlycontinuouscollection_to_bhom,
    material_to_bhom,
)
from ..honeybee_extension.results import load_sql
from ..ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)
from ..ladybug_extension.epw import epw_to_dataframe
from ..ladybug_extension.epw import equality as epw_equality
from ..ladybug_extension.groundtemperature import energyplus_strings
from .model import create_model, get_ground_reflectance, model_equality
from ..helpers import convert_keys_to_snake_case, sanitise_string
from .material import Materials

def hb_model_to_osm(
    _model: Model,
    _epw_file: str,
    _sim_par_: Optional[SimulationParameter] = None,
    measures_: Optional[list[Measure]] = None,
    add_str_: Optional[list[str]] = None,
    _folder_: Optional[str] = None,
    use_ironpython: bool = False,
    run_: int = 0,
) -> tuple[list[str], str, str, str, str, str, str, str]:
    """
    Translate a Honeybee Model to an OpenStudio Model and EnergyPlus IDF and
    optionally run the IDF in EnergyPlus.

    Args:
        _model: A honeybee model object possessing all geometry and corresponding
            energy simulation properties.
        _epw_file: Path to an .epw file on this computer as a text string.
        _sim_par_: A honeybee Energy SimulationParameter object that describes all
            of the setting for the simulation. If None, some default simulation
            parameters will automatically be used.
        measures_: An optional list of measures to apply to the OpenStudio model
            upon export. Use the "HB Load Measure" component to load a measure
            into Grasshopper and assign input arguments. Measures can be
            downloaded from the NREL Building Components Library (BCL) at
            (https://bcl.nrel.gov/).
        add_str_: THIS OPTION IS JUST FOR ADVANCED USERS OF ENERGYPLUS.
            You can input additional text strings here that you would like
            written into the IDF.  The input here should be complete EnergyPlus
            objects as a single string following the IDF format. This input can
            be used to write objects into the IDF that are not currently supported
            by Honeybee.
        _folder_: An optional folder on this computer, into which the IDF and result
            files will be written.
        run_: Set to "True" to translate the Honeybee jsons to an OpenStudio Model
            (.osm) and EnergyPlus Input Data File (.idf) and then simulate the
            .idf in EnergyPlus. This will ensure that all result files appear
            in their respective outputs from this component.
            _
            This input can also be the integer "2", which will run the whole translation
            and simulation silently (without any batch windows).

    Returns:
        A tuple with the following items:
        - jsons
        - osm
        - osw
        - idf
        - sql
        - zsz
        - rdd
        - html

    Source:
        - https://github.com/ladybug-tools/honeybee-grasshopper-energy/blob/master/honeybee_grasshopper_energy/src/HB%20Model%20to%20OSM.py
    """

    ROOM_COUNT_THRESH = 1000  # threshold at which the CLI is used for translation

    def measures_to_folder(measures: list[Measure], sim_folder: str):
        osw_dict = {}  # dictionary that will be turned into the OSW JSON
        osw_dict['steps'] = []
        mea_folder = os.path.join(sim_folder, 'measures')
        # ensure measures are correctly ordered
        m_dict = {'ModelMeasure': [], 'EnergyPlusMeasure': [], 'ReportingMeasure': []}
        for measure in measures:
            assert isinstance(measure, Measure), 'Expected honeybee-energy Measure. ' \
                'Got {}.'.format(type(measure))
            m_dict[measure.type].append(measure)  # type: ignore
        sorted_measures = m_dict['ModelMeasure'] + m_dict['EnergyPlusMeasure'] + \
            m_dict['ReportingMeasure']
        # add the measures and the measure paths to the OSW
        for measure in sorted_measures:
            measure.validate()  # ensure that all required arguments have values
            osw_dict['steps'].append(measure.to_osw_dict())  # add measure to workflow
            dest_folder = os.path.join(mea_folder, os.path.basename(measure.folder))
            copy_file_tree(measure.folder, dest_folder)
            test_dir = os.path.join(dest_folder, 'tests')
            if os.path.isdir(test_dir):
                nukedir(test_dir, rmdir=True)
        # write the dictionary to a workflow.osw
        osw_json = os.path.join(mea_folder, 'workflow.osw')
        try:
            with open(osw_json, 'w') as fp:
                json.dump(osw_dict, fp, indent=4)
        except UnicodeDecodeError:  # non-unicode character in the dictionary
            with open(osw_json, 'w') as fp:
                json.dump(osw_dict, fp, indent=4, ensure_ascii=False)
        return mea_folder

    # check the presence of openstudio and check that the version is compatible
    check_openstudio_version()
    assert isinstance(_model, Model), \
        'Expected Honeybee Model for _model input. Got {}.'.format(type(_model))

    # process the simulation parameters
    if _sim_par_ is None:
        sim_par = SimulationParameter()
        sim_par.output.add_zone_energy_use()
        sim_par.output.add_hvac_energy_use()
        sim_par.output.add_electricity_generation()
    else:
        sim_par = _sim_par_.duplicate()  # ensure input is not edited
    
    if measures_ is None:
        measures_ = []

    # assign design days from the DDY next to the EPW if there are None
    folder, epw_file_name = os.path.split(_epw_file)
    if len(sim_par.sizing_parameter.design_days) == 0:
        msg = None
        ddy_file = os.path.join(folder, epw_file_name.replace(".epw", ".ddy"))
        if os.path.isfile(ddy_file):
            try:
                sim_par.sizing_parameter.add_from_ddy_996_004(ddy_file)
            except AssertionError:
                pass
            if len(sim_par.sizing_parameter.design_days) == 0:
                msg = (
                    "No ddy_file_ was input into the _sim_par_ sizing "
                    "parameters\n and no design days were found in the .ddy file "
                    "next to the _epw_file."
                )
        else:
            msg = (
                "No ddy_file_ was input into the _sim_par_ sizing parameters\n"
                "and no .ddy file was found next to the _epw_file."
            )
        if msg is not None:
            epw_obj = EPW(_epw_file)
            des_days = [
                epw_obj.approximate_design_day("WinterDesignDay"),
                epw_obj.approximate_design_day("SummerDesignDay"),
            ]
            sim_par.sizing_parameter.design_days = des_days
            msg = (
                msg + "\nDesign days were generated from the input _epw_file but this "
                "\nis not as accurate as design days from DDYs distributed with the EPW."
            )
            CONSOLE_LOGGER.warning(msg)
        
    if sim_par.sizing_parameter.climate_zone is None:
        stat_file = os.path.join(folder, epw_file_name.replace('.epw', '.stat'))
        if os.path.isfile(stat_file):
            stat_obj = STAT(stat_file)
            sim_par.sizing_parameter.climate_zone = stat_obj.ashrae_climate_zone

    # process the simulation folder name and the directory
    _folder_ = hb_folders.default_simulation_folder if _folder_ is None else _folder_
    clean_name = re.sub(r"[^.A-Za-z0-9_-]", "_", _model.display_name)
    directory = os.path.join(_folder_, clean_name, "openstudio")

    # delete any existing files in the directory and prepare it for simulation
    nukedir(directory, True)
    preparedir(directory)
    sch_directory = os.path.join(directory, "schedules")
    preparedir(sch_directory)

    # write the model and simulation parameter to JSONs
    model_json = os.path.join(directory, "{}.hbjson".format(clean_name))
    with open(model_json, "wb") as fp:
        model_str = json.dumps(_model.to_dict(), ensure_ascii=False)
        fp.write(model_str.encode("utf-8"))
    sim_par_json = os.path.join(directory, "simulation_parameter.json")
    with open(sim_par_json, "w") as fp:
        json.dump(sim_par.to_dict(), fp)
    jsons = [model_json, sim_par_json]

    # determine whether to run the translation with cPython or IronPython
    if OSModel is not None:
        vent_sim_control = _model.properties.energy.ventilation_simulation_control
        if vent_sim_control.vent_control_type == "SingleZone":
            if len(_model.rooms) < ROOM_COUNT_THRESH:
                osc_version = tuple(
                    int(v) for v in OSModel().version().str().split(".")
                )
                if osc_version == energy_folders.openstudio_version:
                    use_ironpython = True
    
    if use_ironpython:  # translate the model using IronPython methods
        CONSOLE_LOGGER.debug(
            f"Translating {_model} to OpenStudio Model using Python - {directory}."
        )
        add_str = (
            "\n".join(add_str_)
            if len(add_str_) != 0 and add_str_[0] is not None
            else None
        )
        osm, osw, idf = to_openstudio_sim_folder(
            _model,
            directory,
            epw_file=_epw_file,
            sim_par=sim_par,
            schedule_directory=sch_directory,
            enforce_rooms=True,
            additional_measures=measures_,
            strings_to_inject=add_str,
        )
        if run_ > 0:
            silent = True if run_ > 1 else False
            if idf is not None:  # run the IDF directly through E+
                sql, zsz, rdd, html, err = run_idf(idf, _epw_file, silent=silent)
            else:
                osm, idf = run_osw(osw, measures_only=False, silent=silent)
                if idf is None or not os.path.isfile(idf):
                    _parse_os_cli_failure(directory)
                sql, zsz, rdd, html, err = output_energyplus_files(os.path.dirname(idf))
        else:
            sql = None
            zsz = None
            rdd = None
            html = None
    else:  # translate the model with cPython using OpenStudio CLI
        CONSOLE_LOGGER.debug(
            f"Translating {_model} to OpenStudio Model using OpenStudio CLI - {directory}."
        )
        # write additional strings and measures to a folder
        add_idf = None
        if len(add_str_) != 0 and add_str_[0] is not None:
            add_str = "\n".join(add_str_)
            add_idf = os.path.join(directory, "additional_strings.idf")
            with open(add_idf, "w") as fp:
                fp.write(add_str)
        measure_folder = None
        if len(measures_) != 0 and measures_[0] is not None:
            measure_folder = measures_to_folder(measures_, directory)

        # put together the arguments for the command to be run
        if run_ > 0:  # use the simulate command
            cmds = [
                '"{}"'.format(folders.python_exe_path),
                "-m",
                "honeybee_energy",
                "simulate",
                "model",
                '"{}"'.format(model_json),
                '"{}"'.format(_epw_file),
                "--sim-par-json",
                '"{}"'.format(sim_par_json),
                "--folder",
                '"{}"'.format(directory),
            ]
        else:  # use the translate command
            cmds = [
                '"{}"'.format(folders.python_exe_path),
                "-m",
                "honeybee_energy",
                "translate",
                "model-to-sim-folder",
                '"{}"'.format(model_json),
                '"{}"'.format(_epw_file),
                "--sim-par-json",
                '"{}"'.format(sim_par_json),
                "--folder",
                '"{}"'.format(directory),
            ]
        if add_idf is not None:
            cmds.append("--additional-idf")
            cmds.append('"{}"'.format(add_idf))
        if measure_folder is not None:
            cmds.append("--measures")
            cmds.append('"{}"'.format(measure_folder))
        osm = os.path.join(directory, "in.osm")
        idf = os.path.join(directory, "run", "in.idf")

        # execute the command
        custom_env = os.environ.copy()
        custom_env["PYTHONHOME"] = ""
        cmds = " ".join(cmds)
        if os.name == "nt":
            shell = False if run_ == 1 else True
        else:
            shell = True
        process = subprocess.Popen(cmds, shell=shell, env=custom_env)
        result = process.communicate()  # freeze the canvas while running

        # check if any part of the translation failed
        osw = os.path.join(directory, "workflow.osw")
        osw = osw if os.path.isfile(osw) else None
        if not os.path.isfile(osm):
            # get the error from stdout
            process = subprocess.Popen(
                cmds, shell=shell, env=custom_env, stderr=subprocess.PIPE
            )
            result = process.communicate()  # freeze the canvas while running
            raise ValueError(
                "Failed to translate Model to OpenStudio.\n{}".format(
                    "\n".join(str(result[1]).split("\n")[-3:])
                )
            )
        
        sql, zsz, rdd, html, err = None, None, None, None, None
        if run_ > 0:
            if not os.path.isfile(idf):
                cmds = " ".join(cmds) if os.name == "nt" else cmds
                raise ValueError("Failed to translate Model to EnergyPlus.")
            sql, zsz, rdd, html, err = output_energyplus_files(os.path.dirname(idf))
        
    # parse the error log and report any warnings
    if run_ >= 1 and err is not None:
        err_obj = Err(err)
        for warn in err_obj.severe_errors:
            CONSOLE_LOGGER.warning(warn)
        for error in err_obj.fatal_errors:
            raise Exception(error)
    
    return (
        jsons,
        osm,
        osw,
        idf,
        sql,
        zsz,
        rdd,
        html,
    )

def simulation_id(
    epw_file: Path,
    ground_material: EnergyMaterial | EnergyMaterialVegetation,
    shade_material: EnergyMaterial | EnergyMaterialVegetation,
) -> str:
    """Create an ID for a simulation.

    Args:
        epw_file (Path): The path to the EPW file.
        ground_material (EnergyMaterial | EnergyMaterialVegetation): The ground material.
        shade_material (EnergyMaterial | EnergyMaterialVegetation): The shade material.

    Returns:
        str: The simulation ID.
    """

    epw_id = sanitise_string(epw_file.stem)
    ground_material_id = sanitise_string(ground_material.identifier)
    shade_material_id = sanitise_string(shade_material.identifier)
    return f"{epw_id}__{ground_material_id}__{shade_material_id}"


def simulation_directory(model: Model) -> Path:
    """Get the working directory (where simulation results will be stored) for the given model, and
        create it if it doesn't already exist.

    Args:
        model (Model): A honeybee Model.

    Returns:
        Path: The simulation directory associated with the given model.
    """

    working_dir: Path = Path(hb_folders.default_simulation_folder) / model.identifier
    working_dir.mkdir(parents=True, exist_ok=True)

    return working_dir


def simulate_surface_temperatures(
    model: Model, epw_file: Path, remove_dir: bool = False
) -> dict[str, HourlyContinuousCollection]:
    """Simulate surface temperatures for a Honeybee Model and return the
        resulting SQL results file path.

    Args:
        model (Model): A honeybee Model.
        epw_file (Path): The path to an EPW file.
        remove_dir (bool, optional): Set to True to remove the simulation
            directory

    Returns:
        dict[str, HourlyContinuousCollection]: Surface temperature results.
    """
    
    if not isinstance(model, Model):
        raise ValueError("model must be a Honeybee Model.")

    epw_file = Path(epw_file)
    if not epw_file.exists():
        raise ValueError("epw_file must be a valid file path.")
    epw = EPW(epw_file)

    sim_dir = simulation_directory(model)

    # does the epw file already exist in the sim dir
    epws_match = False
    saved_epw = (sim_dir / epw_file.name)
    if saved_epw.exists():
        if epw_equality(epw, EPW(saved_epw), include_header=True):
            epws_match = True
        else:
            saved_epw.unlink()
    epw.save(saved_epw.as_posix())
    # create ddy file from epw
    ddy_file = sim_dir / saved_epw.with_suffix(".ddy").name
    epw.to_ddy(file_path=ddy_file.as_posix())

    # do the models match
    models_match = False
    saved_model = sim_dir / f"{model.identifier}.hbjson"
    if saved_model.exists():
        if model_equality(model, Model.from_hbjson(saved_model.as_posix())):
            models_match = True
        else:
            saved_model.unlink()
    model.to_hbjson(folder=saved_model.parent.as_posix(), name=saved_model.name)

    sql_path = sim_dir / model.identifier / "openstudio" / "run" / "eplusout.sql"
    print(sql_path)

    # check for existing results and reload if they exist
    matchy_matchy = (
        sql_path.exists(),
        models_match,
        epws_match,
    )
    if not all(matchy_matchy):
        CONSOLE_LOGGER.info(f"Simulating {model.identifier}")

        sim_par = SimulationParameter(
            output=SimulationOutput(
                outputs=["Surface Outside Face Temperature"],
                include_sqlite=True,
                summary_reports=None,
                include_html=False,
            ),
            simulation_control=SimulationControl(
                do_zone_sizing=False,
                do_system_sizing=False,
                do_plant_sizing=False,
                run_for_sizing_periods=False,
                run_for_run_periods=True,
            ),
            shadow_calculation=ShadowCalculation(
                solar_distribution="FullExteriorWithReflections",
                calculation_method="PolygonClipping",
                calculation_update_method="Periodic",
                maximum_figures=200,
            ),
            terrain_type="Country",
            timestep=10,
        )

        (jsons,
        osm,
        osw,
        idf,
        sql_path,
        zsz,
        rdd,
        html,) = hb_model_to_osm(
            _model=model,
            _epw_file=epw_file.as_posix(),
            _sim_par_=sim_par,
            add_str_=energyplus_strings(epw).split("\n"),
            _folder_=sim_dir.as_posix(),
            use_ironpython=True, 
            run_=1,
        )

    else:
        CONSOLE_LOGGER.info(f"Reloading {model.identifier}")

    df = load_sql(Path(sql_path))

    if remove_dir:
        nukedir(sim_dir, rmdir=True)

    return {
        "shaded_down_temperature": collection_from_series(
            df.filter(regex="GROUND_ZONE_UP_SHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
            .rename("Ground Temperature (C)")
        ),
        "unshaded_down_temperature": collection_from_series(
            df.filter(regex="GROUND_ZONE_UP_UNSHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
            .rename("Ground Temperature (C)")
        ),
        "shaded_up_temperature": collection_from_series(
            df.filter(regex="SHADE_ZONE_DOWN")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
            .rename("Sky Temperature (C)")
        ),
        "unshaded_up_temperature": epw.sky_temperature,
    }


def radiant_temperature(
    collections: list[HourlyContinuousCollection], view_factors: list[float] = None
) -> HourlyContinuousCollection:
    """Calculate the MRT from a list of surface temperature collections, and view
        factors to each of those surfaces.

    Args:
        collections (List[HourlyContinuousCollection]):
            A list of hourly continuous collections.
        view_factors (List[float]):
            A list of view factors to each of the collections.
            If None, then all input collections are weighted equally.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of the effective radiant temperature.
    """

    if view_factors is None:
        view_factors = [1 / len(collections)] * len(collections)
    if len(collections) != len(view_factors):
        raise ValueError("The number of collections and view factors must be the same.")
    if sum(view_factors) != 1:
        raise ValueError("The sum of view factors must be 1.")

    mrt_series = (
        np.power(
            (
                np.power(
                    pd.concat([collection_to_series(i) for i in collections], axis=1)
                    + 273.15,
                    4,
                )
                * view_factors
            ).sum(axis=1),
            0.25,
        )
        - 273.15
    )
    mrt_series.name = "Radiant Temperature (C)"
    return collection_from_series(mrt_series)


_ATTRIBUTES = [
    "shaded_down_temperature",
    "shaded_up_temperature",
    "unshaded_down_temperature",
    "unshaded_up_temperature",
    "shaded_radiant_temperature",
    "shaded_longwave_mean_radiant_temperature_delta",
    "shaded_shortwave_mean_radiant_temperature_delta",
    "shaded_mean_radiant_temperature",
    "unshaded_radiant_temperature",
    "unshaded_longwave_mean_radiant_temperature_delta",
    "unshaded_shortwave_mean_radiant_temperature_delta",
    "unshaded_mean_radiant_temperature",
]


@dataclass(init=True, repr=True, eq=True)
class SimulationResult:
    """_"""

    epw_file: Path
    ground_material: EnergyMaterial | EnergyMaterialVegetation
    shade_material: EnergyMaterial | EnergyMaterialVegetation
    identifier: str = None

    shaded_down_temperature: HourlyContinuousCollection = None
    shaded_up_temperature: HourlyContinuousCollection = None

    unshaded_down_temperature: HourlyContinuousCollection = None
    unshaded_up_temperature: HourlyContinuousCollection = None

    shaded_radiant_temperature: HourlyContinuousCollection = None
    shaded_longwave_mean_radiant_temperature_delta: HourlyContinuousCollection = None
    shaded_shortwave_mean_radiant_temperature_delta: HourlyContinuousCollection = None
    shaded_mean_radiant_temperature: HourlyContinuousCollection = None

    unshaded_radiant_temperature: HourlyContinuousCollection = None
    unshaded_longwave_mean_radiant_temperature_delta: HourlyContinuousCollection = None
    unshaded_shortwave_mean_radiant_temperature_delta: HourlyContinuousCollection = None
    unshaded_mean_radiant_temperature: HourlyContinuousCollection = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.identifier})"

    def __post_init__(self):
        """_"""

        # validation
        if not isinstance(self.epw_file, (Path, str)):
            raise ValueError("epw_file must be a Path or str.")
        self.epw_file = Path(self.epw_file).absolute()
        if not self.epw_file.exists():
            raise ValueError("epw_file does not exist.")

        if isinstance(self.ground_material, Materials):
            self.ground_material = self.ground_material.value
        if isinstance(self.shade_material, Materials):
            self.shade_material = self.shade_material.value

        if not isinstance(
            self.ground_material, (EnergyMaterial, EnergyMaterialVegetation)
        ):
            raise ValueError(
                "ground_material must be an EnergyMaterial or EnergyMaterialVegetation."
            )
        if not isinstance(
            self.shade_material, (EnergyMaterial, EnergyMaterialVegetation)
        ):
            raise ValueError(
                "shade_material must be an EnergyMaterial or EnergyMaterialVegetation."
            )

        if self.identifier is None:
            self.identifier = simulation_id(
                self.epw_file, self.ground_material, self.shade_material
            )

        for attr in _ATTRIBUTES:
            if not isinstance(
                getattr(self, attr), (HourlyContinuousCollection, type(None))
            ):
                raise ValueError(
                    f"{attr} must be either an HourlyContinuousCollection, or None."
                )

        # run simulation and populate object with results if not already done
        _epw = EPW(self.epw_file)
        _model = create_model(
            identifier=self.identifier,
            ground_material=self.ground_material,
            shade_material=self.shade_material,
        )

        if not all(
            [
                self.shaded_down_temperature,
                self.unshaded_down_temperature,
                self.shaded_up_temperature,
                self.unshaded_up_temperature,
            ]
        ):
            results = simulate_surface_temperatures(
                model=_model,
                epw_file=self.epw_file,
                remove_dir=not bool(self.identifier),
            )
            for k, v in results.items():
                if isinstance(getattr(self, k), HourlyContinuousCollection):
                    continue
                setattr(self, k, v)

        # calculate other variables
        self.shaded_radiant_temperature = radiant_temperature(
            [
                self.shaded_down_temperature,
                self.shaded_up_temperature,
            ],
        )
        self.unshaded_radiant_temperature = radiant_temperature(
            [
                self.unshaded_down_temperature,
                self.unshaded_up_temperature,
            ],
        )

        # calculate MRT
        params = SolarCalParameter()
        shaded_cal = OutdoorSolarCal(
            location=_epw.location,
            direct_normal_solar=_epw.direct_normal_radiation,
            diffuse_horizontal_solar=_epw.diffuse_horizontal_radiation,
            horizontal_infrared=_epw.horizontal_infrared_radiation_intensity,
            surface_temperatures=self.shaded_radiant_temperature,
            floor_reflectance=get_ground_reflectance(_model),
            sky_exposure=0,
            fraction_body_exposed=0,
            solarcal_body_parameter=params,
        )
        unshaded_cal = OutdoorSolarCal(
            location=_epw.location,
            direct_normal_solar=_epw.direct_normal_radiation,
            diffuse_horizontal_solar=_epw.diffuse_horizontal_radiation,
            horizontal_infrared=_epw.horizontal_infrared_radiation_intensity,
            surface_temperatures=self.unshaded_down_temperature,
            floor_reflectance=get_ground_reflectance(_model),
            sky_exposure=1,
            fraction_body_exposed=1,
            solarcal_body_parameter=params,
        )
        for shadedness, cal in list(
            zip(*[["shaded", "unshaded"], [shaded_cal, unshaded_cal]])
        ):
            for var in [
                "mean_radiant_temperature",
                "shortwave_mrt_delta",
                "longwave_mrt_delta",
            ]:
                setattr(
                    self,
                    f"{shadedness}_{var.replace('mrt', 'mean_radiant_temperature')}",
                    getattr(cal, var),
                )

        # add some accessors for collections as series
        for attr in _ATTRIBUTES:
            setattr(self, f"{attr}_series", collection_to_series(getattr(self, attr)))

    def to_dict(self) -> dict[str, Any]:
        """Convert this object to a dictionary."""
        ground_material_dict = self.ground_material.to_dict()
        shade_material_dict = self.shade_material.to_dict()

        attr_dict = {}
        for attr in _ATTRIBUTES:
            if getattr(self, attr):
                attr_dict[attr] = getattr(self, attr).to_dict()

        d = {
            **{
                "type": "SimulationResult",
                "epw_file": self.epw_file.as_posix(),
                "ground_material": ground_material_dict,
                "shade_material": shade_material_dict,
                "identifier": self.identifier,
            },
            **attr_dict,
        }

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimulationResult":
        """Create this object from a dictionary."""
        if isinstance(d["ground_material"], dict):
            d["ground_material"] = dict_to_material(d["ground_material"])

        if isinstance(d["shade_material"], dict):
            d["shade_material"] = dict_to_material(d["shade_material"])

        for attr in _ATTRIBUTES:
            if d.get(attr, None):
                if isinstance(d[attr], dict):
                    d[attr] = HourlyContinuousCollection.from_dict(d[attr])
            else:
                d[attr] = None

        return cls(
            epw_file=d["epw_file"],
            ground_material=d["ground_material"],
            shade_material=d["shade_material"],
            identifier=d["identifier"],
            shaded_down_temperature=d["shaded_down_temperature"],
            shaded_up_temperature=d["shaded_up_temperature"],
            unshaded_down_temperature=d["unshaded_down_temperature"],
            unshaded_up_temperature=d["unshaded_up_temperature"],
            shaded_radiant_temperature=d["shaded_radiant_temperature"],
            shaded_longwave_mean_radiant_temperature_delta=d[
                "shaded_longwave_mean_radiant_temperature_delta"
            ],
            shaded_shortwave_mean_radiant_temperature_delta=d[
                "shaded_shortwave_mean_radiant_temperature_delta"
            ],
            shaded_mean_radiant_temperature=d["shaded_mean_radiant_temperature"],
            unshaded_radiant_temperature=d["unshaded_radiant_temperature"],
            unshaded_longwave_mean_radiant_temperature_delta=d[
                "unshaded_longwave_mean_radiant_temperature_delta"
            ],
            unshaded_shortwave_mean_radiant_temperature_delta=d[
                "unshaded_shortwave_mean_radiant_temperature_delta"
            ],
            unshaded_mean_radiant_temperature=d["unshaded_mean_radiant_temperature"],
        )

    def to_json(self) -> str:
        """Create a JSON string from this object."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_string: str) -> "SimulationResult":
        """Create this object from a JSON string."""

        return cls.from_dict(json.loads(json_string))

    def to_file(self, path: Path) -> Path:
        """Write this object to a JSON file."""

        if Path(path).suffix != ".json":
            raise ValueError("path must be a JSON file.")

        with open(Path(path), "w") as fp:
            fp.write(self.to_json())

        return Path(path)

    @classmethod
    def from_file(cls, path: Path) -> "SimulationResult":
        """Create this object from a JSON file."""

        with open(Path(path), "r") as fp:
            return cls.from_json(fp.read())

    @property
    def epw(self) -> EPW:
        """Return the EPW object associated with this simulation result."""
        return EPW(self.epw_file)

    @property
    def simulation_directory(self) -> Path:
        """Return the simulation directory for this simulation result."""
        return simulation_directory(self.model)

    @property
    def model(self) -> Model:
        """Return the model object for this simulation result."""
        return create_model(
            identifier=self.identifier,
            ground_material=self.ground_material,
            shade_material=self.shade_material,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Create a Pandas DataFrame from this object.

        Returns:
            pd.DataFrame: Represent this object as a Pandas DataFrame.
        """

        obj_series = []
        for var in dir(self):
            for shadedness in ["shaded", "unshaded"]:
                if not var.startswith(shadedness):
                    continue
                _temp = getattr(self, var)
                if isinstance(_temp, HourlyContinuousCollection):
                    _temp = collection_to_series(_temp)
                    _temp.rename(
                        (shadedness.title(), _temp.name),
                        inplace=True,
                    )
                    obj_series.append(_temp)

        obj_df = pd.concat(obj_series, axis=1)

        return pd.concat(
            [
                pd.concat(
                    [epw_to_dataframe(self.epw, include_additional=True)],
                    axis=1,
                    keys=["EPW"],
                ),
                obj_df,
            ],
            axis=1,
        )

    def description(self, include_shade_material: bool = True) -> str:
        """Create the description for this object.

        Args:
            include_shade_material (bool, optional):
                Set to False to exclude the shade material from the description.
                Defaults to True.

        Returns:
            str:
                A description of this object.
        """
        if include_shade_material:
            return (
                f"{self.epw_file.name} - "
                f"{self.ground_material.identifier} (ground material) - "
                f"{self.shade_material.identifier} (shade material)"
            )

        return (
            f"{self.epw_file.name} - "
            f"{self.ground_material.identifier} (ground material)"
        )
