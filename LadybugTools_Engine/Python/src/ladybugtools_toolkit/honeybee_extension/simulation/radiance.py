"""_"""
# pylint: disable=E0401
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.model import Model
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_radiance.writer import _filter_by_pattern
from honeybee_radiance_command.options.rfluxmtx import RfluxmtxOptions
from honeybee_radiance_command.options.rpict import RpictOptions
from honeybee_radiance_command.options.rtrace import RtraceOptions
from honeybee_radiance_postprocess.results import _filter_grids_by_pattern
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.wea import Wea
from ladybug_geometry.geometry3d import Plane, Point3D
from lbt_recipes.recipe import Recipe, RecipeSettings
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ...bhom.logging import CONSOLE_LOGGER
from ...ladybug_extension.analysisperiod import describe_analysis_period
from ..model import HbModelGeometry
from ..results import load_ill, load_npy, load_res
from .sensorgrids import as_patchcollection, get_limits, groupby_level

# pylint: enable=E0401


def radiance_parameters(
    model: Model,
    detail_dim: float,
    recipe_type: str,
    detail_level: int = 0,
    additional_parameters: str = None,
) -> str:
    """Generate the default "recommended" Radiance parameters for a Honeybee
    Radiance simulation.

    This method also includes the estimation of ambient resolution based on the
    model dimensions.

    Args:
        model: Model
            A Honeybee Model.
        detail_dim: float
            The detail dimension in meters.
        recipe_type: str
            One of the following: 'point-in-time-grid', 'daylight-factor',
            'point-in-time-image', 'annual'.
        detail_level: int
            One of 0 (low), 1 (medium) or 2 (high).
        additional_parameters: str
            Additional parameters to add to the Radiance command. Should be in
            the format of a Radiance command string e.g. '-ab 2 -aa 0.25'.

    Returns:
        str: The Radiance parameters as a string.
    """

    # recommendations for radiance parameters
    rtrace = {
        "ab": [2, 3, 6],
        "ad": [512, 2048, 4096],
        "as_": [128, 2048, 4096],
        "ar": [16, 64, 128],
        "aa": [0.25, 0.2, 0.1],
        "dj": [0, 0.5, 1],
        "ds": [0.5, 0.25, 0.05],
        "dt": [0.5, 0.25, 0.15],
        "dc": [0.25, 0.5, 0.75],
        "dr": [0, 1, 3],
        "dp": [64, 256, 512],
        "st": [0.85, 0.5, 0.15],
        "lr": [4, 6, 8],
        "lw": [0.05, 0.01, 0.005],
        "ss": [0, 0.7, 1],
    }

    rpict = {
        "ab": [2, 3, 6],
        "ad": [512, 2048, 4096],
        "as_": [128, 2048, 4096],
        "ar": [16, 64, 128],
        "aa": [0.25, 0.2, 0.1],
        "ps": [8, 4, 2],
        "pt": [0.15, 0.10, 0.05],
        "pj": [0.6, 0.9, 0.9],
        "dj": [0, 0.5, 1],
        "ds": [0.5, 0.25, 0.05],
        "dt": [0.5, 0.25, 0.15],
        "dc": [0.25, 0.5, 0.75],
        "dr": [0, 1, 3],
        "dp": [64, 256, 512],
        "st": [0.85, 0.5, 0.15],
        "lr": [4, 6, 8],
        "lw": [0.05, 0.01, 0.005],
        "ss": [0, 0.7, 1],
    }

    rfluxmtx = {
        "ab": [3, 5, 6],
        "ad": [5000, 15000, 25000],
        "as_": [128, 2048, 4096],
        "ds": [0.5, 0.25, 0.05],
        "dt": [0.5, 0.25, 0.15],
        "dc": [0.25, 0.5, 0.75],
        "dr": [0, 1, 3],
        "dp": [64, 256, 512],
        "st": [0.85, 0.5, 0.15],
        "lr": [4, 6, 8],
        "lw": [0.000002, 6.67e-07, 4e-07],
        "ss": [0, 0.7, 1],
        "c": [1, 1, 1],
    }

    # VALIDATION
    recipe_types = {
        "point-in-time-grid": [rtrace, RtraceOptions()],
        "daylight-factor": [rtrace, RtraceOptions()],
        "point-in-time-image": [rpict, RpictOptions()],
        "annual": [rfluxmtx, RfluxmtxOptions()],
        "annual-daylight": [rfluxmtx, RfluxmtxOptions()],
        "annual-irradiance": [rfluxmtx, RfluxmtxOptions()],
        "sky-view": [rtrace, RtraceOptions()],
    }
    if recipe_type not in recipe_types:
        raise ValueError(
            f"recipe_type ({recipe_type}) must be one of {recipe_types.keys()}"
        )

    if detail_level not in [0, 1, 2]:
        raise ValueError(
            r"detail_level ({detail_level}) must be one of 0 (low), 1 (medium) or 2 (high)."
        )

    options, obj = recipe_types[recipe_type]
    for opt, vals in options.items():
        setattr(obj, opt, vals[detail_level])

    min_pt, max_pt = model.min, model.max
    x_dim = max_pt.x - min_pt.x
    y_dim = max_pt.y - min_pt.y
    z_dim = max_pt.z - min_pt.z
    longest_dim = max((x_dim, y_dim, z_dim))
    try:
        obj.ar = int((longest_dim * obj.aa) / detail_dim)
    except TypeError as _:
        obj.ar = int((longest_dim * 0.1) / detail_dim)

    if additional_parameters:
        obj.update_from_string(additional_parameters)

    return obj.to_radiance()


@dataclass(init=True, repr=True)
class HoneybeeRadiance:
    """A class for handling simulation using Radiance via Honeybee."""

    model: Model = field(init=True, repr=True)

    def __post_init__(self):
        # check that model contains sensor grids and can be simulated.
        if not self.model.properties.radiance.sensor_grids:
            raise ValueError(
                "The model does not contain any sensor grids. Please add sensor grids to the model."
            )

    @classmethod
    def from_hbjson(cls, hbjson_file: Path) -> "HoneybeeRadiance":
        """_"""
        hbjson_file = Path(hbjson_file)
        return cls(Model.from_hbjson(hbjson_file.as_posix()))

    @property
    def model_name(self) -> str:
        """The name of the model."""
        return self.model.identifier

    @property
    def output_directory(self) -> Path:
        """_"""
        _dir = Path(hb_folders.default_simulation_folder) / self.model_name
        _dir.mkdir(exist_ok=True, parents=True)
        return _dir

    def get_sensorgrids(
        self,
        grids_filter: tuple[str] = "*",
    ) -> list[SensorGrid]:
        """Return a list of sensor grids for the given model.

        Args:
            grids_filter (Tuple[str], optional):
                A list of strings to filter the grids.

        Returns:
            list[SensorGrid]:
                A list of sensor grids.
        """
        if not isinstance(grids_filter, (list, tuple, str)):
            raise ValueError(
                "The grids_filter must be a string, list or tuple of strings."
            )
        return _filter_by_pattern(
            self.model.properties.radiance.sensor_grids, grids_filter
        )

    def get_levels(self) -> list[float]:
        """Return a list of levels for the given model.

        Returns:
            list[float]:
                A list of levels.
        """
        level_grids = groupby_level(self.get_sensorgrids())
        return list(level_grids.keys())

    def simulate_daylight_factor(
        self,
        reload_old: bool = True,
        delete_tempfiles: bool = True,
        detail_level: int = 0,
        detail_dim: float = 0.15,
    ) -> Path:
        """Run a daylight factor recipe for the current model.

        Args:
            reload_old (bool, optional):
                Instead of running a full simulation, reload the existing results ... if they exist.
                Defaults to True.
            delete_tempfiles (bool, optional):
                Delete temporary files created during simulation. This should roughly half the results directory size.
                Defaults to True.
            detail_level (float, optional):
                Set the detail level of the simulation.
                Defaults to 0 which is the lowest detail level.
            detail_dim (float, optional):
                Set the minimum geometry size to be accounted for in the raytracing.
                Defaults to 0.15.

        Returns:
            Path:
                The directory containing simulation results
        """

        recipe_name = "daylight-factor"
        folder_name = "daylight_factor"

        CONSOLE_LOGGER.info(f"Running {recipe_name} recipe for model {self.model_name}")

        recipe = Recipe(recipe_name)
        params = radiance_parameters(
            model=self.model,
            detail_dim=detail_dim,
            recipe_type=recipe_name,
            detail_level=detail_level,
        )
        recipe.input_value_by_name("model", self.model)
        recipe.input_value_by_name("radiance-parameters", params)

        settings = RecipeSettings(
            folder=self.output_directory.as_posix(), reload_old=reload_old
        )

        results_folder = recipe.run(settings)

        if delete_tempfiles:
            for fp in (Path(results_folder) / f"{folder_name}/initial_results").glob(
                "**/*.res"
            ):
                fp.unlink()

        return Path(results_folder) / folder_name

    def daylight_factor(
        self,
        grids_filter: tuple[str] = "*",
    ) -> pd.DataFrame:
        """Obtain daylight factor results for the given model simulation folder.

        Args:
            grids_filter (Tuple[str], optional):
                A list of strings to filter the grids to return.
                Defaults to ("*") which includes all grids.

        Returns:
            pd.DataFrame:
                A pandas DataFrame with daylight factor results per grid requested.
        """

        results_folder = self.output_directory / "daylight_factor/results"
        if not results_folder.exists():
            raise ValueError(
                "The given folder does not contain daylight factor results. Try running a simulation first."
            )

        if not isinstance(grids_filter, (list, tuple, str)):
            raise ValueError(
                "The grids_filter must be a string, list or tuple of strings."
            )

        with open(results_folder / "grids_info.json") as f:
            grids_info = json.load(f)

        renamer = {gi["identifier"]: gi["name"] for gi in grids_info}

        files = [
            results_folder / f"{i['identifier']}.res"
            for i in _filter_grids_by_pattern(grids_info, grids_filter)
        ]

        return (pd.concat([load_res(files)], axis=1).sort_index(axis=1)).rename(
            columns=renamer, level=0
        )

    def summarise_daylight_factor(
        self,
        grids_filter: tuple[str] = "*",
    ) -> pd.DataFrame:
        """Summarise daylight factor results for the given model simulation folder.

        Args:
            grids_filter (Tuple[str], optional):
                A list of strings to filter the grids to return.
                Defaults to ("*") which includes all grids.

        Returns:
            pd.DataFrame:
                A pandas DataFrame with daylight factor results per grid requested.
        """

        summaries = []
        for grid, (grid_name, values) in list(
            zip(
                *[
                    self.get_sensorgrids(grids_filter),
                    self.daylight_factor(grids_filter).items(),
                ]
            )
        ):
            vals = values.dropna().values
            summaries.append(
                pd.Series(
                    {
                        "Area": grid.mesh.area,
                        "Minimum": vals.min(),
                        "Average": vals.mean(),
                        "Median": np.median(vals),
                        "Maximum": vals.max(),
                        "Uniformity": vals.min() / vals.mean(),
                        "Area <2%": np.array(grid.mesh.face_areas)[vals < 2].sum()
                        / grid.mesh.area
                        * 100,
                        "Area >5%": np.array(grid.mesh.face_areas)[vals > 5].sum()
                        / grid.mesh.area
                        * 100,
                    },
                    name=grid_name,
                )
            )
        return pd.DataFrame(summaries)

    def plot_daylight_factor(
        self,
        grids_filter: tuple[str] = "*",
    ) -> list[Path]:
        """Plot daylight factor results for the given model simulation folder.

        Args:
            grids_filter (Tuple[str], optional):
                A list of strings to filter the grids to return.
                Defaults to ("*") which includes all grids.

        Returns:
            list[Path]:
                A list of paths to the plots.
        """

        plot_directory = self.output_directory / "plots"
        plot_directory.mkdir(exist_ok=True, parents=True)

        level_grids = groupby_level(self.get_sensorgrids(grids_filter))
        results = self.daylight_factor(grids_filter)

        sps = []
        for nn, (level, grids) in enumerate(level_grids.items()):
            fig, ax = plt.subplots(
                1,
                1,
            )
            ax.set_aspect("equal")
            ax.autoscale()
            ax.axis("off")
            ax.set_title(f"{level:0.2f}m (Z-level)")

            # add grid/values
            for grid in grids:
                vals = results[grid.full_identifier].dropna().values
                pc = as_patchcollection(
                    sensorgrid=grid, cmap="magma", norm=Normalize(vmin=0, vmax=10)
                )
                pc.set_array(vals)
                ax.add_collection(pc)

            # add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cb = fig.colorbar(
                mappable=ax.get_children()[0], cax=cax, label="Daylight Factor (%)"
            )
            cb.outline.set_visible(False)

            # add wireframe
            for b in HbModelGeometry:
                pc = b.slice_polycollection(
                    model=self.model, plane=Plane(o=Point3D(0, 0, level + 0.5))
                )
                ax.add_collection(pc)

            xlims, ylims = get_limits(grids)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

            plt.tight_layout()

            # save the figure
            sp = plot_directory / f"daylight_factor_{nn:02d}.png"
            fig.savefig(sp, dpi=300, transparent=True)
            plt.close(fig)
            sps.append(sp)

        return sps

    def simulate_annual_daylight(
        self,
        wea: Wea,
        reload_old: bool = True,
        delete_tempfiles: bool = True,
        north: float = 0,
        detail_level: int = 0,
        detail_dim: float = 0.15,
    ) -> Path:
        """Run an annual daylight recipe for the model.

        Args:
            wea (Wea):
                A Ladybug Wea object.
            reload_old (bool, optional):
                Instead of running a full simulation, reload the existing results ... if they exist.
                Defaults to True.
            delete_tempfiles (bool, optional):
                Delete temporary files created during simulation. This should roughly half the results directory size.
                Defaults to True.
            north (float, optional):
                Set the angle to north. Defaults to 0 which assumes the model is correctly oriented.
            detail_level (float, optional):
                Set the detail level of the simulation. Defaults to 0 which is the lowest detail level.
            detail_dim (float, optional):
                Set the minimum geometry size to be accounted for in the raytracing. Defaults to 0.15.

        Returns:
            Path:
                The directory containing simulation results
        """

        recipe_name = "annual-daylight"
        folder_name = "annual_daylight"

        CONSOLE_LOGGER.info(f"Running {recipe_name} recipe for model {self.model_name}")

        recipe = Recipe(recipe_name)
        params = radiance_parameters(
            model=self.model,
            detail_dim=detail_dim,
            recipe_type=recipe_name,
            detail_level=detail_level,
        )
        recipe.input_value_by_name("model", self.model)
        recipe.input_value_by_name("wea", wea)
        recipe.input_value_by_name("radiance-parameters", params)
        recipe.input_value_by_name("north", north)

        settings = RecipeSettings(
            folder=self.output_directory.as_posix(), reload_old=reload_old
        )

        results_folder = recipe.run(settings)

        if delete_tempfiles:
            for fp in (Path(results_folder) / f"{folder_name}/calcs").glob("**/*.ill"):
                fp.unlink()

        return Path(results_folder) / folder_name

    def annual_daylight(
        self,
        grids_filter: tuple[str] = "*",
    ) -> pd.DataFrame:
        """Obtain annual daylight results for the given model simulation folder.

        Args:
            grids_filter (Tuple[str], optional):
                A list of strings to filter the grids.
                Defaults to ("*") which includes all grids.

        Returns:
            pd.DataFrame:
                A pandas DataFrame with daylight results per grid requested.
        """

        results_folder = self.output_directory / "annual_daylight/results"
        if not results_folder.exists():
            raise ValueError(
                "The given folder does not contain annual daylight results. Try running a simulation first."
            )

        if not isinstance(grids_filter, (list, tuple, str)):
            raise ValueError(
                "The grids_filter must be a string, list or tuple of strings."
            )

        with open(results_folder / "grids_info.json") as f:
            grids_info = json.load(f)

        renamer = {gi["identifier"]: gi["name"] for gi in grids_info}

        files = [
            results_folder / f"__static_apertures__/default/total/{i['identifier']}.npy"
            for i in _filter_grids_by_pattern(grids_info, grids_filter)
        ]

        return (pd.concat([load_npy(files)], axis=1).sort_index(axis=1)).rename(
            columns=renamer, level=0
        )

    def simulate_annual_irradiance(
        self,
        wea: Wea,
        reload_old: bool = True,
        delete_tempfiles: bool = True,
        north: float = 0,
        detail_level: int = 0,
        detail_dim: float = 0.15,
    ) -> Path:
        """Run an annual irradiance recipe for the model.

        Args:
            wea (Wea):
                A Ladybug Wea object.
            reload_old (bool, optional):
                Instead of running a full simulation, reload the existing results ... if they exist.
                Defaults to True.
            delete_tempfiles (bool, optional):
                Delete temporary files created during simulation. This should roughly half the results directory size.
                Defaults to True.
            north (float, optional):
                Set the angle to north. Defaults to 0 which assumes the model is correctly oriented.
            detail_level (float, optional):
                Set the detail level of the simulation. Defaults to 0 which is the lowest detail level.
            detail_dim (float, optional):
                Set the minimum geometry size to be accounted for in the raytracing. Defaults to 0.15.

        Returns:
            Path:
                The directory containing simulation results
        """

        recipe_name = "annual-irradiance"
        folder_name = "annual_irradiance"

        CONSOLE_LOGGER.info(f"Running {recipe_name} recipe for model {self.model_name}")

        recipe = Recipe(recipe_name)
        params = radiance_parameters(
            model=self.model,
            detail_dim=detail_dim,
            recipe_type=recipe_name,
            detail_level=detail_level,
        )
        recipe.input_value_by_name("model", self.model)
        recipe.input_value_by_name("wea", wea)
        recipe.input_value_by_name("output-type", "solar")
        recipe.input_value_by_name("radiance-parameters", params)
        recipe.input_value_by_name("north", north)

        settings = RecipeSettings(
            folder=self.output_directory.as_posix(), reload_old=reload_old
        )

        results_folder = recipe.run(settings)

        if delete_tempfiles:
            for fp in (Path(results_folder) / f"{folder_name}/initial_results").glob(
                "**/*.ill"
            ):
                fp.unlink()

        return Path(results_folder) / folder_name

    def annual_irradiance(
        self,
        grids_filter: tuple[str] = "*",
        irradiance_type: str = "total",
    ) -> pd.DataFrame:
        """Obtain annual irradiance results for the given model simulation folder.

        Args:
            grids_filter (Tuple[str], optional):
                A list of strings to filter the grids.
                Defaults to ("*") which includes all grids.
            irradiance_type (str, optional):
                The type of irradiance to return. One of "total", "direct".
                Defaults to "total".

        Returns:
            pd.DataFrame:
                A pandas DataFrame with irradiance results per grid requested.
        """

        results_folder = (
            self.output_directory / f"annual_irradiance/results/{irradiance_type}"
        )
        if not results_folder.exists():
            raise ValueError(
                "The given folder does not contain annual irradiance results. Try running a simulation first."
            )

        if not isinstance(grids_filter, (list, tuple, str)):
            raise ValueError(
                "The grids_filter must be a string, list or tuple of strings."
            )

        with open(results_folder / "grids_info.json") as f:
            grids_info = json.load(f)

        renamer = {gi["identifier"]: gi["name"] for gi in grids_info}

        files = [
            results_folder / f"{i['identifier']}.ill"
            for i in _filter_grids_by_pattern(grids_info, grids_filter)
        ]

        return (pd.concat([load_ill(files)], axis=1).sort_index(axis=1)).rename(
            columns=renamer, level=0
        )

    def simulate_sky_view(
        self,
        reload_old: bool = True,
        delete_tempfiles: bool = True,
        detail_level: int = 0,
        detail_dim: float = 0.15,
    ) -> Path:
        """Run a sky view recipe for the current model.

        Args:
            reload_old (bool, optional):
                Instead of running a full simulation, reload the existing results ... if they exist.
                Defaults to True.
            delete_tempfiles (bool, optional):
                Delete temporary files created during simulation. This should roughly half the results directory size.
                Defaults to True.
            detail_level (float, optional):
                Set the detail level of the simulation.
                Defaults to 0 which is the lowest detail level.
            detail_dim (float, optional):
                Set the minimum geometry size to be accounted for in the raytracing.
                Defaults to 0.15.

        Returns:
            Path:
                The directory containing simulation results
        """

        recipe_name = "sky-view"
        folder_name = "sky_view"

        CONSOLE_LOGGER.info(f"Running {recipe_name} recipe for model {self.model_name}")

        recipe = Recipe(recipe_name)
        params = radiance_parameters(
            model=self.model,
            detail_dim=detail_dim,
            recipe_type=recipe_name,
            detail_level=detail_level,
        )
        recipe.input_value_by_name("model", self.model)
        recipe.input_value_by_name("radiance-parameters", params)

        settings = RecipeSettings(
            folder=self.output_directory.as_posix(), reload_old=reload_old
        )

        results_folder = recipe.run(settings)

        if delete_tempfiles:
            for fp in (Path(results_folder) / f"{folder_name}/initial_results").glob(
                "**/*.res"
            ):
                fp.unlink()

        return Path(results_folder) / folder_name

    def sky_view(
        self,
        grids_filter: tuple[str] = "*",
    ) -> pd.DataFrame:
        """Obtain sky view results for the given model simulation folder.

        Args:
            grids_filter (Tuple[str], optional):
                A list of strings to filter the grids.
                Defaults to ("*") which includes all grids.

        Returns:
            pd.DataFrame:
                A pandas DataFrame with sky view results per grid requested.
        """
        results_folder = self.output_directory / "sky_view/results/sky_view"
        if not results_folder.exists():
            raise ValueError(
                "The given folder does not contain sky view results. Try running a simulation first."
            )

        if not isinstance(grids_filter, (list, tuple, str)):
            raise ValueError(
                "The grids_filter must be a string, list or tuple of strings."
            )

        with open(results_folder / "grids_info.json") as f:
            grids_info = json.load(f)

        renamer = {gi["identifier"]: gi["name"] for gi in grids_info}

        files = [
            results_folder / f"{i['identifier']}.res"
            for i in _filter_grids_by_pattern(grids_info, grids_filter)
        ]

        return (pd.concat([load_res(files)], axis=1).sort_index(axis=1)).rename(
            columns=renamer, level=0
        )

    def simulate_direct_sun_hours(
        self,
        wea: Wea,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        north: float = 0,
        delete_tempfiles: bool = True,
        reload_old: bool = True,
    ) -> Path:
        """Run a direct sun hours recipe for the current model.

        Args:
            wea (Wea):
                A Ladybug Wea object.
            analysis_period (AnalysisPeriod, optional):
                The analysis period to run the simulation for. Defaults to AnalysisPeriod().
            north (float, optional):
                Set the angle to north. Defaults to 0 which assumes the model is correctly oriented.
            delete_tempfiles (bool, optional):
                Delete temporary files created during simulation. This should roughly half the results directory size.
                Defaults to True.
            reload_old (bool, optional):
                Instead of running a full simulation, reload the existing results ... if they exist.
                Defaults to True.

        Returns:
            Path:
                The directory containing simulation results
        """

        ap_str = describe_analysis_period(
            analysis_period=analysis_period, save_path=True, include_timestep=True
        )
        simulation_directory = self.output_directory / f"direct_sun_hours_{ap_str}"
        simulation_directory.mkdir(exist_ok=True, parents=True)

        recipe_name = "direct-sun-hours"
        folder_name = "direct_sun_hours"

        CONSOLE_LOGGER.info(
            f"Running {recipe_name} recipe for model {self.model_name} over "
            f"{describe_analysis_period(analysis_period, include_timestep=True)}"
        )

        recipe = Recipe(recipe_name)
        recipe.input_value_by_name("model", self.model)
        recipe.input_value_by_name("wea", wea)
        recipe.input_value_by_name("timestep", analysis_period.timestep)
        recipe.input_value_by_name("north", north)

        settings = RecipeSettings(
            folder=simulation_directory.as_posix(), reload_old=reload_old
        )

        results_folder = recipe.run(settings)

        if delete_tempfiles:
            for fp in (Path(results_folder) / f"{folder_name}/initial_results").glob(
                "**/*.res"
            ):
                fp.unlink()
            for fp in (Path(results_folder) / f"{folder_name}/initial_results").glob(
                "**/*.ill"
            ):
                fp.unlink()

        return Path(results_folder) / folder_name

    def direct_sun_hours(
        self,
        grids_filter: tuple[str] = "*",
        sun_hours_type: str = "cumulative",
    ) -> pd.DataFrame:
        """Obtain direct sun hours results for the given model simulation folder.

        Args:
            grids_filter (Tuple[str], optional):
                A list of strings to filter the grids.
                Defaults to ("*") which includes all grids.
            sun_hours_type (str, optional):
                The type of direct sun hours to return. One of "cumulative".
                Defaults to "cumulative".

        Returns:
            pd.DataFrame:
                A pandas DataFrame with direct sun hours results per grid requested.
        """

        if sun_hours_type != "cumulative":
            raise NotImplementedError(
                "Only cumulative direct sun hours are currently supported."
            )

        results_folders = list(self.output_directory.glob("direct_sun_hours_*"))

        if not results_folders:
            raise ValueError(
                "The given folder does not contain direct sun hours results. Try running a simulation first."
            )

        dfs = []
        for results_folder in results_folders:
            rf = results_folder / f"direct_sun_hours/results/{sun_hours_type}"
            with open(rf / "grids_info.json") as f:
                grids_info = json.load(f)

            renamer = {gi["identifier"]: gi["name"] for gi in grids_info}

            files = [
                rf / f"{i['identifier']}.res"
                for i in _filter_grids_by_pattern(grids_info, grids_filter)
            ]
            df = load_res(files).rename(columns=renamer)
            dfs.append(df)

        return pd.concat(dfs, axis=1, keys=[rf.name for rf in results_folders])
