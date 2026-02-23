import concurrent
import inspect
import json
import os
import shlex
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, TypeVar

import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from honeybee.model import Aperture, Face, Floor, Model, RoofCeiling, Room, Shade, Wall
from honeybee_radiance.modifier.material import Glass, Plastic
from honeybee_radiance.modifierset import (
    ApertureModifierSet,
    FloorModifierSet,
    ModifierSet,
    RoofCeilingModifierSet,
    ShadeModifierSet,
    WallModifierSet,
)
from honeybee_radiance.sensorgrid import SensorGrid
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.energyflux import Irradiance
from ladybug.epw import EPW, AnalysisPeriod, Header
from ladybug.wea import Wea
from ladybug_geometry.geometry2d import Vector2D
from ladybug_geometry.geometry3d import Vector3D
from pydantic import BaseModel, Field, root_validator

from ..convert.to_ladybug import to_ladybug
from ..convert.to_pandas import to_pandas
from ..honeybee_radiance_extension.util import load_ill, load_npy, make_annual #TODO: I know this already exists somewhere
from ..bhom.logger import CONSOLE_LOGGER
from ..util import deterministic_hash, run_cmd #TODO

T = TypeVar("T", bound="BaseModel")


class WindowShadeConfig(BaseModel):
    # location/weather
    epw_file: str = Field(
        description="Path to the EPW file for the location of this building.",
        unit="str",
    )
    # window properties
    window_width: float = Field(
        description="Width of the window in meters.", default=3, gt=0.0
    )
    window_height: float = Field(
        description="Height of the window in meters.", default=2, gt=0.0
    )
    window_sill_height: float = Field(
        description="Height of the window sill from the floor in meters.",
        default=1.0,
        ge=0.0,
    )
    # space properties
    orientation: float = Field(
        description="Orientation of the window in degrees clockwise from North.",
        default=180,
        ge=0.0,
        le=360.0,
    )
    zone_depth: float = Field(
        description="Depth of the zone the window is in, in meters, from the internal reveal thickness.",
        default=5.0,
        gt=1.0,
    )
    zone_width: float = Field(
        description="Width of the zone the window is in, in meters.",
        default=5.0,
        gt=1.0,
    )
    zone_height: float = Field(
        description="Height of the zone the window is in, in meters.",
        default=3.1,
        gt=1.0,
    )
    # shade properties
    external_reveal_depth: float = Field(
        description="Depth of the window reveal in meters.", default=0.0, ge=0.0
    )
    internal_reveal_depth: float = Field(
        description="Depth of the internal window reveal in meters.",
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    vertical_louver_spacing: float = Field(
        description="The spacing between vertical fins in m.",
        default=0.0,
        ge=0.0,
        le=5.0,
        allow_inf_nan=True,
    )
    vertical_louver_angle: float = Field(
        description="The angle of the vertical fins in degrees from the window plane. 0 would be window normal, while -45 would point 'left' from the interior perspective and +45 would point 'right'",
        default=0.0,
        ge=-45.0,
        le=45.0,
    )
    vertical_louver_offset: float = Field(
        description="The distance of the vertical fins from the window plane in meters.",
        default=0.0,
        ge=0.0,
        le=2.0,
    )
    vertical_louver_depth: float = Field(
        description="The depth of the vertical fins in meters.",
        default=0.1,
        ge=0.0,
        le=1.0,
    )
    horizontal_louver_spacing: float = Field(
        description="The spacing between horizontal fins in m.",
        default=0.0,
        ge=0.0,
        le=5.0,
        allow_inf_nan=True,
    )
    horizontal_louver_angle: float = Field(
        description="The angle of the horizontal fins in degrees from the window plane. 0 would be window normal, while -45 would point 'down' from the interior perspective and +45 would point 'up'",
        default=0.0,
        ge=-45.0,
        le=45.0,
    )
    horizontal_louver_offset: float = Field(
        description="The distance of the horizontal fins from the window plane in meters.",
        default=0.0,
        ge=0.0,
        le=2.0,
    )
    horizontal_louver_depth: float = Field(
        description="The depth of the horizontal fins in meters.", default=0.1, ge=0.0, le=1.0,
    )
    # fabric performance
    window_solar_heat_gain_coefficient: float = Field(
        description="Solar Heat Gain Coefficient (SHGC) of the window.",
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    window_light_transmissivity: float = Field(
        description="Visible light transmissivity of the window.",
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    fin_reflectance: float = Field(
        description="Reflectivity of the shading fins.", default=0.5, ge=0.0, le=1.0
    )
    zone_floor_reflectance: float = Field(
        description="Reflectivity of the zone floor.", default=0.2, ge=0.0, le=1.0
    )
    zone_ceiling_reflectance: float = Field(
        description="Reflectivity of the zone ceiling.", default=0.8, ge=0.0, le=1.0
    )
    zone_wall_reflectance: float = Field(
        description="Reflectivity of the zone walls.", default=0.5, ge=0.0, le=1.0
    )

    @root_validator(pre=False)
    def interrelated_things(cls, values):
        # dimension checks
        internal_reveal_depth = values.get("internal_reveal_depth")
        zone_depth = values.get("zone_depth")
        if internal_reveal_depth >= zone_depth:
            raise ValueError("internal_reveal_depth must be less than zone_depth")

        zone_width = values.get("zone_width")
        window_width = values.get("window_width")
        if zone_width <= window_width:
            raise ValueError("zone_width must be greater than window_width")

        zone_height = values.get("zone_height")
        window_height = values.get("window_height")
        if zone_height <= window_height:
            raise ValueError("zone_height must be greater than window_height")

        window_sill_height = values.get("window_sill_height")
        if window_sill_height + window_height >= zone_height:
            raise ValueError(
                f"window_sill_height ({window_sill_height}) + window_height ({window_height}) must be less than zone_height ({zone_height})"
            )

        # louver info
        vertical_louver_spacing = values.get("vertical_louver_spacing")
        vertical_louver_depth = values.get("vertical_louver_depth")
        if np.isinf(vertical_louver_spacing) and vertical_louver_depth > 0.0:
            CONSOLE_LOGGER.warning(
                "vertical_louver_spacing is infinite, but vertical_louver_depth is greater than 0. Setting vertical_louver_depth to 0."
            )
            values["vertical_louver_depth"] = 0.0
            values["vertical_louver_spacing"] = window_width

        horizontal_louver_spacing = values.get("horizontal_louver_spacing")
        horizontal_louver_depth = values.get("horizontal_louver_depth")
        if np.isinf(horizontal_louver_spacing) and horizontal_louver_depth > 0.0:
            CONSOLE_LOGGER.warning(
                "horizontal_louver_spacing is infinite, but horizontal_louver_depth is greater than 0. Setting horizontal_louver_depth to 0."
            )
            values["horizontal_louver_depth"] = 0.0
            values["horizontal_louver_spacing"] = window_height

        return values

    @classmethod
    def from_random(cls: type[T], epw_file: str, seed: Optional[int] = None) -> T:
        """Create this object with random values."""
        np.random.seed(seed)

        values = {}
        for name, field in cls.__fields__.items():
            field_type = field.outer_type_
            constraints = field.field_info

            # enums
            if hasattr(field_type, "__members__"):
                values[name] = np.random.choice(list(field_type))

            # bool
            elif inspect.isclass(field_type) and issubclass(field_type, bool):
                values[name] = np.random.choice([True, False])

            # int
            elif inspect.isclass(field_type) and issubclass(field_type, int):
                # get min possible and max possible values
                vmin = constraints.ge if constraints.ge is not None else constraints.gt
                vmax = constraints.le if constraints.le is not None else constraints.lt
                if vmin is None:
                    vmin = 1
                if vmax is None:
                    vmax = vmin + 10
                values[name] = np.random.randint(
                    int(vmin), int(vmax) + 1, dtype=np.int64
                )

            # float
            elif inspect.isclass(field_type) and issubclass(field_type, float):
                # get min possible and max possible values
                vmin = constraints.ge if constraints.ge is not None else constraints.gt
                vmax = constraints.le if constraints.le is not None else constraints.lt
                if vmin is None:
                    vmin = 0.1
                if vmax is None:
                    vmax = vmin + 10.0
                values[name] = float(np.random.uniform(float(vmin), float(vmax)))

            # str
            elif field_type is str:
                values[name] = "{}_{}".format(
                    cls.__name__, np.random.randint(100_000, 999_999)
                )

            # nested pydantic models
            elif inspect.isclass(field_type) and issubclass(field_type, BaseModel):
                values[name] = field_type.from_random(seed=seed)  # type: ignore

            else:
                values[name] = None

        # modify values to to ensure rules are followed
        if values["internal_reveal_depth"] >= values["zone_depth"]:
            values["internal_reveal_depth"] = values["zone_depth"] / 2.0
        if values["zone_width"] <= values["window_width"]:
            values["zone_width"] = values["window_width"] + 1.0
        if values["zone_height"] <= values["window_height"]:
            values["zone_height"] = values["window_height"] + 1.0
        window_sill_height = values["window_sill_height"]
        if window_sill_height + values["window_height"] >= values["zone_height"]:
            values["window_sill_height"] = (
                values["zone_height"] - values["window_height"] - 0.1
            )

        values["epw_file"] = epw_file
        return cls(**values)  # type: ignore

    @property
    def _has_shades(self) -> bool:
        """Check if the configuration has any shading elements."""
        return any(
            [
                self.external_reveal_depth > 0.0,
                self.internal_reveal_depth > 0.0,
                self.vertical_louver_depth > 0.0,
                self.horizontal_louver_depth > 0.0,
            ]
        )

    @property
    def unique_id(self):
        """A unique hash for this object."""
        d = self.dict()

        # remove all keys that end with _lock or _cache
        keys_to_remove = [
            key for key in d.keys() if key.endswith("_lock") or key.endswith("_cache")
        ]
        for key in keys_to_remove:
            d.pop(key)

        if not self._has_vertical_louvers():
            d.pop("vertical_louver_spacing")
            d.pop("vertical_louver_angle")
            d.pop("vertical_louver_offset")
            d.pop("vertical_louver_depth")

        if not self._has_horizontal_louvers():
            d.pop("horizontal_louver_spacing")
            d.pop("horizontal_louver_angle")
            d.pop("horizontal_louver_offset")
            d.pop("horizontal_louver_depth")

        return deterministic_hash(d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.unique_id:<020d}"

    def without_shades(self) -> "WindowShadeConfig":
        """Get a copy of this configuration without any shading elements."""
        return self.copy(
            update={
                "external_reveal_depth": 0.0,
                "internal_reveal_depth": 0.0,
                "vertical_louver_depth": 0.0,
                "horizontal_louver_depth": 0.0,
                "vertical_louver_spacing": 0.0,
                "horizontal_louver_spacing": 0.0,
            }
        )

    @property
    def aperture_area(self) -> float:
        """Get the area of the window aperture in m2."""
        return self.window_width * self.window_height

    @property
    def directory(self) -> Path:
        """Get the directory path for this shading configuration."""

        # some configurations are effectively identical, so these properties are handled here
        # the presence of vertical fins and horizontal fins

        base_dir = Path(hb_folders.default_simulation_folder) / ".window_shade"
        dir_path = base_dir / f"{self.__class__.__name__}_{self.unique_id:<020d}"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    # region: HB Model
    # methods for generating aspects of the model that is simulated

    def _wall_modifier(self) -> Plastic:
        """Create wall modifier for the zone."""
        return Plastic.from_single_reflectance(
            identifier=f"wall_modifier_{self.zone_wall_reflectance}",
            rgb_reflectance=self.zone_wall_reflectance,
        )

    def _roof_ceiling_modifier(self) -> Plastic:
        """Create roof/ceiling modifier for the zone."""
        return Plastic.from_single_reflectance(
            identifier=f"roof_ceiling_modifier_{self.zone_ceiling_reflectance}",
            rgb_reflectance=self.zone_ceiling_reflectance,
        )

    def _floor_modifier(self) -> Plastic:
        """Create floor modifier for the zone."""
        return Plastic.from_single_reflectance(
            identifier=f"floor_modifier_{self.zone_floor_reflectance}",
            rgb_reflectance=self.zone_floor_reflectance,
        )

    def _louver_shade_modifier(self) -> Plastic:
        """Create louver shade modifier for the window."""
        return Plastic.from_single_reflectance(
            identifier=f"louver_shade_modifier_{self.fin_reflectance}",
            rgb_reflectance=self.fin_reflectance,
        )

    def _aperture_modifier(self) -> Glass:
        """Create aperture modifier for the window."""
        return Glass.from_single_transmissivity(
            identifier=f"aperture_modifier_{self.window_light_transmissivity}",
            rgb_transmissivity=self.window_light_transmissivity,  # type: ignore
        )

    def _reveal_shade_exterior(self) -> list[Shade]:
        """Create exterior reveal depth shades for the window."""
        shades: list[Shade] = []
        aperture = self._aperture()
        if self.external_reveal_depth > 0:
            temp_shades = aperture.extruded_border(
                depth=self.external_reveal_depth, indoor=False
            )
            for shd in temp_shades:
                shd: Shade
                shd._parent = None
            shades.extend(temp_shades)
        for shade in shades:
            shade.properties.radiance.modifier = self._wall_modifier()  # type: ignore
        return shades

    def _reveal_shade_interior(self) -> list[Shade]:
        """Create interior reveal depth shades for the window."""
        shades: list[Shade] = []
        aperture = self._aperture()
        if self.internal_reveal_depth > 0:
            temp_shades = aperture.extruded_border(
                depth=self.internal_reveal_depth, indoor=True
            )
            for shd in temp_shades:
                shd: Shade
                shd._parent = None
            shades.extend(temp_shades)
        for shade in shades:
            shade.properties.radiance.modifier = self._wall_modifier()  # type: ignore
        return shades

    def _louver_shade_vertical(self) -> list[Shade]:
        """Create vertical louvers for the window."""
        shades: list[Shade] = []
        aperture = self._aperture()
        if self._has_vertical_louvers():
            temp_shades = aperture.louvers_by_distance_between(
                distance=self.vertical_louver_spacing,
                depth=self.vertical_louver_depth,
                offset=self.vertical_louver_offset,  # type: ignore
                angle=self.vertical_louver_angle,  # type: ignore
                indoor=False,
                contour_vector=Vector2D(1, 0),
            )
            for shd in temp_shades:
                shd: Shade
                shd._parent = None
            shades.extend(temp_shades)
        for shade in shades:
            shade.properties.radiance.modifier = self._louver_shade_modifier()  # type: ignore
        return shades

    def _louver_shade_horizontal(self) -> list[Shade]:
        """Create horizontal louvers for the window."""
        shades: list[Shade] = []
        aperture = self._aperture()
        if self._has_horizontal_louvers():
            temp_shades = aperture.louvers_by_distance_between(
                distance=self.horizontal_louver_spacing,
                depth=self.horizontal_louver_depth,
                offset=self.horizontal_louver_offset,  # type: ignore
                angle=-self.horizontal_louver_angle,  # type: ignore
                indoor=False,
                contour_vector=Vector2D(0, 1),
            )
            for shd in temp_shades:
                shd: Shade
                shd._parent = None
            shades.extend(temp_shades)
        for shade in shades:
            shade.properties.radiance.modifier = self._louver_shade_modifier()  # type: ignore
        return shades

    def _room(self) -> Room:
        """Create the room for the shading configuration."""

        # the room is made with front face to north by default
        room = Room.from_box(
            identifier="room",
            width=self.zone_width,
            depth=self.zone_depth,
            height=self.zone_height,
        )
        room.move(
            moving_vec=Vector3D(
                x=-self.zone_width / 2,  # type: ignore
                y=-self.zone_depth / 2,  # type: ignore
                z=0,
            )
        )
        room.rotate_xy(angle=-self.orientation, origin=room.center)

        wall_modifier = self._wall_modifier()
        roof_ceiling_modifier = self._roof_ceiling_modifier()
        floor_modifier = self._floor_modifier()

        for face in room.faces:
            face: Face
            if face.type == Wall():
                face.properties.radiance.modifier = wall_modifier  # type: ignore
            if face.type == RoofCeiling():
                face.properties.radiance.modifier = roof_ceiling_modifier  # type: ignore
            if face.type == Floor():
                face.properties.radiance.modifier = floor_modifier  # type: ignore

        return room

    def _aperture(self) -> Aperture:
        """Create the window aperture for the shading configuration."""
        room = self._room()
        face: Face = room.faces[1]
        aperture: Aperture = face.aperture_by_width_height(
            width=self.window_width,
            height=self.window_height,
            sill_height=self.window_sill_height,  # type: ignore
        )

        aperture_modifier = self._aperture_modifier()
        aperture.properties.radiance.modifier = aperture_modifier  # type: ignore
        return aperture

    def _sensor_grid_working_plane(self) -> SensorGrid:
        """Create the sensor grid for the working plane."""

        # create room , with modification to depth to account for any interior reveal depth
        original_room = self._room()
        room = Room.from_box(
            identifier="_room",
            width=self.zone_width,
            depth=self.zone_depth - self.internal_reveal_depth,
            height=self.zone_height,
        )
        room.move(
            moving_vec=Vector3D(
                x=-self.zone_width / 2,  # type: ignore
                y=-self.zone_depth / 2,  # type: ignore
                z=0,
            )
        )
        room.rotate_xy(angle=-self.orientation, origin=original_room.center)

        plane_height = 0.8
        mesh = room.generate_grid(
            x_dim=0.5,
            y_dim=0.5,
            offset=plane_height,  # type: ignore
        )
        return SensorGrid.from_mesh3d(
            identifier="working_plane",
            mesh=mesh,
        )

    def _sensor_grid_aperture(self) -> SensorGrid:
        """Create the sensor grid for the window aperture."""
        aperture = self._aperture()

        x_dim = 0.5
        y_dim = 0.5

        # if there are vertical louvers, then override the horizontal spacing so that sensors are between each louver
        if self.vertical_louver_depth > 0 and self.vertical_louver_spacing > 0:
            x_dim = min(self.vertical_louver_spacing, x_dim)

        # if there are horizontal louvers, then override the vertical spacing so that sensors are between each louver
        if self.horizontal_louver_depth > 0 and self.horizontal_louver_spacing > 0:
            y_dim = min(self.horizontal_louver_spacing, y_dim)

        return SensorGrid.from_mesh3d(
            identifier="aperture",
            mesh=aperture.geometry.mesh_grid(
                x_dim=x_dim,
                y_dim=y_dim,
                offset=0.001,  # type: ignore
            ),
        )

    def _has_vertical_louvers(self) -> bool:
        """Check if the configuration has vertical louvers."""
        return (
            self.vertical_louver_depth > 0
            and self.vertical_louver_spacing > 0
            and self.vertical_louver_spacing < self.window_width
        )

    def _has_horizontal_louvers(self) -> bool:
        """Check if the configuration has horizontal louvers."""
        return (
            self.horizontal_louver_depth > 0
            and self.horizontal_louver_spacing > 0
            and self.horizontal_louver_spacing < self.window_height
        )

    def hb_model(self) -> Model:
        """Create the model to simulate shading performance using Radiance."""

        # create room
        room = self._room()

        # add aperture to room
        room.faces[1].add_sub_face(self._aperture())

        # create shades
        shades = (
            self._reveal_shade_interior()
            + self._reveal_shade_exterior()
            + self._louver_shade_vertical()
            + self._louver_shade_horizontal()
        )

        # create model
        model = Model(
            identifier=f"{self.__class__.__name__}_{self.unique_id:<020d}",
            rooms=[room],
            orphaned_shades=shades,
        )

        # create and add sensor grids
        sensor_grids = [
            self._sensor_grid_aperture(),
            self._sensor_grid_working_plane(),
        ]
        model.properties.radiance.add_sensor_grids(sensor_grids)  # type: ignore

        return model

    # endregion: HB Model

    @property
    def epw(self) -> EPW:
        """Get the EPW object for the EPW file location.

        Returns:
            EPW: The EPW object for the EPW file location.
        """

        return EPW(self.epw_file)

    @property
    def wea(self) -> Wea:
        """Get the Wea object for the EPW file location.

        Returns:
            Wea: The Wea object for the EPW file location.
        """

        return Wea.from_epw_file(self.epw_file)

    def aperture_irradiance(self) -> HourlyContinuousCollection:
        """Get the hourly radiation on the window aperture.

        Returns:
            HourlyContinuousCollection: The hourly radiation on the window aperture in W/m2.
        """

        directory = self.directory
        recipe_name = "annual-irradiance"
        model = self.hb_model()
        grid_name = "aperture"

        model_hbjson_file = directory / "model.hbjson"
        sim_epw_file = directory / Path(self.epw_file).name
        wea_file = directory / Path(self.epw_file).with_suffix(".wea").name
        inputs_json_file = directory / f"{recipe_name}_inputs.json".replace("-", "_")
        results_file = (
            directory
            / recipe_name.replace("-", "_")
            / "results/total"
            / f"{grid_name}.ill"
        )
        sun_up_hours_file = (
            directory
            / recipe_name.replace("-", "_")
            / "results/total"
            / "sun-up-hours.txt"
        )

        if not all(
            i.exists()
            for i in [
                model_hbjson_file,
                sim_epw_file,
                wea_file,
                inputs_json_file,
                results_file,
                sun_up_hours_file,
            ]
        ):
            # write model
            if not model_hbjson_file.exists():
                model.to_hbjson(folder=directory.as_posix(), name="model")

            # write EPW
            if not sim_epw_file.exists():
                sim_epw_file.write_bytes(Path(self.epw_file).read_bytes())

            # write WEA
            if not wea_file.exists():
                self.wea.write(wea_file.as_posix())

            # create inputs JSON file
            inputs_dict = {
                "grid-filter": f"{grid_name}",
                "model": model_hbjson_file.as_posix(),
                "north": 0.0,
                "output-type": "solar",
                "radiance-parameters": "-ab 2 -ad 5000 -lw 2e-05 -dr 0",
                "timestep": 1,
                "wea": wea_file.as_posix(),
            }
            if not inputs_json_file.exists():
                with open(inputs_json_file, "w") as f:
                    json.dump(inputs_dict, f, indent=4)

            # run simulation
            cmd = (
                f"lbt-recipes run {recipe_name} "
                f"{shlex.quote(inputs_json_file.as_posix())} "
                f"--project-folder {shlex.quote(directory.as_posix())} "
                f"-w {os.cpu_count() - 1} "  # type: ignore
            )
            CONSOLE_LOGGER.debug(f"{model.identifier} - {cmd}")
            run_cmd(cmd=cmd, cwd=directory, log_id=model.identifier)

        # load results
        rad_df = make_annual(
            load_ill(ill_file=results_file, sun_up_hours_file=sun_up_hours_file)
        ).fillna(0)

        # get sensorgrid mesh pt areas to attribute are to each pt and get average across grid
        sensor_grid: SensorGrid = [
            i
            for i in model.properties.radiance.sensor_grids  # type: ignore
            if i.identifier == grid_name
        ][0]
        pt_areas = sensor_grid.mesh.face_areas  # type: ignore
        rad_values = np.average(
            rad_df.values,
            weights=pt_areas,
            axis=1,
        )

        rad_collection = HourlyContinuousCollection(
            header=Header(
                data_type=Irradiance(),
                unit="W/m2",
                analysis_period=AnalysisPeriod(),
                metadata={
                    "epw": Path(self.epw_file).name,
                    "config": model.identifier,
                    "time-zone": self.epw.location.time_zone,
                },
            ),
            values=rad_values.tolist(),
        )

        return rad_collection

    def working_plane_illuminance(self) -> pd.DataFrame:
        """"""
        directory = self.directory
        recipe_name = "annual-daylight-enhanced"
        model = self.hb_model()
        grid_name = "working_plane"

        model_hbjson_file = directory / "model.hbjson"
        sim_epw_file = directory / Path(self.epw_file).name
        wea_file = directory / Path(self.epw_file).with_suffix(".wea").name
        inputs_json_file = directory / f"{recipe_name}_inputs.json".replace("-", "_")
        results_file = (
            directory
            / recipe_name.replace("-", "_")
            / "results/__static_apertures__/default/total"
            / f"{grid_name}.npy"
        )
        sun_up_hours_file = (
            directory
            / recipe_name.replace("-", "_")
            / "results"
            / "sun-up-hours.txt"
        )

        if not all(
            i.exists()
            for i in [
                model_hbjson_file,
                sim_epw_file,
                wea_file,
                inputs_json_file,
                results_file,
                sun_up_hours_file,
            ]
        ):
            # write model
            if not model_hbjson_file.exists():
                model.to_hbjson(folder=directory.as_posix(), name="model")

            # write EPW
            if not sim_epw_file.exists():
                sim_epw_file.write_bytes(Path(self.epw_file).read_bytes())

            # write WEA
            if not wea_file.exists():
                self.wea.write(wea_file.as_posix())

            # create inputs JSON file
            inputs_dict = {
                "grid-filter": f"{grid_name}",
                "model": model_hbjson_file.as_posix(),
                "north": 0.0,
                "radiance-parameters": "-ab 2 -ad 5000 -lw 2e-05 -dr 0",
                "wea": wea_file.as_posix(),
                "thresholds": "-t 300 -lt 100 -ut 3000",
            }
            if not inputs_json_file.exists():
                with open(inputs_json_file, "w") as f:
                    json.dump(inputs_dict, f, indent=4)

            # run simulation
            cmd = (
                f"lbt-recipes run {recipe_name} "
                f"{shlex.quote(inputs_json_file.as_posix())} "
                f"--project-folder {shlex.quote(directory.as_posix())} "
                f"-w {os.cpu_count() - 1} "  # type: ignore
            )
            CONSOLE_LOGGER.debug(f"{model.identifier} - {cmd}")
            run_cmd(cmd=cmd, cwd=directory, log_id=model.identifier)
        
        # load results
        df = make_annual(load_npy(npy_file=results_file)).fillna(0)

        return df

    def peak_solar_gain(self) -> float:
        """Get the maximum solar gain through the window, in W.

        Returns:
            float: The maximum solar gain in W.
        """
        aperture_area = self.aperture_area
        collection = self.aperture_irradiance()
        collection_normed = collection.aggregate_by_area(
            area=aperture_area, area_unit="m2"
        )

        return collection_normed.max * self.window_solar_heat_gain_coefficient

    def shade_irradiance_reduction(self) -> pd.Series:
        """Get fractional change in irradiance onto the aperture compared to an unshaded version.

        Returns:
            HourlyContinuousCollection:
                The fractional reduction in irradiance onto the aperture due to any shading.
        """

        self_unshaded = self.without_shades()
        
        # check that unobstructed config is different
        if self_unshaded.unique_id != self.unique_id:
            with ProcessPoolExecutor(max_workers=2) as executor:
                future_unshaded = executor.submit(self_unshaded.aperture_irradiance)
                future_shaded = executor.submit(self.aperture_irradiance)
                unshaded_irradiance = future_unshaded.result()
                shaded_irradiance = future_shaded.result()
        else:
            unshaded_irradiance = self_unshaded.aperture_irradiance()
            shaded_irradiance = self.aperture_irradiance()
        
        # drop "config" key from metadata of each collection
        for col in [unshaded_irradiance, shaded_irradiance]:
            if "config" in col.header.metadata:
                col.header.metadata.pop("config")

        effectiveness: pd.Series = (
            1 - to_pandas(unshaded_irradiance - shaded_irradiance) / unshaded_irradiance
        ) * 100

        effectiveness.name = ("Fraction", "%", effectiveness.name[2])  # type: ignore

        return to_ladybug(effectiveness)

    def get_dla(self) -> float:
        """Get the Daylight Autonomy (DLA) for the shading configuration.

        Returns:
            float: The DLA value as a percentage (0-100).
        """
        raise NotImplementedError()

    def get_df(self) -> float:
        """Get the Daylight Factor (DF) for the shading configuration.

        Returns:
            float: The DF value as a percentage (0-100).
        """
        raise NotImplementedError()

    def max_solar_gain_time(self) -> str:
        """Get the time of year when maximum solar gain occurs.

        Returns:
            str: The time of year in 'Month Day' format.
        """
        raise NotImplementedError()

    def description(self) -> str:
        """Get a description of the shading configuration.

        Returns:
            str: A textual description of the shading configuration.
        """
        raise NotImplementedError()
        desc = f"WindowShadeConfig with window {self.window_width}m wide x {self.window_height}m high, sill at {self.window_sill_height}m, in a zone {self.zone_width}m wide x {self.zone_depth}m deep x {self.zone_height}m high, oriented at {self.orientation}°."
        if self.external_reveal_depth > 0:
            desc += f" External reveal depth of {self.external_reveal_depth}m."
        if self.internal_reveal_depth > 0:
            desc += f" Internal reveal depth of {self.internal_reveal_depth}m."
        if self.vertical_louver_depth > 0:
            desc += f" Vertical fins spaced every {self.vertical_louver_spacing}m, angled at {self.vertical_louver_angle}°, offset by {self.vertical_louver_offset}m, with depth {self.vertical_louver_depth}m."
        if self.horizontal_louver_depth > 0:
            desc += f" Horizontal fins spaced every {self.horizontal_louver_spacing}m, angled at {self.horizontal_louver_angle}°, offset by {self.horizontal_louver_offset}m, with depth {self.horizontal_louver_depth}m."
        return desc
