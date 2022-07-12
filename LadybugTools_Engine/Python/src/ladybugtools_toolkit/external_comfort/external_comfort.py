from __future__ import annotations

import getpass
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from honeybee.config import folders as hb_folders
from honeybee.model import Model
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug.epw import EPW

from .encoder import Encoder
from .model import create_model
from .simulate import surface_temperature_results_exist, solar_radiation_results_exist

hb_folders.default_simulation_folder = f"C:/Users/{getpass.getuser()}/simulation"


@dataclass(frozen=True)
class ExternalComfort:
    epw: EPW = field(init=True, repr=True)
    ground_material: _EnergyMaterialOpaqueBase = field(init=True, repr=True)
    shade_material: _EnergyMaterialOpaqueBase = field(init=True, repr=True)
    identifier: str = field(init=True, repr=False, default=None)
    model: Model = field(init=False, repr=False)

    def __post_init__(self) -> ExternalComfort:
        object.__setattr__(
            self,
            "model",
            create_model(self.ground_material, self.shade_material, self.identifier),
        )

        # Save EPW into working directory folder for posterity and reloading when necessary
        self.epw.save(
            Path(hb_folders.default_simulation_folder)
            / self.model.identifier
            / Path(self.epw.file_path).name
        )

        object.__setattr__(
            self,
            "identifier",
            self.model.identifier,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.identifier}, {self.epw}, {self.ground_material.identifier}, {self.shade_material.identifier})"

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        return {
            "epw": self.epw,
            "ground_material": self.ground_material,
            "shade_material": self.shade_material,
            "model": self.model,
        }

    def to_json(self, file_path: str) -> Path:
        """Return this object as a json file

        Returns:
            Path: The json file path.
        """

        file_path: Path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)

        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, cls=Encoder, indent=4)

        return file_path

    def results_already_exist(self) -> bool:
        # check for results and model and epw equality

        return None
