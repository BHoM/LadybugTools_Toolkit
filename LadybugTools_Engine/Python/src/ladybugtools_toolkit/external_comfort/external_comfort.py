from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW

from ..external_comfort.simulate.mean_radiant_temperature_collections import (
    mean_radiant_temperature_collections,
)
from .encoder import Encoder
from .model import create_model


class ExternalComfort:
    """An object containing the configuration for an ExternalComfortResult.

    Args:
        epw (EPW): An EPW object to be used for the mean radiant temperature simulation.
        ground_material (_EnergyMaterialOpaqueBase): A material to use for the ground surface.
        shade_material (_EnergyMaterialOpaqueBase): A material to use for any shade surface.
        identifier (str, optional): A unique identifier for this configuration. If not provided,
            then one will be created.

    Returns:
        ExternalComfort: An object containing the configuration for an ExternalComfortResult.
    """

    def __init__(
        self,
        epw: EPW,
        ground_material: _EnergyMaterialOpaqueBase,
        shade_material: _EnergyMaterialOpaqueBase,
        identifier: str = None,
    ) -> ExternalComfort:

        self.epw = epw
        self.ground_material = ground_material
        self.shade_material = shade_material
        self.identifier = identifier

        self.model = create_model(
            self.ground_material, self.shade_material, self.identifier
        )

        # get results from MRT simulation
        self._mrt_collections = mean_radiant_temperature_collections(
            self.model, self.epw
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.identifier}, {self.epw}, {self.ground_material.identifier}, {self.shade_material.identifier})"

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        return {
            **{
                "epw": self.epw,
                "ground_material": self.ground_material,
                "shade_material": self.shade_material,
                "model": self.model,
            },
            **self._mrt_collections,
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

    @property
    def shaded_below_temperature(self) -> HourlyContinuousCollection:
        return self._mrt_collections["shaded_below_temperature"]

    @property
    def shaded_above_temperature(self) -> HourlyContinuousCollection:
        return self._mrt_collections["shaded_above_temperature"]

    @property
    def shaded_direct_radiation(self) -> HourlyContinuousCollection:
        return self._mrt_collections["shaded_direct_radiation"]

    @property
    def shaded_diffuse_radiation(self) -> HourlyContinuousCollection:
        return self._mrt_collections["shaded_diffuse_radiation"]

    @property
    def shaded_longwave_mean_radiant_temperature(self) -> HourlyContinuousCollection:
        return self._mrt_collections["shaded_longwave_mean_radiant_temperature"]

    @property
    def shaded_mean_radiant_temperature(self) -> HourlyContinuousCollection:
        return self._mrt_collections["shaded_mean_radiant_temperature"]

    @property
    def unshaded_below_temperature(self) -> HourlyContinuousCollection:
        return self._mrt_collections["unshaded_below_temperature"]

    @property
    def unshaded_above_temperature(self) -> HourlyContinuousCollection:
        return self._mrt_collections["unshaded_above_temperature"]

    @property
    def unshaded_direct_radiation(self) -> HourlyContinuousCollection:
        return self._mrt_collections["unshaded_direct_radiation"]

    @property
    def unshaded_diffuse_radiation(self) -> HourlyContinuousCollection:
        return self._mrt_collections["unshaded_diffuse_radiation"]

    @property
    def unshaded_longwave_mean_radiant_temperature(self) -> HourlyContinuousCollection:
        return self._mrt_collections["unshaded_longwave_mean_radiant_temperature"]

    @property
    def unshaded_mean_radiant_temperature(self) -> HourlyContinuousCollection:
        return self._mrt_collections["unshaded_mean_radiant_temperature"]
