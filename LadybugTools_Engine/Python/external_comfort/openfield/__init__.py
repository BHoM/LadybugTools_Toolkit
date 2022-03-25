from __future__ import annotations

import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

from external_comfort.material.material_from_string import material_from_string
from external_comfort.model.create_model import create_model
from external_comfort.simulate.energyplus import energyplus
from external_comfort.simulate.mean_radiant_temperature import mean_radiant_temperature
from external_comfort.simulate.radiance import radiance
from external_comfort.simulate.radiant_temperature_from_collections import (
    radiant_temperature_from_collections,
)
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug.epw import EPW, HourlyContinuousCollection


class Openfield:
    def __init__(
        self,
        epw: EPW,
        ground_material: Union[str, _EnergyMaterialOpaqueBase],
        shade_material: Union[str, _EnergyMaterialOpaqueBase],
    ) -> Openfield:
        """An Openfield object containing the inputs necessary for an outdoor radiant temperature simulation.

        Args:
            epw (EPW): An EPW object containing the weather data for the simulation.
            ground_material (_EnergyMaterialOpaqueBase): An EnergyMaterialOpaqueBase object for the ground.
            shade_material (_EnergyMaterialOpaqueBase): An EnergyMaterialOpaqueBase object for the shade.

        Returns:
            Openfield: An Openfield object containing the inputs for an outdoor radiant temperature simulation.
        """
        self.epw = epw if isinstance(epw, EPW) else EPW(epw)
        self.ground_material = (
            ground_material
            if isinstance(ground_material, _EnergyMaterialOpaqueBase)
            else material_from_string(ground_material)
        )
        self.shade_material = (
            shade_material
            if isinstance(shade_material, _EnergyMaterialOpaqueBase)
            else material_from_string(shade_material)
        )
        self.model = create_model(self.ground_material, self.shade_material)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(epw={Path(self.epw.file_path).name}, ground_material={self.ground_material.identifier}, shade_material={self.shade_material.identifier})"


class OpenfieldResult:
    def __init__(self, openfield: Openfield) -> OpenfieldResult:
        """An object containing the results of an outdoor radiant temperature simulation.

        Args:
            openfield (Openfield): An Openfield object containing the inputs for an outdoor radiant temperature simulation.

        Returns:
            OpenfieldResult: An object containing results of an outdoor radiant temperature simulation.
        """
        self.openfield = openfield

        self.shaded_below_temperature: HourlyContinuousCollection = None
        self.shaded_above_temperature: HourlyContinuousCollection = None
        self.shaded_direct_radiation: HourlyContinuousCollection = None
        self.shaded_diffuse_radiation: HourlyContinuousCollection = None
        self.shaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = None
        self.shaded_mean_radiant_temperature: HourlyContinuousCollection = None

        self.unshaded_below_temperature: HourlyContinuousCollection = None
        self.unshaded_above_temperature: HourlyContinuousCollection = (
            self.openfield.epw.sky_temperature
        )
        self.unshaded_direct_radiation: HourlyContinuousCollection = None
        self.unshaded_diffuse_radiation: HourlyContinuousCollection = None
        self.unshaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = (
            None
        )
        self.unshaded_mean_radiant_temperature: HourlyContinuousCollection = None

        # Run EnergyPlus and Radiance simulations
        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            for f in [radiance, energyplus]:
                results.append(executor.submit(f, self.openfield.model, self.openfield.epw))
        
        # Populate simulation results
        for x in results:
            for k, v in x.result().items():
                setattr(self, k, v)

        # Populate calculated results
        self.shaded_longwave_mean_radiant_temperature = (
            radiant_temperature_from_collections(
                [
                    self.shaded_below_temperature,
                    self.shaded_above_temperature,
                ],
                [0.5, 0.5],
            )
        )

        self.unshaded_longwave_mean_radiant_temperature = (
            radiant_temperature_from_collections(
                [
                    self.unshaded_below_temperature,
                    self.unshaded_above_temperature,
                ],
                [0.5, 0.5],
            )
        )

        self.shaded_mean_radiant_temperature = mean_radiant_temperature(
            self.openfield.epw,
            self.shaded_longwave_mean_radiant_temperature,
            self.shaded_direct_radiation,
            self.shaded_diffuse_radiation,
        )

        self.unshaded_mean_radiant_temperature = mean_radiant_temperature(
            self.openfield.epw,
            self.unshaded_longwave_mean_radiant_temperature,
            self.unshaded_direct_radiation,
            self.unshaded_diffuse_radiation,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(epw={Path(self.openfield.epw.file_path).name}, ground_material={self.openfield.ground_material.identifier}, shade_material={self.openfield.shade_material.identifier})"
