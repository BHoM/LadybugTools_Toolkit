from __future__ import annotations
import inspect
import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug.epw import EPW, HourlyContinuousCollection

from external_comfort.model import create_model
from external_comfort.material import MATERIALS
from external_comfort.simulate import (
    _convert_radiation_to_mean_radiant_temperature,
    _radiant_temperature_from_collections,
    _run_energyplus,
    _run_radiance,
)


class Openfield:
    def __init__(
        self,
        epw: EPW,
        ground_material: _EnergyMaterialOpaqueBase,
        shade_material: _EnergyMaterialOpaqueBase,
        run: bool = False
    ) -> Openfield:
        """An Openfield object containing the inputs and results for an outdoor radiant temperature simulation.

        Args:
            epw (EPW): An EPW object containing the weather data for the simulation.
            ground_material (_EnergyMaterialOpaqueBase): An EnergyMaterialOpaqueBase object for the ground.
            shade_material (_EnergyMaterialOpaqueBase): An EnergyMaterialOpaqueBase object for the shade.
            run (bool, optional): A boolean to run the simulation upon creation. Defaults to False.

        Returns:
            Openfield: An Openfield object containing the inputs and results for an outdoor radiant temperature simulation.
        """
        self.epw = epw
        self.ground_material = ground_material
        self.shade_material = shade_material
        self.model = create_model(self.ground_material, self.shade_material, "testing")

        self._shaded_below_temperature: HourlyContinuousCollection = None
        self._shaded_above_temperature: HourlyContinuousCollection = None
        self._shaded_direct_radiation: HourlyContinuousCollection = None
        self._shaded_diffuse_radiation: HourlyContinuousCollection = None
        self._shaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = (
            None
        )
        self._shaded_mean_radiant_temperature: HourlyContinuousCollection = None

        self._unshaded_below_temperature: HourlyContinuousCollection = None
        self._unshaded_above_temperature: HourlyContinuousCollection = (
            self.epw.sky_temperature
        )
        self._unshaded_direct_radiation: HourlyContinuousCollection = None
        self._unshaded_diffuse_radiation: HourlyContinuousCollection = None
        self._unshaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = (
            None
        )
        self._unshaded_mean_radiant_temperature: HourlyContinuousCollection = None

        if run:
            self.__simulate()

    @property
    def shaded_below_temperature(self) -> HourlyContinuousCollection:
        if not self._shaded_below_temperature:
            self.__run_energyplus()
        return self._shaded_below_temperature

    @property
    def shaded_above_temperature(self) -> HourlyContinuousCollection:
        if not self._shaded_above_temperature:
            self.__run_energyplus()
        return self._shaded_above_temperature

    @property
    def unshaded_below_temperature(self) -> HourlyContinuousCollection:
        if not self._unshaded_below_temperature:
            self.__run_energyplus()
        return self._unshaded_below_temperature

    @property
    def unshaded_above_temperature(self) -> HourlyContinuousCollection:
        if not self._unshaded_above_temperature:
            self.__run_energyplus()
        return self._unshaded_above_temperature

    @property
    def shaded_direct_radiation(self) -> HourlyContinuousCollection:
        if not self._shaded_direct_radiation:
            self.__run_radiance()
        return self._shaded_direct_radiation

    @property
    def shaded_diffuse_radiation(self) -> HourlyContinuousCollection:
        if not self._shaded_diffuse_radiation:
            self.__run_radiance()
        return self._shaded_diffuse_radiation

    @property
    def unshaded_direct_radiation(self) -> HourlyContinuousCollection:
        if not self._unshaded_direct_radiation:
            self.__run_radiance()
        return self._unshaded_direct_radiation

    @property
    def unshaded_diffuse_radiation(self) -> HourlyContinuousCollection:
        if not self._unshaded_diffuse_radiation:
            self.__run_radiance()
        return self._unshaded_diffuse_radiation

    @property
    def shaded_longwave_mean_radiant_temperature(self) -> HourlyContinuousCollection:
        if not self._shaded_longwave_mean_radiant_temperature:
            self._shaded_longwave_mean_radiant_temperature = (
                _radiant_temperature_from_collections(
                    [self.shaded_below_temperature, self.shaded_above_temperature],
                    [0.5, 0.5],
                )
            )
        return self._shaded_longwave_mean_radiant_temperature

    @property
    def unshaded_longwave_mean_radiant_temperature(self) -> HourlyContinuousCollection:
        if not self._unshaded_longwave_mean_radiant_temperature:
            self._unshaded_longwave_mean_radiant_temperature = (
                _radiant_temperature_from_collections(
                    [self.unshaded_below_temperature, self.unshaded_above_temperature],
                    [0.5, 0.5],
                )
            )
        return self._unshaded_longwave_mean_radiant_temperature

    @property
    def shaded_mean_radiant_temperature(self) -> HourlyContinuousCollection:
        if not self._shaded_mean_radiant_temperature:
            self._shaded_mean_radiant_temperature = (
                _convert_radiation_to_mean_radiant_temperature(
                    self.epw,
                    self.shaded_longwave_mean_radiant_temperature,
                    self.shaded_direct_radiation,
                    self.shaded_diffuse_radiation,
                )
            )
        return self._shaded_mean_radiant_temperature

    @property
    def unshaded_mean_radiant_temperature(self) -> HourlyContinuousCollection:
        if not self._unshaded_mean_radiant_temperature:
            self._unshaded_mean_radiant_temperature = (
                _convert_radiation_to_mean_radiant_temperature(
                    self.epw,
                    self.unshaded_longwave_mean_radiant_temperature,
                    self.unshaded_direct_radiation,
                    self.unshaded_diffuse_radiation,
                )
            )
        return self._unshaded_mean_radiant_temperature

    def __str__(self) -> str:
        gm = f"{self.ground_material.identifier.title().replace('_', ' ')} ground"
        sm = f"{self.shade_material.identifier.title().replace('_', ' ')} shade"
        return f"<{self.__class__.__name__}: {Path(self.epw.file_path).name}, {gm} and {sm}>"

    def __repr__(self) -> str:
        return self.__str__()

    def __run_energyplus(self) -> Openfield:
        for prop in [
            "_shaded_below_temperature",
            "_unshaded_below_temperature",
            "_shaded_above_temperature",
            "_unshaded_above_temperature",
        ]:
            if not isinstance(getattr(self, prop), HourlyContinuousCollection):
                for k, v in _run_energyplus(self.model, self.epw).items():
                    setattr(self, f"_{k}", v)
        return self

    def __run_radiance(self) -> Openfield:
        for prop in [
            "_shaded_direct_radiation",
            "_unshaded_direct_radiation",
            "_shaded_diffuse_radiation",
            "_unshaded_diffuse_radiation",
        ]:
            if not isinstance(getattr(self, prop), HourlyContinuousCollection):
                for k, v in _run_radiance(self.model, self.epw).items():
                    setattr(self, f"_{k}", v)
        return self

    def __simulate(self) -> Openfield:
        """Run both EnergyPlus and Radiance to return contextual radiant-temperature data collections."""

        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            for f in [_run_radiance, _run_energyplus]:
                results.append(executor.submit(f, self.model, self.epw))
            
        for x in results:
            for k, v in x.result().items():
                setattr(self, f"_{k}", v)
        return self

if __name__ == "__main__":

    epw = EPW(
        r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\test\GBR_London.Gatwick.037760_IWEC.epw"
    )
    ground_material = MATERIALS["CONCRETE_LIGHTWEIGHT"]
    shade_material = MATERIALS["FABRIC"]

    openfield = Openfield(epw, ground_material, shade_material, True)

    print(openfield.unshaded_mean_radiant_temperature, openfield.shaded_mean_radiant_temperature)
