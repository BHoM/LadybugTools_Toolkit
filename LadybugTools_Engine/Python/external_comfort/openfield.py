from __future__ import annotations
import sys
sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from pathlib import Path
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug.epw import EPW, HourlyContinuousCollection

from external_comfort.model import create_model
from external_comfort.material import MATERIALS
from external_comfort.simulate import (
    _convert_radiation_to_mean_radiant_temperature,
    _radiant_temperature_from_collections,
    _run_energyplus,
    _run_radiance
)


class Openfield:
    def __init__(
        self,
        epw: EPW,
        ground_material: _EnergyMaterialOpaqueBase,
        shade_material: _EnergyMaterialOpaqueBase,
    ) -> Openfield:
        """An Openfield object containing the inputs and results for an outdoor radiant temperature simulation.

        Args:
            epw (EPW): An EPW object containing the weather data for the simulation.
            ground_material (_EnergyMaterialOpaqueBase): An EnergyMaterialOpaqueBase object for the ground.
            shade_material (_EnergyMaterialOpaqueBase): An EnergyMaterialOpaqueBase object for the shade.

        Returns:
            Openfield: An Openfield object containing the inputs and results for an outdoor radiant temperature simulation.
        """
        self.epw = epw
        self.ground_material = ground_material
        self.shade_material = shade_material
        self.model = create_model(self.ground_material, self.shade_material)

        self.shaded_ground_temperature: HourlyContinuousCollection = None
        self.shaded_shade_temperature: HourlyContinuousCollection = None
        self.shaded_direct_radiation: HourlyContinuousCollection = None
        self.shaded_diffuse_radiation: HourlyContinuousCollection = None
        self.shaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = None
        self.shaded_mean_radiant_temperature: HourlyContinuousCollection = None

        self.unshaded_ground_temperature: HourlyContinuousCollection = None
        self.unshaded_sky_temperature: HourlyContinuousCollection = self.epw.sky_temperature
        self.unshaded_direct_radiation: HourlyContinuousCollection = None
        self.unshaded_diffuse_radiation: HourlyContinuousCollection = None
        self.unshaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = None
        self.unshaded_mean_radiant_temperature: HourlyContinuousCollection = None

    def __str__(self) -> str:
        gm = f"{self.ground_material.identifier.title().replace('_', ' ')} ground"
        sm = f"{self.shade_material.identifier.title().replace('_', ' ')} shade"
        return f"<{self.__class__.__name__}: {Path(self.epw.file_path).name}, {gm} and {sm}>"

    def __repr__(self) -> str:
        return self.__str__()

    def simulate(self) -> Openfield:

        if self.shaded_mean_radiant_temperature and self.unshaded_mean_radiant_temperature:
            print(f"MRT simulation already completed for {str(self)}")
            return self

        print(f"Simulating MRT for {str(self)}")
        results = {
            **_run_energyplus(self.model, self.epw),
            **_run_radiance(self.model, self.epw),
        }

        self.shaded_ground_temperature = results["shaded_ground_temperature"]
        self.shaded_shade_temperature = results["shade_temperature"]
        self.shaded_direct_radiation = results["shaded_direct_radiation"]
        self.shaded_diffuse_radiation = results["shaded_diffuse_radiation"]
        self.shaded_longwave_mean_radiant_temperature = _radiant_temperature_from_collections(
            [results["shaded_ground_temperature"], results["shade_temperature"]],
            [0.5, 0.5]
        )
        self.shaded_mean_radiant_temperature = (
            _convert_radiation_to_mean_radiant_temperature(
                self.epw,
                self.shaded_longwave_mean_radiant_temperature,
                self.shaded_direct_radiation,
                self.shaded_diffuse_radiation,
            )
        )

        self.unshaded_ground_temperature = results["unshaded_ground_temperature"]
        self.unshaded_direct_radiation = results["unshaded_direct_radiation"]
        self.unshaded_diffuse_radiation = results["unshaded_diffuse_radiation"]
        self.unshaded_longwave_mean_radiant_temperature = _radiant_temperature_from_collections(
            [results["unshaded_ground_temperature"], self.epw.sky_temperature],
            [0.5, 0.5]
        )
        self.unshaded_mean_radiant_temperature = (
            _convert_radiation_to_mean_radiant_temperature(
                self.epw,
                self.unshaded_longwave_mean_radiant_temperature,
                self.unshaded_direct_radiation,
                self.unshaded_diffuse_radiation,
            )
        )

        return self

if __name__ == "__main__":
    epw = EPW(r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\test\GBR_London.Gatwick.037760_IWEC.epw")
    ground_material = MATERIALS["CONCRETE_LIGHTWEIGHT"]
    shade_material = MATERIALS["FABRIC"]
    openfield = Openfield(epw, ground_material, shade_material)
    openfield.simulate()
    print(openfield.unshaded_mean_radiant_temperature, openfield.shaded_mean_radiant_temperature)