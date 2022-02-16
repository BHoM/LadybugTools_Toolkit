from __future__ import annotations
from audioop import add
from pathlib import Path
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug.epw import EPW, HourlyContinuousCollection, AnalysisPeriod, Header
from ladybug.datatype.energyflux import Radiation

from external_comfort.model import create_model
from external_comfort.material import MATERIALS
from external_comfort.simulate import (
    _convert_radiation_to_mean_radiant_temperature,
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
        self.epw = epw
        self.ground_material = ground_material
        self.shade_material = shade_material
        self.model = create_model(self.ground_material, self.shade_material)

        self.shaded_ground_temperature: HourlyContinuousCollection = None
        self.shaded_shade_temperature: HourlyContinuousCollection = None
        self.shaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = None
        self.shaded_mean_radiant_temperature: HourlyContinuousCollection = None

        self.unshaded_ground_temperature: HourlyContinuousCollection = None
        self.unshaded_direct_radiation: HourlyContinuousCollection = None
        self.unshaded_diffuse_radiation: HourlyContinuousCollection = None
        self.unshaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = None
        self.unshaded_mean_radiant_temperature: HourlyContinuousCollection = None

    def __str__(self):
        gm = f"{self.ground_material.identifier.title().replace('_', ' ')} ground"
        sm = f"{self.shade_material.identifier.title().replace('_', ' ')} shade"
        return f"<{self.__class__.__name__}: {Path(self.epw.file_path).name}, {gm} and {sm}>"

    def __repr__(self):
        return self.__str__()

    def simulate_mean_radiant_temperature(self) -> Openfield:

        print(f"Simulating MRT for {str(self)}")
        results = {
            **_run_energyplus(self.model, self.epw),
            **_run_radiance(self.model, self.epw),
        }

        # # Assign simulation results to object
        # self.shaded_ground_temperature = results["shaded_ground_temperature"]
        # self.shaded_shade_temperature = results["shade_temperature"]
        # self.shaded_longwave_mean_radiant_temperature = (
        #     results["shaded_ground_temperature"] + results["shade_temperature"]
        # ) / 2
        # self.shaded_mean_radiant_temperature = (
        #     _convert_radiation_to_mean_radiant_temperature(
        #         self.epw,
        #         self.shaded_longwave_mean_radiant_temperature,
        #         self.shaded_direct_radiation,
        #         self.shaded_diffuse_radiation,
        #     )
        # )

        # self.unshaded_ground_temperature = results["unshaded_ground_temperature"]
        # self.unshaded_direct_radiation = results["unshaded_direct_radiation"]
        # self.unshaded_diffuse_radiation = results["unshaded_diffuse_radiation"]
        # self._unshaded_longwave_mean_radiant_temperature = (
        #     results["unshaded_ground_temperature"] + self.epw.sky_temperature
        # ) / 2
        # self.unshaded_mean_radiant_temperature = (
        #     _convert_radiation_to_mean_radiant_temperature(
        #         self.epw,
        #         self._unshaded_longwave_mean_radiant_temperature,
        #         self.unshaded_direct_radiation,
        #         self.unshaded_diffuse_radiation,
        #     )
        # )

        return results

if __name__ == "__main__":
    epw = EPW(r"C:\Users\tgerrish\BuroHappold\Sustainability and Physics - epws\FIN_SO_Alajarvi.Moksy.027870_TMYx.2004-2018.epw")
    ground_material = MATERIALS["ASPHALT"]
    shade_material = MATERIALS["FABRIC"]

    model = create_model(ground_material, shade_material)

    results = {
        **_run_energyplus(model, epw),
        **_run_radiance(model, epw),
    }