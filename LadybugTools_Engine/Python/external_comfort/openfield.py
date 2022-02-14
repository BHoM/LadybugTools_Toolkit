from __future__ import annotations
from pathlib import Path
from typing import Dict
from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
from ladybug.epw import EPW, HourlyContinuousCollection
from external_comfort.ground_temperature import energyplus_ground_temperature_strings

from external_comfort.model import create_model
from external_comfort.simulate import _convert_radiation_to_mean_radiant_temperature, _run_energyplus, _run_radiance

class Openfield:

    def __init__(
        self, epw: EPW, ground_material: _EnergyMaterialOpaqueBase, shade_material: _EnergyMaterialOpaqueBase
    ) -> Openfield:
        self.epw = epw
        self.ground_material = ground_material
        self.shade_material = shade_material
        self.model = create_model(self.ground_material, self.shade_material)

        self.results = {"shaded": [], "unshaded": []}

        def __str__(self):
            gm = f"{self.ground_material.identifier.title().replace('_', ' ')} ground"
            sm = f"{self.shade_material.identifier.title().replace('_', ' ')} shade"
            return f"<{self.__class__.__name__}: {Path(self.epw.file_path).name}, {gm} and {sm}>"

        def __repr__(self):
            return self.__str__()

        def simulate_mean_radiant_temperature(self) -> Dict[str, HourlyContinuousCollection]:

            # Check that MRT doesn't already exist for the given configuration
            mrt_unshaded = None
            mrt_shaded = None

            for i in self.simulation_results["unshaded"]:
                if self.ground_material.identifier == i["ground_material"]:
                    mrt_unshaded = i["mean_radiant_temperature"]

            for i in self.simulation_results["shaded"]:
                if (self.ground_material.identifier == i["ground_material"]) and (
                    self.shade_material.identifier == i["shade_material"]
                ):
                    mrt_shaded = i["mean_radiant_temperature"]

            if (mrt_unshaded is not None) and (mrt_shaded is not None):
                print(f"Using existing MRT for {str(self)}")
                return self
            
            print(f"Simulating MRT for {str(self)}")
            additional_strings = energyplus_ground_temperature_strings(self.epw)
            results = {**_run_energyplus(self.model, self.epw, additional_strings), **_run_radiance(self.model, self.epw)}

            # TODO - sort these methods out to make them less obfuscated! And so that descriptiors are as good as they can be!
            base_surface_temp_unshaded = (results["unshaded_ground_temperature"] + epw.sky_temperature) / 2
            base_surface_temp_shaded = (results["shaded_ground_temperature"] + results["shade_temperature"]) / 2

            mrt_unshaded = _convert_radiation_to_mean_radiant_temperature(
                self.epw, base_surface_temp_unshaded, results["unshaded_direct_radiation"], results["unshaded_diffuse_radiation"]
            )
            mrt_shaded = _convert_radiation_to_mean_radiant_temperature(
                self.epw, base_surface_temp_shaded, results["shaded_direct_radiation"], results["shaded_diffuse_radiation"]
            )
