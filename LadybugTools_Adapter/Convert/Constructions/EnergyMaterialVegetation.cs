using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.EnergyMaterialVegetation ToEnergyMaterialVegetation(Dictionary<string, object> oldObject)
        {
            if (Enum.TryParse((string)oldObject["roughness"], out Roughness roughness))
            {
                return new oM.LadybugTools.EnergyMaterialVegetation()
                {
                    Name = (string)oldObject["identifier"],
                    Thickness = (double)oldObject["thickness"],
                    Conductivity = (double)oldObject["conductivity"],
                    Density = (double)oldObject["density"],
                    SpecificHeat = (double)oldObject["specific_heat"],
                    Roughness = roughness,
                    SoilThermalAbsorptance = (double)oldObject["soil_thermal_absorptance"],
                    SoilSolarAbsorptance = (double)oldObject["soil_solar_absorptance"],
                    SoilVisibleAbsorptance = (double)oldObject["soil_visible_absorptance"],
                    PlantHeight = (double)oldObject["plant_height"],
                    LeafAreaIndex = (double)oldObject["leaf_area_index"],
                    LeafReflectivity = (double)oldObject["leaf_reflectivity"],
                    LeafEmissivity = (double)oldObject["leaf_emissivity"],
                    MinimumStomatalResistance = (double)oldObject["min_stomatal_resist"],
                };
            }
            else
            {
                BH.Engine.Base.Compute.RecordError("The roughness attribute could not be parsed into an enum.");
                return null;
            }
        }

        public static Dictionary<string, object> FromEnergyMaterialVegetation(BH.oM.LadybugTools.EnergyMaterialVegetation energyMaterial)
        {
            return new Dictionary<string, object>
            {
                { "type", "EnergyMaterialVegetation" },
                { "identifier", energyMaterial.Name },
                { "thickness", energyMaterial.Thickness },
                { "conductivity", energyMaterial.Conductivity },
                { "density", energyMaterial.Density },
                { "specific_heat", energyMaterial.SpecificHeat },
                { "roughness", energyMaterial.Roughness.ToString() },
                { "soil_thermal_absorptance", energyMaterial.SoilThermalAbsorptance },
                { "soil_solar_absorptance", energyMaterial.SoilSolarAbsorptance },
                { "soil_visible_absorptance", energyMaterial.SoilVisibleAbsorptance },
                { "plant_height", energyMaterial.PlantHeight },
                { "leaf_area_index", energyMaterial.LeafAreaIndex },
                { "leaf_reflectivity", energyMaterial.LeafReflectivity },
                { "leaf_emissivity",  energyMaterial.LeafEmissivity },
                { "min_stomatal_resist", energyMaterial.MinimumStomatalResistance }
            };
        }
    }
}
