using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.EnergyMaterial ToEnergyMaterial(Dictionary<string, object> oldObject)
        {
            if (Enum.TryParse((string)oldObject["roughness"], out Roughness roughness))
            {
                return new oM.LadybugTools.EnergyMaterial()
                {
                    Name = (string)oldObject["identifier"],
                    Thickness = (double)oldObject["thickness"],
                    Conductivity = (double)oldObject["conductivity"],
                    Density = (double)oldObject["density"],
                    SpecificHeat = (double)oldObject["specific_heat"],
                    Roughness = roughness,
                    ThermalAbsorptance = (double)oldObject["thermal_absorptance"],
                    SolarAbsorptance = (double)oldObject["solar_absorptance"],
                    VisibleAbsorptance = (double)oldObject["visible_absorptance"]
                };
            }
            else
            {
                BH.Engine.Base.Compute.RecordError("The roughness attribute could not be parsed into an enum.");
                return null;
            }
        }

        public static Dictionary<string, object> FromEnergyMaterial(BH.oM.LadybugTools.EnergyMaterial energyMaterial)
        {
            return new Dictionary<string, object>()
            {
                { "identifier", energyMaterial.Name },
                { "thickness", energyMaterial.Thickness },
                { "conductivity", energyMaterial.Conductivity },
                { "density", energyMaterial.Density },
                { "specific_heat", energyMaterial.SpecificHeat },
                { "roughness", energyMaterial.Roughness.ToString() },
                { "thermal_absorptance", energyMaterial.ThermalAbsorptance },
                { "solar_absorptance", energyMaterial.SolarAbsorptance },
                { "visible_absorptance", energyMaterial.VisibleAbsorptance }
            };
        }
    }
}
