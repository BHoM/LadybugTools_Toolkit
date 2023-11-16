using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.EnergyMaterialVegetation EnergyMaterialVegetation(BH.Adapter.LadybugTools.EnergyMaterialVegetation oldObject)
        {
            return new oM.LadybugTools.EnergyMaterialVegetation()
            {
                Name = oldObject.Identifier,
                Thickness = oldObject.Thickness,
                Conductivity = oldObject.Conductivity,
                Density = oldObject.Density,
                SpecificHeat = oldObject.SpecificHeat,
                Roughness = oldObject.Roughness,
                SoilThermalAbsorptance = oldObject.SoilThermalAbsorptance,
                SoilSolarAbsorptance = oldObject.SoilSolarAbsorptance,
                SoilVisibleAbsorptance = oldObject.SoilVisibleAbsorptance,
                PlantHeight = oldObject.PlantHeight,
                LeafAreaIndex = oldObject.LeafAreaIndex,
                LeafReflectivity = oldObject.LeafReflectivity,
                LeafEmissivity = oldObject.LeafEmissivity,
                MinimumStomatalResistance = oldObject.MinStomatalResist,
            };
        }
    }
}
