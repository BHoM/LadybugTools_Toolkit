using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.EnergyMaterial EnergyMaterial(BH.Adapter.LadybugTools.EnergyMaterial oldObject)
        {
            return new oM.LadybugTools.EnergyMaterial()
            {
                Name = oldObject.Identifier,
                Thickness = oldObject.Thickness,
                Conductivity = oldObject.Conductivity,
                Density = oldObject.Density,
                SpecificHeat = oldObject.SpecificHeat,
                Roughness = oldObject.Roughness,
                ThermalAbsorptance = oldObject.ThermalAbsorptance,
                SolarAbsorptance = oldObject.SolarAbsorptance,
                VisibleAbsorptance = oldObject.VisibleAbsorptance
            };
        }
    }
}
