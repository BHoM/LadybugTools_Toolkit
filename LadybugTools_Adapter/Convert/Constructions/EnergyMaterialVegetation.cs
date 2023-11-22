/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2023, the respective contributors. All rights reserved.
 *
 * Each contributor holds copyright over their respective contributions.
 * The project versioning (Git) records all such contribution source information.
 *                                           
 *                                                                              
 * The BHoM is free software: you can redistribute it and/or modify         
 * it under the terms of the GNU Lesser General Public License as published by  
 * the Free Software Foundation, either version 3.0 of the License, or          
 * (at your option) any later version.                                          
 *                                                                              
 * The BHoM is distributed in the hope that it will be useful,              
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                 
 * GNU Lesser General Public License for more details.                          
 *                                                                            
 * You should have received a copy of the GNU Lesser General Public License     
 * along with this code. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>.      
 */

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
                try
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
                catch (Exception ex)
                {
                    BH.Engine.Base.Compute.RecordError($"An error ocurred during conversion of a {typeof(EnergyMaterialVegetation).FullName}. Returning a default {typeof(EnergyMaterialVegetation).FullName}:\n The error: {ex}");
                    return new EnergyMaterialVegetation();
                }
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
