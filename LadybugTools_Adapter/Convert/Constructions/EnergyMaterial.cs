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
        public static BH.oM.LadybugTools.EnergyMaterial ToEnergyMaterial(Dictionary<string, object> oldObject)
        {
            if (Enum.TryParse((string)oldObject["roughness"], out Roughness roughness))
            {
                try
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
                catch (Exception ex)
                {
                    BH.Engine.Base.Compute.RecordError($"An error ocurred during conversion of a {typeof(EnergyMaterial).FullName}. Returning a default {typeof(EnergyMaterial).FullName}:\n The error: {ex}");
                    return new EnergyMaterial();
                }
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
                { "type", "EnergyMaterial" },
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
