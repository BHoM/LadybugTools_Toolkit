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
            string name = "";
            double thickness = 0.1;
            double conductivity = 1.0;
            double density = 1000.0;
            double specificHeat = 900.0;
            Roughness roughness = Roughness.MediumRough;
            double thermalAbsorptance = 0.9;
            double solarAbsorptance = 0.7;
            double visibleAbsorptance = 0.7;

            try
            {
                name = (string)oldObject["identifier"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the identifier of the EnergyMaterial. returning name as default (\"\").\n The error: {ex}");
            }

            try
            {
                thickness = (double)oldObject["thickness"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the thickness of the EnergyMaterial. returning thickness as default ({thickness}).\n The error: {ex}");
            }

            try
            {
                conductivity = (double)oldObject["conductivity"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the conductivity of the EnergyMaterial. returning conductivity as default ({conductivity}).\n The error: {ex}");
            }

            try
            {
                density = (double)oldObject["density"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the density of the EnergyMaterial. returning density as default ({density}).\n The error: {ex}");
            }

            try
            {
                specificHeat = (double)oldObject["specific_heat"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the specific heat of the EnergyMaterial. returning specific heat as default ({specificHeat}).\n The error: {ex}");
            }

            if (Enum.TryParse((string)oldObject["roughness"], out Roughness result))
                roughness = result;
            else
                BH.Engine.Base.Compute.RecordError($"An error occurred when trying to parse the roughness of the EnergyMaterial. returning roughness as default ({roughness})");

            try
            {
                thermalAbsorptance = (double)oldObject["thermal_absorptance"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the thermal absorptance of the EnergyMaterial. returning thermal absorptance as default ({thermalAbsorptance}).\n The error: {ex}");
            }

            try
            {
                solarAbsorptance = (double)oldObject["solar_absorptance"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the solar absorptance of the EnergyMaterial. returning solar absorptance as default ({solarAbsorptance}).\n The error: {ex}");
            }

            try
            {
                visibleAbsorptance = (double)oldObject["visible_absorptance"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the visible absorptance of the EnergyMaterial. returning visible absorptance as default ({visibleAbsorptance}).\n The error: {ex}");
            }

            return new oM.LadybugTools.EnergyMaterial()
            {
                Name = name,
                Thickness = thickness,
                Conductivity = conductivity,
                Density = density,
                SpecificHeat = specificHeat,
                Roughness = roughness,
                ThermalAbsorptance = thermalAbsorptance,
                SolarAbsorptance = solarAbsorptance,
                VisibleAbsorptance = visibleAbsorptance
            };
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
