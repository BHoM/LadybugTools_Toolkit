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
            string name = "";
            double thickness = 0.1;
            double conductivity = 0.35;
            double density = 1100.0;
            double specificHeat = 1200.0;
            Roughness roughness = Roughness.MediumRough;
            double soilThermalAbsorptance = 0.9;
            double soilSolarAbsorptance = 0.7;
            double soilVisibleAbsorptance = 0.7;
            double plantHeight = 0.2;
            double leafAreaIndex = 1.0;
            double leafReflectivity = 0.22;
            double leafEmissivity = 0.95;
            double minimumStomatalResistance = 180.0;

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
                soilThermalAbsorptance = (double)oldObject["soil_thermal_absorptance"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the soil thermal absorptance of the EnergyMaterial. returning soil thermal absorptance as default ({soilThermalAbsorptance}).\n The error: {ex}");
            }

            try
            {
                soilSolarAbsorptance = (double)oldObject["soil_solar_absorptance"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the soil solar absorptance of the EnergyMaterial. returning soil solar absorptance as default ({soilSolarAbsorptance}).\n The error: {ex}");
            }

            try
            {
                soilThermalAbsorptance = (double)oldObject["soil_thermal_absorptance"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the soil thermal absorptance of the EnergyMaterial. returning soil thermal absorptance as default ({soilThermalAbsorptance}).\n The error: {ex}");
            }

            try
            {
                soilVisibleAbsorptance = (double)oldObject["soil_visible_absorptance"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the soil visible absorptance of the EnergyMaterial. returning soil visible absorptance as default ({soilVisibleAbsorptance}).\n The error: {ex}");
            }

            try
            {
                plantHeight = (double)oldObject["plant_height"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the plant height of the EnergyMaterial. returning plant height as default ({plantHeight}).\n The error: {ex}");
            }

            try
            {
                leafAreaIndex = (double)oldObject["leaf_area_index"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the leaf area index of the EnergyMaterial. returning leaf area index as default ({leafAreaIndex}).\n The error: {ex}");
            }

            try
            {
                leafReflectivity = (double)oldObject["leaf_reflectivity"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the leaf reflectivity of the EnergyMaterial. returning leaf reflectivity as default ({leafReflectivity}).\n The error: {ex}");
            }

            try
            {
                leafEmissivity = (double)oldObject["leaf_emissivity"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the leaf emissivity of the EnergyMaterial. returning leaf emissivity as default ({leafEmissivity}).\n The error: {ex}");
            }

            try
            {
                minimumStomatalResistance = (double)oldObject["min_stomatal_resist"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the minimum stomatal resistance of the EnergyMaterial. returning minimum stomatal resistance as default ({minimumStomatalResistance}).\n The error: {ex}");
            }

            return new oM.LadybugTools.EnergyMaterialVegetation()
            {
                Name = name,
                Thickness = thickness,
                Conductivity = conductivity,
                Density = density,
                SpecificHeat = specificHeat,
                Roughness = roughness,
                SoilThermalAbsorptance = soilThermalAbsorptance,
                SoilSolarAbsorptance = soilSolarAbsorptance,
                SoilVisibleAbsorptance = soilVisibleAbsorptance,
                PlantHeight = plantHeight,
                LeafAreaIndex = leafAreaIndex,
                LeafReflectivity = leafReflectivity,
                LeafEmissivity = leafEmissivity,
                MinimumStomatalResistance = minimumStomatalResistance,
            };
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
