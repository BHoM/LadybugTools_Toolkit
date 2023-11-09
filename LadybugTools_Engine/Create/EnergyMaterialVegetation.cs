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

using BH.oM.Base.Attributes;
using BH.oM.LadybugTools;
using System.ComponentModel;

namespace BH.Engine.LadybugTools
{
    public static partial class Create
    {
        [Description("Create an EnergyMaterialVegetation object.")]
        [Input("identifier", "The identifier for this material object.")]
        [Input("thickness", "The thickness of the material.")]
        [Input("conductivity", "The conductivity of the material.")]
        [Input("density", "The density of the material.")]
        [Input("specificHeat", "The specific heat of the material.")]
        [Input("roughness", "The roughness of the material.")]
        [Input("plantHeight", "The height of the vegetation in meters.")]
        [Input("leafAreaIndex", "The leaf area index of the vegetation.")]
        [Input("leafReflectivity", "The reflectivity of the leaves.")]
        [Input("leafEmissivity", "The emissivity of the leaves.")]
        [Input("minStomatalResist", "The minimum stomatal resistance of the leaves.")]
        [Input("soilThermalAbsorptance", "The thermal absorptance of the soil.")]
        [Input("soilSolarAbsorptance", "The solar absorptance of the soil.")]
        [Input("soilVisibleAbsorptance", "The visible absorptance of the soil.")]
        [Output("energyMaterialVegetation", "An EnergyMaterialVegetation object.")]
        public static EnergyMaterialVegetation EnergyMaterialVegetation(
            string identifier,
            double thickness = 0.1,
            double conductivity = 0.35,
            double density = 1100,
            double specificHeat = 1200,
            Roughness roughness = Roughness.MediumRough,
            double plantHeight = 0.2,
            double leafAreaIndex = 1.0,
            double leafReflectivity = 0.22,
            double leafEmissivity = 0.95,
            double minStomatalResist = 180,
            double soilThermalAbsorptance = 0.9,
            double soilSolarAbsorptance = 0.7,
            double soilVisibleAbsorptance = 0.7
        )
        {
            if (identifier.Contains("!"))
            {
                BH.Engine.Base.Compute.RecordError("identifier cannot contain '!' character");
                return null;
            }

            if (identifier.Length > 100)
            {
                BH.Engine.Base.Compute.RecordError("identifier cannot be longer than 100 characters");
                return null;
            }

            if (specificHeat < 100)
            {
                BH.Engine.Base.Compute.RecordError("specificHeat must be greater than 100");
                return null;
            }

            if (conductivity < 0)
            {
                BH.Engine.Base.Compute.RecordError("conductivity must be greater than 0");
                return null;
            }

            if (density < 0)
            {
                BH.Engine.Base.Compute.RecordError("density must be greater than 0");
                return null;
            }

            if (thickness <= 0)
            {
                BH.Engine.Base.Compute.RecordError("thickness must be greater than 0");
                return null;
            }

            if (roughness == Roughness.Undefined)
            {
                BH.Engine.Base.Compute.RecordError("roughness must be defined");
                return null;
            }

            if (soilThermalAbsorptance < 0 || soilThermalAbsorptance > 1 || soilSolarAbsorptance < 0 || soilSolarAbsorptance > 1 || soilVisibleAbsorptance < 0 || soilVisibleAbsorptance > 1)
            {
                BH.Engine.Base.Compute.RecordError("soilThermalAbsorptance, soilSolarAbsorptance, and soilVisibleAbsorptance must be between 0 and 1");
                return null;
            }

            if (plantHeight < 0.005 || plantHeight > 1)
            {
                BH.Engine.Base.Compute.RecordError("plantHeight must be between 0.005 and 1");
                return null;
            }

            if (leafAreaIndex < 0.001 || leafAreaIndex > 5)
            {
                BH.Engine.Base.Compute.RecordError("leafAreaIndex must be between 0.001 and 5");
                return null;
            }

            if (leafEmissivity < 0.8 || leafEmissivity > 1)
            {
                BH.Engine.Base.Compute.RecordError("leafEmissivity must be between 0.8 and 1");
                return null;
            }

            if (leafReflectivity < 0.05 || leafReflectivity > 0.5)
            {
                BH.Engine.Base.Compute.RecordError("leafReflectivity must be between 0.05 and 0.5");
                return null;
            }

            if (minStomatalResist < 50 || minStomatalResist > 300)
            {
                BH.Engine.Base.Compute.RecordError("minStomatalResist must be between 50 and 300");
                return null;
            }

            return new oM.LadybugTools.EnergyMaterialVegetation()
            {
                Identifier = identifier,
                Thickness = thickness,
                Conductivity = conductivity,
                Density = density,
                SpecificHeat = specificHeat,
                Roughness = roughness,
                PlantHeight = plantHeight,
                LeafAreaIndex = leafAreaIndex,
                LeafReflectivity = leafReflectivity,
                LeafEmissivity = leafEmissivity,
                SoilThermalAbsorptance = soilThermalAbsorptance,
                SoilSolarAbsorptance = soilSolarAbsorptance,
                SoilVisibleAbsorptance = soilVisibleAbsorptance,
                MinStomatalResist = minStomatalResist,
            };
        }
    }
}


