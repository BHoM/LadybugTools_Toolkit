/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2024, the respective contributors. All rights reserved.
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
        [Description("Create an EnergyMaterial object.")]
        [Input("identifier", "A unique identifier for the material. This identifier should not contain any '!' characters and should be less than 100 characters long.")]
        [Input("thickness", "The thickness of the material in meters.")]
        [Input("conductivity", "The conductivity of the material in W/m-K.")]
        [Input("density", "The density of the material in kg/m3.")]
        [Input("specificHeat", "The specific heat of the material in J/kg-K.")]
        [Input("roughness", "The roughness of the material.")]
        [Input("thermalAbsorptance", "The thermal absorptance of the material. 0-1.")]
        [Input("solarAbsorptance", "The solar absorptance of the material. 0-1.")]
        [Input("visibleAbsorptance", "The visible absorptance of the material. 0-1.")]
        [Output("energyMaterial", "An EnergyMaterial object.")]
        public static EnergyMaterial EnergyMaterial(
            string identifier,
            double thickness,
            double conductivity,
            double density,
            double specificHeat,
            Roughness roughness = Roughness.MediumRough,
            double thermalAbsorptance = 0.9,
            double solarAbsorptance = 0.7,
            double visibleAbsorptance = 0.7
        )
        {
            if (identifier.Contains("!"))
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(identifier)} cannot contain '!' character.");
                return null;
            }

            if (identifier.Length > 100)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(identifier)} cannot be longer than 100 characters.");
                return null;
            }

            if (specificHeat < 100)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(specificHeat)} must be greater than 100.");
                return null;
            }

            if (conductivity < 0)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(conductivity)} must be greater than 0.");
                return null;
            }

            if (density < 0)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(density)} must be greater than 0.");
                return null;
            }

            if (thickness <= 0)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(thickness)} must be greater than 0.");
                return null;
            }

            if (roughness == Roughness.Undefined)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(roughness)} must be defined.");
                return null;
            }

            if (thermalAbsorptance < 0 || thermalAbsorptance > 1)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(thermalAbsorptance)} must be between 0 and 1.");
                return null;
            }

            if (solarAbsorptance < 0 || solarAbsorptance > 1)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(solarAbsorptance)} must be between 0 and 1.");
                return null;
            }

            if (visibleAbsorptance < 0 || visibleAbsorptance > 1)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(visibleAbsorptance)} must be between 0 and 1.");
                return null;
            }

            return new oM.LadybugTools.EnergyMaterial()
            {
                Name = identifier,
                Thickness = thickness,
                Conductivity = conductivity,
                Density = density,
                SpecificHeat = specificHeat,
                Roughness = roughness,
                ThermalAbsorptance = thermalAbsorptance,
                SolarAbsorptance = solarAbsorptance,
                VisibleAbsorptance = visibleAbsorptance,
            };
        }
    }
}

