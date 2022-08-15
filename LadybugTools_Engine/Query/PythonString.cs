/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2022, the respective contributors. All rights reserved.
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
using BH.oM.Python;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Get the python code string representation of an ExternalComfortTypology object.")]
        [Input("typology", "An External Comfort Typology object.")]
        [Output("pythonString", "The python code string representation of an ExternalComfortTypology object.")]
        public static string PythonString(this ExternalComfortTypology typology)
        {
            List<string> shelters = new List<string>();
            foreach (ExternalComfortShelter shelter in typology.Shelters)
            {
                shelters.Add(shelter.PythonString());
            }
            return $"Typology(name='{typology.Name}', shelters=[{String.Join(", ", shelters)}], evaporative_cooling_effectiveness={typology.EvaporativeCoolingEffectiveness})";
        }

        [Description("Get the python code string representation of an ExternalComfortShelter object.")]
        [Input("typology", "An External Comfort Shelter object.")]
        [Output("pythonString", "The python code string representation of an ExternalComfortShelter object.")]
        public static string PythonString(this ExternalComfortShelter shelter)
        {
            return $"Shelter(porosity={shelter.Porosity}, azimuth_range=({shelter.StartAzimuth}, {shelter.EndAzimuth}), altitude_range=({shelter.StartAltitude}, {shelter.EndAltitude}))";
        }

        [Description("Get the python code string representation of an ExternalComfortMaterial object.")]
        [Input("typology", "An External Comfort Material object.")]
        [Output("pythonString", "The python code string representation of an ExternalComfortMaterial object.")]
        public static string PythonString(this ExternalComfortMaterial material)
        {
            if (new List<double>() { 
                material.Conductivity, 
                material.Density, 
                material.ThermalAbsorptance, 
                material.SolarAbsorptance, 
                material.VisibleAbsorptance, 
                material.SpecificHeat, 
                material.Thickness 
            }.Contains(double.NaN) || material.Roughness == Roughness.Undefined )
            {
                BH.Engine.Base.Compute.RecordError("The ExternalComfortMaterial created contains null values that are not possible to simulate.");
            }
            return $"EnergyMaterial(identifier='{material.Identifier}', roughness='{material.Roughness}', thickness={material.Thickness}, conductivity={material.Conductivity}, density={material.Density}, specific_heat={material.SpecificHeat}, thermal_absorptance={material.ThermalAbsorptance}, solar_absorptance={material.SolarAbsorptance}, visible_absorptance={material.VisibleAbsorptance})";
        }
    }
}

