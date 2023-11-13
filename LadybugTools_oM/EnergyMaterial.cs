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


using System.ComponentModel;
using BH.oM.Base.Attributes;

namespace BH.oM.LadybugTools
{
    [NoAutoConstructor]
    public class EnergyMaterial : IEnergyMaterialOpaque
    {
        [Description("The object type - for use Python-side")]
        public virtual string Type { get; set; } = "EnergyMaterial";

        [Description("The name of this EnergyMaterial.")]
        public virtual string Identifier { get; set; }

        [Description("Thickness of material (m).")]
        [Length]
        public virtual double Thickness { get; set; }

        [Description("Conductivity of material (W/mK).")]
        public virtual double Conductivity { get; set; }

        [Description("Density of material (kg/m3).")]
        public virtual double Density { get; set; }

        [Description("Specific heat capacity of material (J/kgK).")]
        public virtual double SpecificHeat { get; set; }

        [Description("The roughness of the material.")]
        public virtual Roughness Roughness { get; set; }

        [Description("Thermal absorptivity (emissivity) of material (0-1).")]
        public virtual double ThermalAbsorptance { get; set; }

        [Description("Solar absorptivity of material (0-1).")]
        public virtual double SolarAbsorptance { get; set; }

        [Description("Light absorptivity (1 - albedo) of material (0-1).")]
        public virtual double VisibleAbsorptance { get; set; }
    }
}
