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

using System.Collections.Generic;
using BH.oM.Base;

namespace BH.oM.Ladybug
{
    public class Openfield : BHoMObject
    {
        public virtual string EPW { get; set; } = "";
        public virtual string GroundMaterial { get; set; } = "";
        public virtual string ShadeMaterial { get; set; } = "";

        public virtual List<double> ShadedGroundSurfaceTemperature { get; set; } = new List<double>();
        public virtual List<double> ShadeSurfaceTemperature { get; set; } = new List<double>();
        public virtual List<double> ShadedDirectRadiation { get; set; } = new List<double>();
        public virtual List<double> ShadedDiffuseRadiation { get; set; } = new List<double>();
        public virtual List<double> ShadedLongwaveRadiantTemperature { get; set; } = new List<double>();
        public virtual List<double> ShadedMeanRadiantTemperature { get; set; } = new List<double>();
        public virtual List<double> ShadedUniversalThermalClimateIndex { get; set; } = new List<double>();

        public virtual List<double> UnshadedGroundSurfaceTemperature { get; set; } = new List<double>();
        public virtual List<double> SkyTemperature { get; set; } = new List<double>();
        public virtual List<double> UnshadedDirectRadiation { get; set; } = new List<double>();
        public virtual List<double> UnshadedDiffuseRadiation { get; set; } = new List<double>();
        public virtual List<double> UnshadedLongwaveRadiantTemperature { get; set; } = new List<double>();
        public virtual List<double> UnshadedMeanRadiantTemperature { get; set; } = new List<double>();
        public virtual List<double> UnshadedUniversalThermalClimateIndex { get; set; } = new List<double>();
    }
}
