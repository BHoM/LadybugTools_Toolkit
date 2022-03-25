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
using System.ComponentModel;
using BH.oM.Base;

namespace BH.oM.Ladybug
{
    public class ExternalComfortResult : BHoMObject
    {
        [Description("The External Comfort object these results are associated with.")]
        public virtual ExternalComfortConfiguration ExternalComfortConfiguration { get; set; } = new ExternalComfortConfiguration();

        [Description("The surface temperature of the ground beneath the shade material.")]
        public virtual List<double> ShadedGroundSurfaceTemperature { get; set; } = new List<double>();
        [Description("The surface temperature of the material above providing shade.")]
        public virtual List<double> ShadeSurfaceTemperature { get; set; } = new List<double>();
        [Description("The direct radiation incident beneath the shade.")]
        public virtual List<double> ShadedDirectRadiation { get; set; } = new List<double>();
        [Description("The diffuse radiation incident beneath the shade.")]
        public virtual List<double> ShadedDiffuseRadiation { get; set; } = new List<double>();
        [Description("The longwave mean radiant temperature from surrounding surfaces (shade and ground).")]
        public virtual List<double> ShadedLongwaveRadiantTemperature { get; set; } = new List<double>();
        [Description("The mean radiant temperature from surrounding surfaces (shade and ground).")]
        public virtual List<double> ShadedMeanRadiantTemperature { get; set; } = new List<double>();

        [Description("The surface temperature of the ground.")]
        public virtual List<double> UnshadedGroundSurfaceTemperature { get; set; } = new List<double>();
        [Description("The \"surface temperature\" of the sky.")]
        public virtual List<double> SkyTemperature { get; set; } = new List<double>();
        [Description("The direct radiation incident in an exposed condition.")]
        public virtual List<double> UnshadedDirectRadiation { get; set; } = new List<double>();
        [Description("The diffuse radiation incident in an exposed condition.")]
        public virtual List<double> UnshadedDiffuseRadiation { get; set; } = new List<double>();
        [Description("The longwave mean radiant temperature from surrounding surfaces (sky and ground).")]
        public virtual List<double> UnshadedLongwaveRadiantTemperature { get; set; } = new List<double>();
        [Description("The mean radiant temperature from surrounding surfaces (sky, sun and ground).")]
        public virtual List<double> UnshadedMeanRadiantTemperature { get; set; } = new List<double>();
    }
}
