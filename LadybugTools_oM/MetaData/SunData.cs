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
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace BH.oM.LadybugTools
{
    [NoAutoConstructor]
    public class SunData
    {
        [Description("The azimuth angle at sunrise.")]
        public virtual double SunriseAzimuth { get; set; } = double.NaN;

        [Description("The time that sunrise happens.")]
        public virtual DateTime SunriseTime { get; set; } = DateTime.MinValue;

        [Description("The altitude angle at solar noon (when the sun is at its highest point).")]
        public virtual double NoonAltitude { get; set; } = double.NaN;

        [Description("The time that the altitude is highest.")]
        public virtual DateTime NoonTime { get; set; } = DateTime.MinValue;

        [Description("The azimuth angle at sunset.")]
        public virtual double SunsetAzimuth { get; set; } = double.NaN;

        [Description("The time that sunset happens.")]
        public virtual DateTime SunsetTime { get; set; } = DateTime.MinValue;
    }
}
