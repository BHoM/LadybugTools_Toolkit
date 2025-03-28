/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2025, the respective contributors. All rights reserved.
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
    public class SolarRadiationData: ISimulationData
    {
        [DisplayText("Maximum Value")]
        [Description("The maximum incoming solar radiation.")]
        public double MaxValue { get; set; } = double.NaN;

        [DisplayText("Minimum Value")]
        [Description("The minimum incoming solar radiation.")]
        public double MinValue { get; set; } = double.NaN;

        [DisplayText("Maximum Direction")]
        [Description("The direction, in degrees(°) clockwise from north that the maximum incoming solar radiation is coming from.")]
        public double MaxDirection { get; set; } = double.NaN;

        [DisplayText("Minimum Direction")]
        [Description("The direction, in degrees(°) clockwise from north that the minimum incoming solar radiation is coming from.")]
        public double MinDirection { get; set; } = double.NaN;

        [DisplayText("Maximum Tilt")]
        [Description("The angle, in degrees(°) above the horizon that the maximum incoming solar radiation is coming from.")]
        public double MaxTilt { get; set; } = double.NaN;

        [DisplayText("Minimum Tilt")]
        [Description("The angle, in degrees(°) above the horizon that the minimum incoming solar radiation is coming from.")]
        public double MinTilt { get; set;} = double.NaN;
    }
}
