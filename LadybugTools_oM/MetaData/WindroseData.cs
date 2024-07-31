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
using System.Linq;
using System.Text;

namespace BH.oM.LadybugTools
{
    [NoAutoConstructor]
    public class WindroseData : ISimulationData
    {
        [Description("The direction that the prevailing wind is coming from between two angles as a tuple, where 0 degrees is north.")]
        public virtual List<double> PrevailingDirection { get; set; } = Enumerable.Repeat<double>(double.NaN, 2).ToList();

        [Description("The 95 percentile wind speed value in the prevailing direction.")]
        public virtual double PrevailingPercentile95 { get; set; } = double.NaN;

        [Description("The median (50 percentile) wind speed value in the prevailing direction.")]
        public virtual double PrevailingPercentile50 { get; set; } = double.NaN;

        [Description("The 95 percentile wind speed value.")]
        public virtual double Percentile95 { get; set; } = double.NaN;

        [Description("The median (50 percentile) wind speed value.")]
        public virtual double Percentile50 { get; set; } = double.NaN;

        [Description("The percentage of calm hours / total hours.")]
        public virtual double PercentageOfCalmHours { get; set; } = double.NaN;
    }
}
