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
using System.Linq;
using System.Text;

namespace BH.oM.LadybugTools
{
    [NoAutoConstructor]
    public class WindroseData : ISimulationData
    {
        [DisplayText("Prevailing Direction")]
        [Description("The direction bin of the prevailing wind, defined as two values (in degrees) for the lower and upper values for the bin, where 0 degrees is north.")]
        public virtual List<double> PrevailingDirection { get; set; } = Enumerable.Repeat<double>(double.NaN, 2).ToList();

        [DisplayText("Prevailing 95th Percentile")]
        [Description("The 95th percentile wind speed value in the prevailing direction.")]
        public virtual double PrevailingPercentile95 { get; set; } = double.NaN;

        [DisplayText("Prevailing 50th Percentile")]
        [Description("The median (50th percentile) wind speed value in the prevailing direction.")]
        public virtual double PrevailingPercentile50 { get; set; } = double.NaN;

        [DisplayText("95th Percentile")]
        [Description("The 95th percentile wind speed value.")]
        public virtual double Percentile95 { get; set; } = double.NaN;

        [DisplayText("50th Percentile")]
        [Description("The median (50th percentile) wind speed value.")]
        public virtual double Percentile50 { get; set; } = double.NaN;

        [DisplayText("Ratio Of Calm Hours")]
        [Description("The ratio of calm hours to total hours. Calm hours are hours with a wind speed of 1e-10 or less.")]
        public virtual double RatioOfCalmHours { get; set; } = double.NaN;
    }
}

