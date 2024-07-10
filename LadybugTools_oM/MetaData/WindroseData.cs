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
using System.Linq;
using System.Text;

namespace BH.oM.LadybugTools
{
    [NoAutoConstructor]
    public class WindroseData : ISimulationData
    {
        public virtual List<double> PrevailingDirection { get; set; } = Enumerable.Repeat<double>(double.NaN, 2).ToList();

        public virtual double PrevailingPercentile95 { get; set; } = double.NaN;

        public virtual double PrevailingPercentile50 { get; set; } = double.NaN;

        public virtual double Percentile95 { get; set; } = double.NaN; 

        public virtual double Percentile50 { get; set; } = double.NaN;

        public virtual double PercentageOfCalmHours { get; set; } = double.NaN;
    }
}
