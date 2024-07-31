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
    public class CollectionData : ISimulationData
    {
        [Description("The maximum value in the collection.")]
        public virtual double HighestValue { get; set; } = double.NaN;

        [Description("The minimum value in the collection")]
        public virtual double LowestValue { get; set; } = double.NaN;

        [Description("The date and time for when the maximum value occurs.")]
        public virtual DateTime HighestIndex { get; set; } = DateTime.MinValue;

        [Description("The date and time for when the minimum value occurs.")]
        public virtual DateTime LowestIndex { get; set; } = DateTime.MinValue;

        [Description("The median (50 percentile) value in the collection.")]
        public virtual double MedianValue { get; set; } = double.NaN;

        [Description("The mean value of the collection.")]
        public virtual double MeanValue { get; set; } = double.NaN;

        [Description("The mean values for each month.")]
        public virtual List<double> MonthlyMeans { get; set; } = Enumerable.Repeat<double>(double.NaN, 12).ToList();
    }
}
