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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class CollectionData : ISimulationData
    {
        public virtual double HighestValue { get; set; } = double.NaN;

        public virtual double LowestValue { get; set; } = double.NaN;

        public virtual DateTime HighestIndex { get; set; } = DateTime.MinValue;

        public virtual DateTime LowestIndex { get; set; } = DateTime.MinValue;

        public virtual double MedianValue { get; set; } = double.NaN;

        public virtual double MeanValue { get; set; } = double.NaN;

        public virtual List<double> MonthlyMeans { get; set; } = Enumerable.Repeat<double>(double.NaN, 12).ToList();

        public virtual List<List<double>> MonthlyDiurnalRanges { get; set; } = Enumerable.Repeat<List<double>>(new List<double> { double.NaN, double.NaN }, 12).ToList();
    }
}
