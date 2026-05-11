/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2026, the respective contributors. All rights reserved.
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

using BH.oM.Adapter;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class CompareEPWKeyPlotCommand : ISimulationCommand
    {
        [Description("The EPW file that acts as the base for comparisons.")]
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        [Description("Key (e.g. Dry Bulb Temperature) to compare.")]
        public virtual EPWKey EPWKey { get; set; } = EPWKey.Undefined;

        [Description("The list of EPW files to be compared with the base file (or each other).")]
        public virtual List<FileSettings> EPWCompareFiles { get; set; } = new List<FileSettings>();

        [Description("Whether to plot a time series chart. If set to false, plots data as a histogram instead.")]
        public virtual bool PlotTimeseries { get; set; } = true;

        [Description("The location to place the image file once complete.")]
        public virtual string OutputLocation { get; set; } = "";
    }
}
