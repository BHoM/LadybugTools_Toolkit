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

using BH.oM.Adapter;
using BH.oM.Base;
using BH.oM.Base.Attributes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Reflection;
using System.Text;

namespace BH.oM.LadybugTools
{
    [Description("Use in conjunction with the LadybugToolsAdapter to run a diurnal analysis on a specific key of an epw file, and output a plot.")]
    public class DiurnalPlotCommand : ISimulationCommand
    {
        [DisplayText("EPW File")]
        [Description("The EPW file to analyse.")]
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        [DisplayText("EPW Key")]
        [Description("The key in the EPW file to analyse.")]
        public virtual EPWKey EPWKey { get; set; } = EPWKey.Undefined;

        [Description("The colour of the average line on the plot.")]
        public virtual Color Colour { get; set; }

        [DisplayText("Output Location")]
        [Description("The directory to output the file. Leave empty to return a base64 string representation of that image.")]
        public virtual string OutputLocation { get; set; } = "";

        [Description("Title of the plot, will appear above any information on the top of the plot.")]
        public virtual string Title { get; set; } = "";

        [DisplayText("Diurnal Period")]
        [Description("The diurnal period to analyse. Daily for 365 samples/timestep, weekly for 52, monthly for 30.")]
        public virtual DiurnalPeriod Period { get; set; } = DiurnalPeriod.Undefined;
    }
}

