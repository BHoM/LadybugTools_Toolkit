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

using System;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using BH.oM.Adapter;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    [Description("Command that, when executed with the LadybugToolsAdapter, simulates UTCI values and outputs a heatmap. Output is a BH.oM.LadybugTools.PlotInformation containing the plot and extra information about the collection, and the ExternalComfort object that was used to get the UTCI values (whether the simulation ran or not).")]
    public class UTCIHeatPlotCommand : ISimulationCommand
    {
        [Description("The path to an EPW file.")]
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        [Description("The external comfort object containing the UTCI data to plot. If the UTCI collection is null or empty, then a simulation will be run before plotting to get these values.")]
        public virtual ExternalComfort ExternalComfort { get; set; } = new ExternalComfort();

        [Description("A wind speed multiplier to modify the wind speed by. Default is 1.")]
        public virtual double WindSpeedMultiplier { get; set; } = 1;

        [Description("A list of 10 colours to use for each UTCI category, leave empty to use the default UTCI colours.")]
        public virtual List<Color> BinColours { get; set; } = new List<Color>();

        [Description("Full file path (with file name) to save the plot to. Leave blank to output a base 64 string representation of the image instead.")]
        public virtual string OutputLocation { get; set; } = "";
    }
}

