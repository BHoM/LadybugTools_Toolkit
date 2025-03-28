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
using BH.oM.Base.Attributes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace BH.oM.LadybugTools
{
    [Description("Command that, when executed with the LadybugToolsAdapter, generates a windrose from the epw file using the given parameters.\nOutputs a string file path if the OutputLocation is given, or the base64 string representation of the image if no path is given.")]
    public class WindroseCommand : ISimulationCommand
    {
        [DisplayText("EPW File")]
        [Description("The path to an EPW file.")]
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        [DisplayText("Analysis Period")]
        [Description("The analysis period to use for plotting, default to whole non-leap year.")]
        public virtual AnalysisPeriod AnalysisPeriod { get; set; } = new AnalysisPeriod();

        [DisplayText("Number Of Directional Bins")]
        [Description("The number of directional bins to plot on the windrose.")]
        public virtual int NumberOfDirectionBins { get; set; } = 36;

        [DisplayText("Colour Map")]
        [Description("A Matplotlib colour map. Corresponds to the 'cmap' parameter of plot methods. See https://matplotlib.org/stable/users/explain/colors/colormaps.html for examples of valid keys. Default of 'viridis'.")]
        public virtual string ColourMap { get; set; } = "viridis";

        [DisplayText("Output Location")]
        [Description("Full file path (with file name) to save the plot to. Leave blank to output a base 64 string representation of the image instead.")]
        public virtual string OutputLocation { get; set; } = "";
    }
}

