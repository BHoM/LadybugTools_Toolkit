/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2023, the respective contributors. All rights reserved.
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
    [Description("Command that, when executed with the LadybugToolsAdapter, simulates UTCI values and outputs a heatmap.\nOutput is a string of either the path to the image (if OutputLocation is not set) or the base 64 string representation of it.")]
    public class UTCIHeatPlotCommand : ISimulationCommand
    {
        [Description("The path to an EPW file.")]
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        [Description("An opaque energy material to use as the ground material in the simulation.")]
        public virtual IEnergyMaterialOpaque GroundMaterial { get; set; } = null;

        [Description("An opaque energy material to use as the shade material in the simulation.")]
        public virtual IEnergyMaterialOpaque ShadeMaterial { get; set; } = null;

        [Description("A typology to use in the simulation containing any shelters and evaporative cooling effects. TargetWindSpeed is not used in this simulation")]
        public virtual Typology Typology { get; set; } = null;

        [Description("A wind speed multiplier to modify the wind speed by. Default is 1.")]
        public virtual double WindSpeedMultiplier { get; set; } = 1;

        [Description("A list of 10 colours to use for each UTCI category, leave empty to use the default UTCI colours.")]
        public virtual List<Color> BinColours { get; set; } = new List<Color>();

        [Description("Full file path (with file name) to save the plot to. Leave blank to output a base 64 string representation of the image instead.")]
        public virtual string OutputLocation { get; set; } = "";
    }
}