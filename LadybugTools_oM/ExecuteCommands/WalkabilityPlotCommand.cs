﻿/*
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
    public class WalkabilityPlotCommand: ISimulationCommand
    {
        [DisplayText("EPW File")]
        [Description("The EPW file to use for analysis.")]
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        [DisplayText("External Comfort")]
        [Description("The external comfort object containing the UTCI data to plot. If the UTCI collection is null or empty, then a simulation will be run before plotting to get these values.")]
        public virtual ExternalComfort ExternalComfort { get; set; } = null;

        [DisplayText("Output Location")]
        [Description("The location to place any images generated by the command. Leave empty to return a base64 string representation of the image.")]
        public virtual string OutputLocation { get; set; } = "";
    }
}
