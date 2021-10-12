﻿/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2021, the respective contributors. All rights reserved.
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

using BH.oM.Base;
using BH.oM.Reflection.Attributes;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Create the virtual environment associated with this toolkit.")]
        [Input("force", "Force the recreation of the environment.")]
        [Input("run", "Run the installer for this toolkits Python virtual environment.")]
        [Output("executable", "The path to the virtual environment's Python executable.")]
        public static bool CheckVirtualEnvironmentInstalled()
        {
            if (!File.Exists(Python.Query.VirtualEnvironmentExecutable(Compute.VIRTUALENV_NAME)))
            {
                BH.Engine.Reflection.Compute.RecordError("It looks like the LadybugTools virtual environment hasn't been installed.");
                return false;
            }
            return true;
        }
    }
}