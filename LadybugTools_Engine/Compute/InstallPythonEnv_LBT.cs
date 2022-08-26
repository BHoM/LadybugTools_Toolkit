/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2022, the respective contributors. All rights reserved.
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
using BH.oM.Python;
using System.ComponentModel;
using System.IO;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Create the BHoM Python envrionment for LadybugTools_Toolkit. This creates a replica of what is found in the Pollination installed python environment, for extension using BHoM.")]
        [Input("run", "Run the installation process.")]
        [Output("env", "The LadybugTools_Toolkit Python Environment, with BHoM code accessible.")]
        public static PythonEnvironment InstallPythonEnv_LBT(bool run = false)
        {
            return BH.Engine.Python.Compute.InstallReferencedVirtualenv(
                name: Query.ToolkitName(),
                executable: @"C:\Program Files\ladybug_tools\python\python.exe",
                localPackage: Path.Combine(Engine.Python.Query.CodeDirectory(), Query.ToolkitName()),
                run: run
            );
        }
    }
}
