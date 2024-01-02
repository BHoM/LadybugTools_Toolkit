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
using BH.oM.Python;
using BH.oM.Python.Enums;
using System.ComponentModel;
using System.IO;
using BH.Engine.Python;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Create the BHoM Python environment for LadybugTools_Toolkit. This creates a replica of what is found in the Pollination installed python environment, for extension using BHoM.")]
        [Input("run", "Run the installation process.")]
        [Input("reinstall", "Reinstall the environment if it already exists.")]
        [Output("env", "The LadybugTools_Toolkit Python Environment, with BHoM code accessible.")]
        public static PythonEnvironment InstallPythonEnv_LBT(bool run = false, bool reinstall = false)
        {
            if (!run)
                return null;

            // check if referenced Python is installed, and get executable and version if it is
            if (!Query.IsPollinationInstalled())
                return null;
            string referencedExecutable = System.Environment.GetFolderPath(System.Environment.SpecialFolder.ProgramFiles) + @"\ladybug_tools\python\python.exe";
            PythonVersion pythonVersion = Python.Query.Version(referencedExecutable);

            // check if environment already exists. If it does, and no reinstall requested, load it
            bool exists = Python.Query.VirtualEnvironmentExists(envName: Query.ToolkitName(), pythonVersion: pythonVersion);
            if (run && exists && !reinstall)
                return Python.Compute.VirtualEnvironment(version: pythonVersion, name: Query.ToolkitName(), reload: true);

            // create environment from scratch
            PythonEnvironment env = Python.Compute.VirtualEnvironment(version: pythonVersion, name: Query.ToolkitName(), reload: false);

            // install local package
            env.InstallPackageLocal(Path.Combine(Python.Query.DirectoryCode(), Query.ToolkitName()));

            // create requirements from referenced executable
            string requirementsTxt = Python.Compute.RequirementsTxt(referencedExecutable, Path.Combine(Python.Query.DirectoryEnvironments(), $"requirements_{Query.ToolkitName()}.txt"));
            env.InstallRequirements(requirementsTxt);

            return env;
        }
    }
}


