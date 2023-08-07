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
        [PreviousVersion("6.3", "BH.Engine.LadybugTools.Compute.InstallPythonEnv_LBT(System.Boolean)")]
        public static PythonEnvironment InstallPythonEnv_LBT(bool run = false, bool reinstall = false)
        {
            // check if referenced Python is installed
            string referencedExecutable = @"C:\Program Files\ladybug_tools\python\python.exe";
            if (!File.Exists(referencedExecutable))
            {
                Base.Compute.RecordError($"Could not find referenced python executable at {referencedExecutable}. Please install Pollination try again.");
                return null;
            }

            if (!run)
                return null;

            // find out whether this environment already exists
            bool exists = Python.Query.VirtualEnvironmentExists(Query.ToolkitName());

            if (reinstall)
                Python.Compute.RemoveVirtualEnvironment(Query.ToolkitName());
            
            // obtain python version
            PythonVersion pythonVersion = Python.Query.Version(referencedExecutable);

            // create virtualenvironment
            PythonEnvironment env = Python.Compute.VirtualEnvironment(version: pythonVersion, name: Query.ToolkitName(), reload: true);

            // return null if environment could not be created/loaded
            if (env == null)
                return null;

            // install packages if this is a reinstall, or the environment did not originally exist
            if (reinstall || !exists)
            {
                // install local package
                env.InstallPackageLocal(Path.Combine(Python.Query.DirectoryCode(), Query.ToolkitName()));

                // create requiremetns from referenced executable
                string requirementsTxt = Python.Compute.RequirementsTxt(referencedExecutable, Path.Combine(Python.Query.DirectoryEnvironments(), $"requirements_{Query.ToolkitName()}.txt"));
                env.InstallRequirements(requirementsTxt);
                File.Delete(requirementsTxt);
            }

            return env;
        }
    }
}

