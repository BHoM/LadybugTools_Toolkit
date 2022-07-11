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
using BH.Engine.Python;
using System.ComponentModel;
using System.IO;
using System.Collections.Generic;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Install additional packages and code to LadybugTools_Toolkit Python Environment.")]
        [Input("run", "Run the installation process for the BHoM Python Environment.")]
        [Output("env", "The LadybugTools_Toolkit Python Environment, with BHoM code added.")]
        public static PythonEnvironment LadybugToolsPythonEnvironment(bool run = false)
        {
            // set-up bits and pieces describing the env, prior to running checks/processes
            string toolkitName = Query.ToolkitName();
            string envsDir = Engine.Python.Query.EnvironmentsDirectory();
            string codeDir = Engine.Python.Query.CodeDirectory();
            string toolkitEnvDir = Path.Combine(envsDir, toolkitName);
            oM.Python.PythonEnvironment thisEnv = new oM.Python.PythonEnvironment()
            {
                Name = Query.ToolkitName(),
                Executable = Path.Combine(@"C:\Program Files\ladybug_tools\python\python.exe"),
            };
            bool thisEnvExists = Engine.Python.Query.EnvironmentExists(@"C:\Program Files\ladybug_tools", "python");

            if (!thisEnvExists)
            {
                BH.Engine.Base.Compute.RecordError("It seems that Ladybug Tools/Pollination has not been installed. " +
                    "Please run the Pollination installer in order to use the Python Environment associated with this toolkit.");
                return null;
            }

            if (run)
            {
                // check that the env contains BHoM code references already, and if it does, then return the existing env
                string installedPkgsCmd = $"{Engine.Python.Modify.AddQuotesIfRequired(thisEnv.Executable)} -m pip list";
                string installedPkgs = Engine.Python.Compute.RunCommandStdout(installedPkgsCmd);
                if (
                    installedPkgs.Contains("ladybugtools-toolkit") &&
                    installedPkgs.Contains("python-toolkit") &&
                    installedPkgs.Contains("ipykernel")
                )
                {
                    return thisEnv;
                }

                // reference the BHoM code within this environment
                List<string> additionalPackages = new List<string>() {
                        Path.Combine(BH.Engine.Python.Query.CodeDirectory(), "Python_Toolkit"),
                        Path.Combine(BH.Engine.Python.Query.CodeDirectory(), toolkitName),
                    };
                string installedPkgsResult = thisEnv.InstallLocalPackages(additionalPackages);

                // load base Python environment to ensure additional BHoM code is available
                oM.Python.PythonEnvironment baseEnv = BH.Engine.Python.Compute.PythonToolkitEnvironment(run);

                // install ipykernel into this env and register with the base BHoM Python environment
                Engine.Python.Compute.InstallPackages(thisEnv, new List<string>() { "ipykernel" });
                string kernelCreateCmd = $"{Engine.Python.Modify.AddQuotesIfRequired(thisEnv.Executable)} -m ipykernel install --name={toolkitName}";
                Engine.Python.Compute.RunCommandStdout(kernelCreateCmd);

                return thisEnv;
            }

            return null;
        }
    }
}
