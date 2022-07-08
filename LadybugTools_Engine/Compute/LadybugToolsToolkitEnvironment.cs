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

using System.ComponentModel;
using System.IO;

using BH.oM.Python;
using BH.oM.Base.Attributes;
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
            // install Python_Toolkit BHoM Environment if it's not already installed
            BH.Engine.Python.Compute.PythonToolkitEnvironment(true);

            string toolkitName = Query.ToolkitName();
            string toolkitEnvironmentDirectory = @"C:\Program Files\ladybug_tools\python";

            if (run)
            {
                if (!Directory.Exists(toolkitEnvironmentDirectory))
                {
                    BH.Engine.Base.Compute.RecordError("It seems that Ladybug Tools/Pollination has not been installed. " +
                        "Please run the installer from https://app.pollination.cloud in order to access the Python Environment " +
                        "associated with this toolkit.");
                    return null;
                }

                // obtain the environment
                oM.Python.PythonEnvironment env = new oM.Python.PythonEnvironment()
                {
                    Name = Query.ToolkitName(),
                    Executable = Path.Combine(toolkitEnvironmentDirectory, "python.exe"),
                };

                // install the BHoM code into this environment
                List<string> additionalPackages = new List<string>() { 
                    Path.Combine(BH.Engine.Python.Query.CodeDirectory(), "Python_Toolkit"),
                    Path.Combine(BH.Engine.Python.Query.CodeDirectory(), toolkitName),
                };
                foreach (string pkg in additionalPackages)
                {
                    Engine.Python.Compute.InstallLocalPackage(env, pkg, true);
                }

                // install ipykernel and register environment with the base BHoM Python environment
                Engine.Python.Compute.InstallPackages(env, new List<string>() { "ipykernel" });
                string kernelCreateCmd = $"{Engine.Python.Modify.AddQuotesIfRequired(env.Executable)} -m ipykernel install --name={toolkitName}";
                Engine.Python.Compute.RunCommandStdout(kernelCreateCmd);

                return env;
            }

            return null;
        }
    }
}
