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
using System.Collections.Generic;

using BH.oM.Python;
using BH.oM.Base.Attributes;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("LadybugTools_Toolkit\nMethod used to create the Python environment used to run all Python scripts within this toolkit.")]
        [Input("run", "Starts the installation of the toolkit if true. Stays idle otherwise.")]
        [Input("force", "Forces re-installation of the toolkits Python environment (and the base Python environment) if true. This is used when the environment has been updated or package versions have changed.")]
        [Output("pythonEnvironment", "The LadybugTools_Toolkit Python environment.")]
        public static PythonEnvironment InstallLadybugToolsPythonEnvironment(bool run = false, bool force = false)
        {
            PythonEnvironment pythonEnvironment = BH.Engine.Python.Compute.InstallPythonEnvironment(run, force, @"C:\ProgramData\BHoM\Settings\Python\LadybugTools_Toolkit.json");

            // set up the %USER%\ladybug_tools folder in the same way as ladybug/honeybee would using teh GH installer/updater component.


            //// Set-up the %USER%\ladybug_tools folder to contain necessary referenced code/tools
            //string ladybugToolsDirectory = Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.UserProfile), "ladybug_tools");
            //Directory.CreateDirectory(ladybugToolsDirectory);

            //// check that simulation/processing dependencies are available
            //if (!Directory.Exists(Path.Combine(ladybugToolsDirectory, "openstudio")))
            //    // This if-statement Python-esque formatting is here for you Fraser
            //    BH.Engine.Base.Compute.RecordWarning($"Openstudio is not installed in {Path.Combine(ladybugToolsDirectory, "openstudio")} and any commands used that call to it will not work. Follow instructions https://github.com/ladybug-tools/lbt-grasshopper/wiki/1.1-Windows-Installation-Steps#optional-steps to install it.");

            //if (!Directory.Exists(Path.Combine(ladybugToolsDirectory, "radiance")))
            //    BH.Engine.Base.Compute.RecordWarning($"Radiance is not installed in {Path.Combine(ladybugToolsDirectory, "radiance")} and any commands used that call to it will not work. Follow instructions https://github.com/ladybug-tools/lbt-grasshopper/wiki/1.1-Windows-Installation-Steps#optional-steps to install it.");
            
            //if (!Directory.Exists(Path.Combine(ladybugToolsDirectory, "resources", "measures", "honeybee_openstudio_gem")))
            //    BH.Engine.Base.Compute.RecordWarning($"Honeybee Openstudio Gem measures are not available in {Path.Combine(ladybugToolsDirectory, "resources", "measures", "honeybee_openstudio_gem")} and any commands used that call to them will not work. Follow instructions https://github.com/ladybug-tools/lbt-grasshopper/wiki/1.1-Windows-Installation-Steps to install it.");

            return pythonEnvironment;
        }
    }
}
