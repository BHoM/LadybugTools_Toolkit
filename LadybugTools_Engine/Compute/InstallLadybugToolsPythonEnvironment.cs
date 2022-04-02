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

using BH.Engine.Python;
using BH.oM.Python;
using BH.oM.Base.Attributes;
using System.IO;

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
            // The following checks for beta 5.1 only. These json files to be added to the installer for later versions. 
            if (!File.Exists(@"C:\ProgramData\BHoM\Settings\Python\LadybugTools_Toolkit.json"))
                Base.Compute.RecordError("LadybugTools_Toolkit.json does not exist. Visit the following page for more information: https://github.com/BHoM/LadybugTools_Toolkit/wiki/Beta-version-5.1---Installation-instructions");
            if (!File.Exists(@"C:\ProgramData\BHoM\Settings\Python\Python_Toolkit.json"))
                Base.Compute.RecordError("Python_Toolkit.json does not exist. Visit the following page for more information: https://github.com/BHoM/LadybugTools_Toolkit/wiki/Beta-version-5.1---Installation-instructions");

            return BH.Engine.Python.Compute.InstallPythonEnvironment(run, force, @"C:\ProgramData\BHoM\Settings\Python\LadybugTools_Toolkit.json");
        }
    }
}
