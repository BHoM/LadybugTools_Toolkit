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
using System.IO.Compression;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Download a file from a URL to a directory.")]
        [Input("environment", "The BHoM Python environment in which the package will be updated.")]
        [Input("package", "The package (and it's target version) to be installed/updated.")]
        public static string UpdatePythonPackage(PythonEnvironment environment, PythonPackage package)
        {
            string cmd = $"{environment.PythonExecutable()} -m pip install --upgrade {package.Name}=={package.Version}";
            return BH.Engine.Python.Compute.RunCommandStdout(cmd);
        }
    }
}

