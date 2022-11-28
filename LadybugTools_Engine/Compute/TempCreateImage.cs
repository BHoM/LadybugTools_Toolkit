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

using BH.Engine.Python;
using BH.oM.Base.Attributes;

using System;
using System.Collections.Generic;
using System.ComponentModel;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Create an image as either a path to that image, or the image encoded as base64.")]
        [Input("epwFile", "A path to an EPW file to load some data from.")]
        [Input("asFile", "Set to True to generate a filepath, or False to generate a base64 encoded string.")]
        [Output("imageString", "The resultant resulting result.")]
        public static string TempCreateImage(string epwFile, bool asFile = true)
        {
            BH.oM.Python.PythonEnvironment env = Compute.InstallPythonEnv_LBT(true);

            if (!System.IO.File.Exists(epwFile))
            {
                BH.Engine.Base.Compute.RecordError("The epw file given doesn't appear to exist!");
            }
            string option = asFile ? "path" : "base64";
            string pythonScript = String.Join("\n", new List<string>()
            {
                "from ladybugtools_toolkit.bhomutil.interface import plot_example",
                "",
                "try:",
                $"    result = plot_example(r\"{epwFile}\", \"{option}\")",
                $"    print(result)",
                "except Exception as exc:",
                "    print(exc)",
            });

            return env.RunPythonString(pythonScript).Trim();
        }
    }
}
