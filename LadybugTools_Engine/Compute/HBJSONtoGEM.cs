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
        [Description("Convert a Honeybee JSON into an IES GEM file.")]
        [Input("hbjson", "A Honeybee JSON file.")]
        [Output("gem", "The GEM file.")]
        public static string HBJSONtoGEM(string hbjson)
        {
            if (hbjson == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(hbjson)} input cannot be null.");
            }

            if (!System.IO.File.Exists(hbjson))
            {
                BH.Engine.Base.Compute.RecordError($"{hbjson} doesn't appear to exist!");
            }

            BH.oM.Python.PythonEnvironment env = Compute.InstallPythonEnv_LBT(true);

            string hbjsonFile = System.IO.Path.GetFullPath(hbjson);
            string outputDirectory = System.IO.Path.GetDirectoryName(hbjsonFile);
            string fileName = System.IO.Path.GetFileNameWithoutExtension(hbjsonFile);

            string pythonScript = String.Join("\n", new List<string>()
            {
                "from honeybee.model import Model",
                "from pathlib import Path",
                "",
                "try:",
                $"    model = Model.from_hbjson(r\"{hbjsonFile}\")",
                $"    gem_file = model.to_gem(r\"{outputDirectory}\", name=\"{fileName}\")",
                "    print(gem_file)",
                "except Exception as exc:",
                "    print(exc)",
            });

            return env.RunPythonString(pythonScript).Trim();
        }
    }
}
