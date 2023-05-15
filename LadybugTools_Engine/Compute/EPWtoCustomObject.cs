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

using BH.Engine.Python;
using BH.oM.Base;
using BH.oM.Python;
using BH.oM.Base.Attributes;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Convert an EPW file into a time-indexed CSV version.")]
        [Input("epwFile", "An EPW file.")]
        [Output("object", "A BHoM object wrapping a Ladybug EPW object.")]
        public static CustomObject EPWtoCustomObject(string epwFile)
        {
            if (epwFile == null)
            {
                BH.Engine.Base.Compute.RecordError("epwFile input cannot be null.");
                return null;
            }

            if (!System.IO.File.Exists(epwFile))
            {
                BH.Engine.Base.Compute.RecordError($"{epwFile} doesn't appear to exist!");
                return null;
            }

            PythonEnvironment env = Python.Query.VirtualEnv(Query.ToolkitName());

            string pythonScript = string.Join("\n", new List<string>()
            {
                "import json",
                "from pathlib import Path",
                "from ladybug.epw import EPW",
                "",
                $"epw_path = Path(r'{epwFile}')",
                "try:",
                "    print(json.dumps(EPW(epw_path.as_posix()).to_dict()))",
                "except Exception as exc:",
                "    print(exc)",
            });

            string output = env.RunPythonString(pythonScript).Trim();

            return Serialiser.Convert.FromJson(output) as CustomObject;
        }
    }
}

