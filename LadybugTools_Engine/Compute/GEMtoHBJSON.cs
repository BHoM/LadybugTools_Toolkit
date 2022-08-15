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
        [Description("Convert an IES GEM file into a Honeybee JSON file.")]
        [Input("gem", "The GEM file.")]
        [Output("hbjson", "A Honeybee JSON file.")]
        public static string GEMtoHBJSON(string gem)
        {
            BH.oM.Python.PythonEnvironment env = Compute.LadybugToolsToolkitPythonEnvironment(true);

            string gemFile = System.IO.Path.GetFullPath(gem);
            string outputDirectory = System.IO.Path.GetDirectoryName(gem);
            string fileName = System.IO.Path.GetFileNameWithoutExtension(gem);

            string pythonScript = String.Join("\n", new List<string>()
            {
                "from honeybee.model import Model",
                "from honeybee_ies.reader import model_from_ies",
                "",
                "try:",
                $"    model = model_from_ies(r\"{gemFile}\")",
                $"    hbjson_file = model.to_hbjson(folder=r\"{outputDirectory}\", name=\"{fileName}\")",
                "    print(hbjson_file)",
                "except Exception as exc:",
                "    print(exc)",
            });

            return env.RunPythonString(pythonScript).Trim();
        }
    }
}
