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

using BH.oM.Python;
using BH.oM.Base.Attributes;

using System.ComponentModel;
using System.IO;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Convert an IES GEM file into a Honeybee JSON file.")]
        [Input("gem", "The GEM file.")]
        [Output("hbjson", "A Honeybee JSON file.")]
        public static string GEMtoHBJSON(string gem)
        {
            if (gem == null)
            {
                BH.Engine.Base.Compute.RecordError("gem input cannot be null.");
                return null;
            }

            if (!System.IO.File.Exists(gem))
            {
                BH.Engine.Base.Compute.RecordError($"{gem} doesn't appear to exist!");
                return null;
            }

            PythonEnvironment env = InstallPythonEnv_LBT(true);

            gem = System.IO.Path.GetFullPath(gem);
            string hbjsonFile = System.IO.Path.ChangeExtension(gem, ".hbjson");

            string script = Path.Combine(Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "gem_to_hbjson.py");

            // run the process
            string command = $"{env.Executable} {script} -g \"{gem}\"";
            string result = Python.Compute.RunCommandStdout(command: command, hideWindows: true);

            if (!File.Exists(hbjsonFile))
            {
                BH.Engine.Base.Compute.RecordError($"File conversion failed due to {result}");
            }

            return hbjsonFile;
        }
    }
}
