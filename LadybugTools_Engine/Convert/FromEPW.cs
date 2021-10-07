/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2021, the respective contributors. All rights reserved.
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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using BH.oM.Geometry;
using BH.oM.Base;
using BH.oM.Environment;
using BH.oM.Environment.Elements;
using IronPython;

using BH.Engine.Geometry;
using BH.Engine.Environment;

using BH.oM.Reflection.Attributes;
using System.ComponentModel;
using System.IO;

namespace BH.Engine.LadybugTools
{
    public static partial class Convert
    {
        [Description(".")]
        [Input("input", ".")]
        [Output("output", ".")]
        public static string FromEPW(string epwFile)
        {
            // Create the Python code to be run
            List<string> pythonCode = new List<string>()
            {
                "from ladybug.epw import EPW",
                "import json",
                "",
                "def epw_to_json_string({0}):",
                "    epw = EPW(epw_file)",
                "    json_string = json.dumps(epw.to_dict(), indent=4)",
                "    return json_string"
            };
            string code = String.Format(String.Join("\n", pythonCode.ToArray()), String.Format("\"{0}\"", epwFile));

            // Reference location where LB code stored
            string lib = Path.Combine(Python.Query.EmbeddedPythonHome(), "Lib", "site-packages");

            // Create the Python engine, reference the installed libraries, and set the scope
            Microsoft.Scripting.Hosting.ScriptEngine engine = IronPython.Hosting.Python.CreateEngine();
            engine.SetSearchPaths(new List<string>() { lib });
            Microsoft.Scripting.Hosting.ScriptScope scope = engine.CreateScope();

            // Run the Python code
            string thing = engine.Execute(code, scope);

            return thing;

            // IronPython LAdybug read file into LB EPW
            // LB convert to JSON in memory
            // Pass back to BHoM and create custom object from JSON string
            //return new CustomObject();
        }
    }
}

