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
using BH.oM.Ladybug;
using BH.oM.Python;
using BH.oM.Base.Attributes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using BH.oM.Base;
using System.IO;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Post-process a spatial-comfort simulation and return results.")]
        [Input("epw", "An EPW file.")]
        [Output("spatialComfortResult", "A spatial comfort result object containing simulation results and paths to outputs.")]
        public static CustomObject SpatialComfort(string epw, string simulationDirectory)
        {
            PythonEnvironment pythonEnvironment = Python.Query.LoadPythonEnvironment(Query.ToolkitName());
            if (!pythonEnvironment.IsInstalled())
            {
                BH.Engine.Base.Compute.RecordError($"Install the {Query.ToolkitName()} Python environment before running this method (using {Query.ToolkitName()}.Compute.InstallPythonEnvironment).");
                return null;
            }

            if (!Query.SpatialComfortPossible(simulationDirectory))
                return null;

            
            if (!File.Exists(epw))
            {
                BH.Engine.Base.Compute.RecordError($"The EPW file given cannot be found.");
                return null;
            }

            string outputPath = Path.Combine(Path.GetTempPath(), "ecr.json");

            string epwPath = Path.GetFullPath(epw);

            string pythonScript = String.Join("\n", new List<string>() 
            {
                "import sys",
                $"sys.path.insert(0, '{pythonEnvironment.CodeDirectory()}')",
                "",
                "from ladybug.epw import EPW",
                "from external_comfort.external_comfort import ExternalComfort, ExternalComfortResult",
                "from external_comfort.material import MATERIALS",
                "",
                $"epw = EPW(r'{epwPath}')",
                $"ec = ExternalComfort(epw, ground_material=MATERIALS['ConcreteHeavyweight'], shade_material=MATERIALS['Fabric'])",
                "ecr = ExternalComfortResult(ec)",
                $"ecr.to_json(r'{outputPath}')",
                "print('Nothing to see here!')",
            });

            string output = Python.Compute.RunPythonString(pythonEnvironment, pythonScript).Trim();

            string jsonString = "";
            using (StreamReader r = new StreamReader(outputPath))
            {
                jsonString = r.ReadToEnd();
            }


            return Serialiser.Convert.FromJson(jsonString) as CustomObject;
        }
    }
}
