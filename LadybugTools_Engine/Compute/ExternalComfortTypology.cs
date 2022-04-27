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
        [Description("Run an External Comfort simulation and return results.")]
        [Input("epw", "An EPW file.")]
        [Input("groundMaterial", "A pre-defined ground material.")]
        [Input("shadeMaterial", "A pre-defined shade material.")]
        [Input("typology", "A pre-defined external comfort typology.")]
        [Output("typologyResult", "A typology result object containing simulation results and typology specific comfort metrics.")]
        public static CustomObject ExternalComfortTypology(string epw, ExternalComfortMaterial groundMaterial, ExternalComfortMaterial shadeMaterial, BH.oM.Ladybug.ExternalComfortTypology typology)
        {
            PythonEnvironment pythonEnvironment = Python.Query.LoadPythonEnvironment(Query.ToolkitName());
            if (!pythonEnvironment.IsInstalled())
            {
                BH.Engine.Base.Compute.RecordError($"Install the {Query.ToolkitName()} Python environment before running this method (using {Query.ToolkitName()}.Compute.InstallPythonEnvironment).");
                return null;
            }

            if (!ExternalComfortPossible())
                return null;

            if (groundMaterial == ExternalComfortMaterial.Undefined || shadeMaterial == ExternalComfortMaterial.Undefined)
            {
                BH.Engine.Base.Compute.RecordError($"Please input a valid ExternalComfortMaterial.");
                return null;
            }

            if (typology == BH.oM.Ladybug.ExternalComfortTypology.Undefined)
            {
                BH.Engine.Base.Compute.RecordError($"Please input a valid ExternalComfortTypology.");
                return null;
            }

            if (!File.Exists(epw))
            {
                BH.Engine.Base.Compute.RecordError($"The EPW file given cannot be found.");
                return null;
            }

            string epwPath = Path.GetFullPath(epw);

            string outputPath = Path.Combine(Path.GetTempPath(), $"{System.Guid.NewGuid()}.json");

            string pythonScript = String.Join("\n", new List<string>() 
            {
                "import sys",
                $"sys.path.insert(0, '{pythonEnvironment.CodeDirectory()}')",
                "",
                "from ladybug.epw import EPW",
                "from external_comfort.external_comfort import ExternalComfort, ExternalComfortResult",
                "from external_comfort.material import MATERIALS",
                "from external_comfort.typology import Typologies, TypologyResult",
                "",
                $"epw = EPW(r'{epwPath}')",
                $"ec = ExternalComfort(epw, ground_material=MATERIALS['{groundMaterial}'], shade_material=MATERIALS['{shadeMaterial}'])",
                "ecr = ExternalComfortResult(ec)",
                $"typ = Typologies.{typology}.value",
                "typr = TypologyResult(typ, ecr)",
                $"typr.to_json(r'{outputPath}')",
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

