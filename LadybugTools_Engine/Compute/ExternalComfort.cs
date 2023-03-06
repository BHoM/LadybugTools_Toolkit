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
using BH.oM.Python;
using BH.oM.Base.Attributes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using BH.oM.Base;
using System.IO;
using BH.oM.LadybugTools;
using BH.Engine.Serialiser;
using Rhino.DocObjects;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Run an External Comfort simulation and return results.")]
        [Input("simulationResult", "A simulation result object.")]
        [Input("typology", "An ExternalComfortTypology.")]
        [Output("externalComfort", "An external comfort result object containing simulation results.")]
        public static ExternalComfort ExternalComfort(SimulationResult simulationResult, Typology typology)
        {
            if (simulationResult == null)
            {
                BH.Engine.Base.Compute.RecordError("simulationResult input cannot be null.");
                return null;
            }

            if (typology == null)
            {
                BH.Engine.Base.Compute.RecordError("typology input cannot be null.");
                return null;
            }

            // construct the base object
            ExternalComfort externalComfort = new ExternalComfort()
            {
                SimulationResult = simulationResult,
                Typology = typology,
            };

            // send to Python to simulate/load
            string externalComfortJsonStr = System.Text.RegularExpressions.Regex.Unescape(externalComfort.ToJson());
            PythonEnvironment env = Python.Query.ExistingEnvironment(Query.ToolkitName());
            string pythonScript = string.Join("\n", new List<string>()
            {
                "try:",
                "    import json",
                "    from ladybugtools_toolkit.external_comfort.external_comfort import ExternalComfort",
                $"    external_comfort = ExternalComfort.from_json('{externalComfortJsonStr}')",
                "    print(external_comfort.to_json())",
                "except Exception as exc:",
                "    print(json.dumps({'error': str(exc)}))",
            });

            string output = env.RunPythonString(pythonScript).Trim().Split(new string[] { "\r\n", "\r", "\n" }, StringSplitOptions.None).Last();

            // reload from Python results
            return (ExternalComfort)Serialiser.Convert.FromJson(output);
        }
    }
}


