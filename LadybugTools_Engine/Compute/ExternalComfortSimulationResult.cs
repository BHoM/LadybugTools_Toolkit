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
using BH.oM.Python;
using BH.oM.Base.Attributes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using BH.oM.Base;
using System.IO;
using BH.oM.LadybugTools;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Run an External Comfort simulation and return results.")]
        [Input("identifier", "An identifier used to make these results unique. If the materials and epwFile given match a results set with the same identifier, then those reuslts will be returned instead of running the simulation again.")]
        [Input("epwFile", "An EPW file.")]
        [Input("groundMaterial", "A pre-defined ground material.")]
        [Input("shadeMaterial", "A pre-defined shade material.")]
        [Output("externalComfortResult", "An external comfort result object containing simulation results.")]
        public static CustomObject ExternalComfortSimulationResult(string identifier, string epwFile, ExternalComfortMaterial groundMaterial, ExternalComfortMaterial shadeMaterial)
        {
            BH.oM.Python.PythonEnvironment env = Compute.InstallPython_LBT(true);

            string pythonScript = string.Join("\n", new List<string>()
            {
                "try:",
                "    from pathlib import Path",
                "    from ladybug.epw import EPW",
                "    from ladybugtools_toolkit.external_comfort import SimulationResult",
                "    from ladybugtools_toolkit.external_comfort.encoder.encoder import Encoder",
                "    from honeybee_energy.material.opaque import EnergyMaterial",
                "    import json",
                "",
                $"    epw_path = Path(r'{epwFile}')",
                "    epw = EPW(epw_path.as_posix())",
                $"    gnd_mat = {groundMaterial.PythonString()}",
                $"    shd_mat = {shadeMaterial.PythonString()}",
                "",
                $"    simulation_result = SimulationResult(epw, gnd_mat, shd_mat, identifier='{identifier}')",
                "    print(json.dumps(simulation_result.to_dict(), cls=Encoder))",
                "except Exception as exc:",
                "    print(json.dumps({'error': str(exc)}))",
            });

            string output = env.RunPythonString(pythonScript).Trim().Split(new string[] { "\r\n", "\r", "\n" }, StringSplitOptions.None).Last();

            return Serialiser.Convert.FromJson(output) as CustomObject;
        }
    }
}

