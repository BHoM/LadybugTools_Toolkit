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
using BH.oM.LadybugTools;
using System;
using BH.Engine.Serialiser;
using BH.Adapter.LadybugTools;
using BH.oM.Adapter;
using System.Collections.Generic;
using BH.oM.Data.Requests;
using BH.Engine.Adapter;
using System.Linq;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Run a simulation and return results.")]
        [Input("epwFile", "An EPW file.")]
        [Input("groundMaterial", "A ground material.")]
        [Input("shadeMaterial", "A shade material.")]
        [Output("simulationResult", "An simulation result object containing simulation results.")]
        [PreviousVersion("7.0", "BH.Engine.LadybugTools.Compute.SimulationResult(System.String, BH.oM.LadybugTools.ILadybugToolsMaterial, BH.oM.LadybugTools.ILadybugToolsMaterial)")]
        public static SimulationResult SimulationResult(string epwFile, IEnergyMaterialOpaque groundMaterial, IEnergyMaterialOpaque shadeMaterial)
        {
            // validation prior to passing to Python
            if (epwFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(epwFile)} input cannot be null.");
                return null;
            }
            if (!File.Exists(epwFile))
            {
                BH.Engine.Base.Compute.RecordError($"{epwFile} does not exist.");
                return null;
            }

            if (groundMaterial == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(groundMaterial)} input cannot be null.");
                return null;
            }

            if (shadeMaterial == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(shadeMaterial)} input cannot be null.");
                return null;
            }

            // construct adapter and config
            LadybugToolsAdapter adapter = new LadybugToolsAdapter();
            LadybugConfig config = new LadybugConfig()
            {
                JsonFile = new FileSettings()
                {
                    FileName = $"LBTBHoM_{Guid.NewGuid()}.json",
                    Directory = Path.GetTempPath()
                }
            };

            // construct the base object and file to be passed to Python for simulation
            SimulationResult simulationResult = new SimulationResult()
            {
                EpwFile = Path.GetFullPath(epwFile).Replace(@"\", "/"),
                GroundMaterial = groundMaterial,
                ShadeMaterial = shadeMaterial,
                Name = Compute.SimulationID(epwFile, groundMaterial, shadeMaterial)
            };

            // push object to json file
            adapter.Push(new List<SimulationResult>() { simulationResult }, actionConfig: config);

            // locate the Python executable and file containing the simulation code
            PythonEnvironment env = InstallPythonEnv_LBT(true);
            string script = Path.Combine(Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "simulation_result.py");
            
            // run the simulation
            string command = $"{env.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";
            Python.Compute.RunCommandStdout(command: command, hideWindows: true);

            // reload from Python results
            SimulationResult simulationResultPopulated = adapter.Pull(new FilterRequest(), actionConfig: config).Cast<SimulationResult>().ToList()[0];
            
            // remove temporary file
            File.Delete(config.JsonFile.GetFullFileName());

            return simulationResultPopulated;
        }
    }
}
