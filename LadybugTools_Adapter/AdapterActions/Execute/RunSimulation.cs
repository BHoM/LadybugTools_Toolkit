﻿using BH.Engine.Adapter;
using BH.oM.Adapter;
using BH.oM.Data.Requests;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        private List<object> RunCommand(RunSimulationCommand command, ActionConfig actionConfig)
        {
            // validation prior to passing to Python
            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (!File.Exists(command.EPWFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"FIle '{command.EPWFile.GetFullFileName()}' does not exist.");
                return null;
            }

            if (command.GroundMaterial == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.GroundMaterial)} input cannot be null.");
                return null;
            }

            if (command.ShadeMaterial == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.ShadeMaterial)} input cannot be null.");
                return null;
            }

            // construct adapter and config
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
                EpwFile = Path.GetFullPath(command.EPWFile.GetFullFileName()).Replace(@"\", "/"),
                GroundMaterial = command.GroundMaterial,
                ShadeMaterial = command.ShadeMaterial,
                Name = Engine.LadybugTools.Compute.SimulationID(command.EPWFile.GetFullFileName(), command.GroundMaterial, command.ShadeMaterial)
            };

            // push object to json file
            Push(new List<SimulationResult>() { simulationResult }, actionConfig: config);

            // locate the Python file containing the simulation code
            string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "simulation_result.py");

            // run the simulation
            string cmdCommand = $"{m_environment.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";
            Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);

            // reload from Python results
            List<object> simulationResultPopulated = Pull(new FilterRequest(), actionConfig: config).ToList();

            // remove temporary file
            File.Delete(config.JsonFile.GetFullFileName());

            m_executeSuccess = true;
            return simulationResultPopulated;
        }
    }
}
