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
        private List<object> RunCommand(RunExternalComfortCommand command, ActionConfig actionConfig)
        {
            if (command.SimulationResult == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.SimulationResult)} input cannot be null.");
                return null;
            }

            if (command.Typology == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.Typology)} input cannot be null.");
                return null;
            }

            LadybugConfig config = new LadybugConfig()
            {
                JsonFile = new FileSettings()
                {
                    FileName = $"LBTBHoM_{Guid.NewGuid()}.json",
                    Directory = Path.GetTempPath()
                }
            };

            // construct the base object
            ExternalComfort externalComfort = new ExternalComfort()
            {
                SimulationResult = command.SimulationResult,
                Typology = command.Typology,
            };

            // push objects to json file
            Push(new List<ExternalComfort>() { externalComfort }, actionConfig: config);

            // locate the Python file containing the simulation code
            string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "external_comfort.py");

            // run the calculation
            string cmdCommand = $"{m_environment.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";
            Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);

            // reload from Python results
            List<object> externalComfortPopulated = Pull(new FilterRequest(), actionConfig: config).ToList();

            // remove temporary file
            File.Delete(config.JsonFile.GetFullFileName());

            m_executeSuccess = true;
            return externalComfortPopulated;
        }
    }
}
