using BH.Engine.Adapter;
using BH.oM.Adapter;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        private List<object> RunCommand(WindroseCommand command, ActionConfig actionConfig)
        {
            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (!System.IO.File.Exists(command.EPWFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"File '{command.EPWFile.GetFullFileName()}' does not exist.");
                return null;
            }

            if (command.AnalysisPeriod == null)
            {
                BH.Engine.Base.Compute.RecordError($"Please input a valid analysis period to run this command.");
                return null;
            }

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "windrose.py");

            //check if the colourmap is valid for user warning, but run with input anyway as the map could be defined separately.
            string colourMap = command.ColourMap;
            if (colourMap.ColourMapValidity())
                colourMap = colourMap.ToColourMap().FromColourMap();

            // run the process
            string cmdCommand = $"{m_environment.Executable} {script} -e \"{epwFile}\" -ap \"{command.AnalysisPeriod.FromBHoM().Replace("\"", "\\\"")}\" -cmap \"{colourMap}\" -bins \"{command.NumberOfDirectionBins}\" -p \"{command.OutputLocation}\"";
            string result = Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);

            m_executeSuccess = true;
            return new List<object> { result };
        }
    }
}
