using BH.Engine.Adapter;
using BH.oM.Adapter;
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
        private List<object> RunCommand(SunPathPlotCommand command, ActionConfig actionConfig)
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
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (command.SunSize < 1)
            {
                BH.Engine.Base.Compute.RecordError($"Sun size must be greater than or equal to 1.");
                return null;
            }

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "sunpath.py");

            //run the process
            string cmdCommand = $"{m_environment.Executable} {script} -e \"{epwFile}\" -s {command.SunSize} -ap \"{command.AnalysisPeriod.FromBHoM().Replace("\"", "\\\"")}\" -p \"{command.OutputLocation}\"";
            string result = Engine.Python.Compute.RunCommandStdout(cmdCommand, hideWindows: true);

            m_executeSuccess = true;
            return new List<object>() { result.Split('\n').Last() };
        }
    }
}
