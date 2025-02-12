using BH.Engine.Adapter;
using BH.Engine.Base;
using BH.Engine.Serialiser;
using BH.oM.Adapter;
using BH.oM.Base;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter
    {
        private List<object> RunCommand(WalkabilityPlotCommand command, ActionConfig config)
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

            if (!Query.ValidateExternalComfort(command.ExternalComfort))
            {
                return null;
            }

            Dictionary<string, string> inputObjects = new Dictionary<string, string>()
            {
                { "external_comfort", command.ExternalComfort.FromBHoM() }
            };

            string argFile = Path.GetTempFileName();
            File.WriteAllText(argFile, inputObjects.ToJson());

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "walkability_heatmap.py");

            string returnFile = Path.GetTempFileName();

            // run the process
            string cmdCommand = $"{m_environment.Executable} \"{script}\" -in \"{argFile}\" -r \"{returnFile.Replace('\\', '/')}\" -sp \"{command.OutputLocation}\"";
            string result = Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);

            string resultFile = result.Split('\n').Last();

            if (!File.Exists(resultFile))
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred while running the command: {result}");
                File.Delete(returnFile);
                File.Delete(argFile);
                return new List<object>();
            }

            CustomObject obj = (CustomObject)BH.Engine.Serialiser.Convert.FromJson(System.IO.File.ReadAllText(returnFile));
            File.Delete(returnFile);
            File.Delete(argFile);
            PlotInformation info = Convert.ToPlotInformation(obj, new UTCIData());
            ExternalComfort ec = Convert.ToExternalComfort((obj.CustomData["external_comfort"] as CustomObject).CustomData);

            m_executeSuccess = true;
            return new List<object> { info, ec };
        }
    }
}
