using BH.Engine.Adapter;
using BH.Engine.Base;
using BH.oM.Adapter;
using BH.oM.Base;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        private List<object> RunCommand(CompareEPWKeyPlotCommand command, ActionConfig actionConfig)
        {
            bool ignoreEPWCheck = false;
            if (actionConfig is LadybugConfig config)
                ignoreEPWCheck = config.SkipEPWCheck;

            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (!ignoreEPWCheck)
            {
                foreach (FileSettings epwFile1 in command.EPWCompareFiles)
                {
                    if (!System.IO.File.Exists(epwFile1.GetFullFileName()))
                    {
                        BH.Engine.Base.Compute.RecordError($"File '{epwFile1.GetFullFileName()}' does not exist.");
                        return null;
                    }
                }

                if (!System.IO.File.Exists(command.EPWFile.GetFullFileName()))
                {
                    BH.Engine.Base.Compute.RecordError($"File '{command.EPWFile.GetFullFileName()}' does not exist.");
                    return null;
                }
            }

            if (command.EPWKey == EPWKey.Undefined)
            {
                BH.Engine.Base.Compute.RecordError("Please provide a valid EPW key.");
                return null;
            }

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName()).Replace('\\', '/');
            List<string> epwFileList = command.EPWCompareFiles.Select(e => e.GetFullFileName().Replace('\\', '/')).ToList();

            // run the process
            List<string> args = new List<string>
            {
                "--command", "plot/diurnal",
                "-e", epwFile,
                "-dtk", command.EPWKey.ToText(),
                "-p", command.OutputLocation.Replace('\\', '/'),
                "-el" //append compare epw file list here
            };
            args.AddRange(epwFileList);

            if (command.PlotTimeseries)
                args.Add("-l");

            string result = "";
            bool success;

            if (m_httpClient != null)
            {
                Task<(string, bool)> task = Compute.SendHttp(m_httpClient, args);
                task.Wait();
                (result, success) = task.Result;
            }
            else
            {
                //if the server was not running or some other error happened, try running the python directly.
                string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom", "run_wrapped.py");
                string cmdCommand = $"{m_environment.Executable} {script} {args.Select(x => x.Contains(' ') || string.IsNullOrEmpty(x) ? '"' + x + '"' : x).Aggregate((a, b) => a + " " + b)}";

                result = Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true).Split('\n').Last();
            }

            try
            {
                CustomObject obj = (CustomObject)BH.Engine.Serialiser.Convert.FromJson(result);
                PlotInformation info = Convert.ToPlotInformation(obj, null);
                m_executeSuccess = true;
                return new List<object>() { info };
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError(ex, $"An error occurred when deserialising the output from the script.\n Python output: {result}");
                return new List<object>();
            }
        }
    }
}
