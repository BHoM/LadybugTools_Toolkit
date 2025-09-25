using BH.Engine.Adapter;
using BH.oM.Adapter;
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
        private List<object> RunCommand(EPWToCSVCommand command, ActionConfig actionConfig)
        {
            bool ignoreEPWCheck = false;
            if (actionConfig is LadybugConfig config)
                ignoreEPWCheck = config.SkipEPWCheck;

            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (!Directory.Exists(command.OutputDirectory))
            {
                BH.Engine.Base.Compute.RecordError("The given output directory does not exist.");
                return null;
            }

            if (!ignoreEPWCheck & !System.IO.File.Exists(command.EPWFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"File '{command.EPWFile.GetFullFileName()}' does not exist.");
                return null;
            }

            List<string> args = new List<string>() { "--command", "epw_to_csv", "-e", command.EPWFile.GetFullFileName().Replace('\\', '/'), "-a", command.IncludeAdditionalCalculated.ToString() };

            string result = "";
            bool success;

            if (m_httpClient != null)
            {
                Task<(string, bool)> task = Compute.SendHttp(m_httpClient, args);
                task.Wait();
                (result, success) = task.Result; //in this case, result is the text of the csv file.
            }
            else
            {
                //if the server was not running or some other error happened, try running the python directly.
                string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom", "run_wrapped.py");
                string cmdCommand = $"{m_environment.Executable} {script} {args.Select(x => x.Contains(' ') || string.IsNullOrEmpty(x) ? '"' + x + '"' : x).Aggregate((a, b) => a + " " + b)}";

                result = Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);
                success = result != "";
            }

            if (!success)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred while converting the file to csv.\nPython output: {result}.");
                return new List<object>();
            }

            string outputFileName = Path.Combine(command.OutputDirectory, Path.GetFileNameWithoutExtension(command.EPWFile.FileName) + ".csv");
            File.WriteAllText(outputFileName, result);

            m_executeSuccess = success;
            return new List<object> { outputFileName };
        }
    }
}
