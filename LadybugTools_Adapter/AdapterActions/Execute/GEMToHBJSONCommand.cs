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
    public partial class LadybugToolsAdapter
    {
        public List<object> RunCommand(GEMToHBJSONCommand command, ActionConfig actionConfig)
        {
            bool ignoreEPWCheck = false;
            if (actionConfig is LadybugConfig config)
                ignoreEPWCheck = config.SkipEPWCheck;

            if (command.GEMFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.GEMFile)} input cannot be null.");
                return null;
            }

            if (!Directory.Exists(command.OutputDirectory))
            {
                BH.Engine.Base.Compute.RecordError("The given output directory does not exist.");
                return null;
            }

            if (!System.IO.File.Exists(command.GEMFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"File '{command.GEMFile.GetFullFileName()}' does not exist.");
                return null;
            }

            List<string> args = new List<string>() { "--command", "gem_to_hbjson", "-g", command.GEMFile.GetFullFileName().Replace('\\', '/') };

            string result = "";
            bool success = true;

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
            }

            //as the file output is hard to verify by itself, check that no errors got output to stderr log
            success &= !result.Contains("Traceback (most recent call last):");

            if (!success)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred while converting the file to hbjson.\nPython output: {result}.");
                return new List<object>();
            }

            string outputFileName = Path.Combine(command.OutputDirectory, Path.GetFileNameWithoutExtension(command.GEMFile.FileName) + ".hbjson");
            File.WriteAllText(outputFileName, result);

            m_executeSuccess = success;
            return new List<object> { outputFileName };
        }
    }
}
