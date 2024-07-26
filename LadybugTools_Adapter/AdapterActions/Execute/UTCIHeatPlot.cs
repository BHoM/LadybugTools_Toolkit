using BH.Engine.Adapter;
using BH.Engine.LadyBugTools;
using BH.Engine.Serialiser;
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
        private List<object> RunCommand(UTCIHeatPlotCommand command, ActionConfig actionConfig)
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

            if (command.GroundMaterial == null)
            {
                BH.Engine.Base.Compute.RecordError($"Please input a valid ground material to run this command.");
                return null;
            }

            if (command.ShadeMaterial == null)
            {
                BH.Engine.Base.Compute.RecordError($"Please input a valid shade material to run this command.");
                return null;
            }

            if (command.Typology == null)
            {
                BH.Engine.Base.Compute.RecordError($"Please input a valid Typology to run this command.");
            }

            if (!(command.BinColours.Count == 10 || command.BinColours.Count == 0))
            {
                BH.Engine.Base.Compute.RecordError($"When overriding bin colours 10 colours must be provided, but {command.BinColours.Count} colours were provided instead.");
                return null;
            }
            List<string> colours = command.BinColours.Select(x => x.ToHexCode()).ToList();

            string hexColours = $"[\"{string.Join("\",\"", colours)}\"]";
            if (hexColours == "[\"\"]")
                hexColours = "[]";

            Dictionary<string, string> inputObjects = new Dictionary<string, string>()
            {
                { "ground_material",  command.GroundMaterial.FromBHoM() },
                { "shade_material", command.ShadeMaterial.FromBHoM() },
                { "typology", command.Typology.FromBHoM() },
                { "bin_colours", hexColours }
            };

            string argFile = Path.GetTempFileName();
            File.WriteAllText(argFile, inputObjects.ToJson());

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "utci_heatmap.py");

            // run the process
            string cmdCommand = $"{m_environment.Executable} \"{script}\" -e \"{epwFile}\" -in \"{argFile}\" -ws \"{command.WindSpeedMultiplier}\" -sp \"{command.OutputLocation}\"";
            string result = "";

            try
            {
                result = Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError(ex, "An error occurred while running some python.");
            }
            finally
            {
                File.Delete(argFile);
            }

            m_executeSuccess = true;
            return new List<object> { result.Split('\n').Last() };
        }
    }
}
