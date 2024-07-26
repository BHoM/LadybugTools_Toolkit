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
        private List<object> RunCommand(SolarPanelTiltOptimisationCommand command, ActionConfig actionConfig)
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

            if (command.Azimuths < 3)
            {
                BH.Engine.Base.Compute.RecordError($"Azimuths must be greater than or equal to 1.");
                return null;
            }

            if (command.Altitudes < 3)
            {
                BH.Engine.Base.Compute.RecordError($"Altitudes must be greater than or equal to 1");
                return null;
            }

            if (command.GroundReflectance < 0 || command.GroundReflectance > 1)
            {
                BH.Engine.Base.Compute.RecordError($"Ground reflectance must be between 0 and 1 inclusive.");
                return null;
            }

            if (command.IrradianceType == IrradianceType.Undefined)
            {
                BH.Engine.Base.Compute.RecordError($"Please provide a valid Irradiance Type.");
                return null;
            }

            if (command.AnalysisPeriod == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(AnalysisPeriod)} input cannot be null.");
                return null;
            }

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "directional_solar_radiation.py");

            string cmdCommand = $"{m_environment.Executable} {script} -e \"{epwFile}\" -az {command.Azimuths} -al {command.Altitudes} -gr {command.GroundReflectance} -ir {command.IrradianceType} {(command.Isotropic ? "-iso" : "")} -t \"{command.Title}\" -ap \"{command.AnalysisPeriod.FromBHoM().Replace("\"", "\\\"")}\" -p \"{command.OutputLocation}\"";
            string result = Engine.Python.Compute.RunCommandStdout(cmdCommand, hideWindows: true);

            m_executeSuccess = true;
            return new List<object>() { result.Split('\n').Last() };
        }
    }
}
