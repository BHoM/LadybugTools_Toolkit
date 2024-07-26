using BH.Engine.Adapter;
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
        private List<object> RunCommand(GetMaterialCommand command, ActionConfig actionConfig)
        {
            LadybugConfig config;

            if (actionConfig?.GetType() == typeof(LadybugConfig))
            {
                config = (LadybugConfig)actionConfig;
                config.JsonFile = new FileSettings()
                {
                    FileName = $"LBTBHoM_Materials.json",
                    Directory = Path.GetTempPath()
                };
            }
            else
            {
                config = new LadybugConfig()
                {
                    JsonFile = new FileSettings()
                    {
                        FileName = $"LBTBHoM_Materials.json",
                        Directory = Path.GetTempPath()
                    }
                };
            }

            TimeSpan timeSinceLastUpdate = DateTime.Now - File.GetCreationTime(config.JsonFile.GetFullFileName());
            if (timeSinceLastUpdate.Days > config.CacheFileMaximumAge)
                File.Delete(config.JsonFile.GetFullFileName());

            if (!File.Exists(config.JsonFile.GetFullFileName()))
            {
                string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "get_material.py");

                string cmdCommand = $"{m_environment.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";

                Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);
            }

            List<object> materialObjects = Pull(new FilterRequest(), actionConfig: config).ToList();

            m_executeSuccess = true;
            return materialObjects.Where(m => (m as IEnergyMaterialOpaque).Name.Contains(command.Filter)).ToList();
        }
    }
}
