/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2026, the respective contributors. All rights reserved.
 *
 * Each contributor holds copyright over their respective contributions.
 * The project versioning (Git) records all such contribution source information.
 *                                           
 *                                                                              
 * The BHoM is free software: you can redistribute it and/or modify         
 * it under the terms of the GNU Lesser General Public License as published by  
 * the Free Software Foundation, either version 3.0 of the License, or          
 * (at your option) any later version.                                          
 *                                                                              
 * The BHoM is distributed in the hope that it will be useful,              
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                 
 * GNU Lesser General Public License for more details.                          
 *                                                                            
 * You should have received a copy of the GNU Lesser General Public License     
 * along with this code. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>.      
 */

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
using System.Threading.Tasks;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter
    {
        private List<object> RunCommand(WalkabilityPlotCommand command, ActionConfig actionConfig)
        {
            bool ignoreEPWCheck = false;
            if (actionConfig is LadybugConfig config)
                ignoreEPWCheck = config.SkipEPWCheck;

            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (!ignoreEPWCheck & !System.IO.File.Exists(command.EPWFile.GetFullFileName()))
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

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            // run the process
            List<string> args = new List<string>() { "-command", "plot/walkability_heatmap", "-e", epwFile.Replace('\\', '/'), "-sp", command.OutputLocation.Replace('\\', '/') };

            string result = "";
            bool success;

            if (m_httpClient != null)
            {
                Task<(string, bool)> task = Compute.SendHttp(m_httpClient, args, inputObjects.ToJson());
                task.Wait();
                (result, success) = task.Result;
            }
            else
            {
                //if the server was not running or some other error happened, try running the python directly.
                string argFile = Path.GetTempFileName();
                File.WriteAllText(argFile, inputObjects.ToJson());
                args.Add("-in");
                args.Add(argFile);
                string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom", "run_wrapped.py");
                string cmdCommand = $"{m_environment.Executable} {script} {args.Select(x => x.Contains(' ') || string.IsNullOrEmpty(x) ? '"' + x + '"' : x).Aggregate((a, b) => a + " " + b)}";

                result = Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true).Split('\n').Last();
                System.IO.File.Delete(argFile);
            }

            try
            {
                CustomObject obj = (CustomObject)BH.Engine.Serialiser.Convert.FromJson(result);
                PlotInformation info = Convert.ToPlotInformation(obj, new UTCIData());
                ExternalComfort ec = Convert.ToExternalComfort((obj.CustomData["external_comfort"] as CustomObject).CustomData);
                m_executeSuccess = true;
                return new List<object>() { info, ec };
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError(ex, $"An error occurred when deserialising the output from the script.\n Python output: {result}");
                return new List<object>();
            }
        }
    }
}

