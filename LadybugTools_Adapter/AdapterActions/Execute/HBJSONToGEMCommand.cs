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
        public List<object> RunCommand(HBJSONToGEMCommand command, ActionConfig actionConfig)
        {
            bool ignoreEPWCheck = false;
            if (actionConfig is LadybugConfig config)
                ignoreEPWCheck = config.SkipEPWCheck;

            if (command.HBJSONFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.HBJSONFile)} input cannot be null.");
                return null;
            }

            if (!Directory.Exists(command.OutputDirectory))
            {
                BH.Engine.Base.Compute.RecordError("The given output directory does not exist.");
                return null;
            }

            if (!System.IO.File.Exists(command.HBJSONFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"File '{command.HBJSONFile.GetFullFileName()}' does not exist.");
                return null;
            }

            List<string> args = new List<string>() { "--command", "hbjson_to_gem", "-j", command.HBJSONFile.GetFullFileName().Replace('\\', '/') };

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
                BH.Engine.Base.Compute.RecordError($"An error occurred while converting the file to gem.\nPython output: {result}.");
                return new List<object>();
            }

            string outputFileName = Path.Combine(command.OutputDirectory, Path.GetFileNameWithoutExtension(command.HBJSONFile.FileName) + ".gem");
            File.WriteAllText(outputFileName, result);

            m_executeSuccess = success;
            return new List<object> { outputFileName };
        }
    }
}

