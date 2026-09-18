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
using BH.Engine.Serialiser;
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

            Dictionary<string, object> dict = new Dictionary<string, object>()
            {
                { "gem_file", command.GEMFile.GetFullFileName().Replace('\\', '/') }
            };

            string json = dict.ToJson();

            List<string> args = new List<string>()
            {
                "--command", "gem_to_hbjson"
            };

            (string result, bool success) = ExecutePython(args, json);

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

