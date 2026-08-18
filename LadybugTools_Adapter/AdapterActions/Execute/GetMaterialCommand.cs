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
using BH.oM.Data.Requests;
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
        private List<object> RunCommand(GetMaterialCommand command, ActionConfig actionConfig)
        {
            LadybugConfig config = (actionConfig as LadybugConfig) ?? new LadybugConfig()
            {
                JsonFile = new FileSettings()
                {
                    FileName = "LBTBHoM_Materials.json",
                    Directory = Path.GetTempPath()
                }
            };

            if (File.Exists(config.JsonFile.GetFullFileName()))
            {
                TimeSpan timeSinceLastUpdate = DateTime.Now - File.GetCreationTime(config.JsonFile.GetFullFileName());
                if (timeSinceLastUpdate.Days >= config.CacheFileMaximumAge)
                    File.Delete(config.JsonFile.GetFullFileName());
            }

            // run the process
            if (!File.Exists(config.JsonFile.GetFullFileName()))
            {
                Dictionary<string, object> dict = new Dictionary<string, object>()
                {
                    { "json_file", config.JsonFile.GetFullFileName().Replace('\\', '/') }
                };

                string json = dict.ToJson();

                List<string> args = new List<string>()
                {
                    "--command", "get_material"
                };

                (string result, bool success) = ExecutePython(args, json);

                if (!success)
                {
                    BH.Engine.Base.Compute.RecordError($"A python error occurred while getting materials. Python output:\n{result}");
                    m_executeSuccess = success;
                    return new List<object>();
                }

                result = result.Split('\n').Last();
                File.WriteAllText(config.JsonFile.GetFullFileName(), result);
            }

            List<object> materialObjects = Pull(new FilterRequest(), actionConfig: config).ToList();

            m_executeSuccess = true;
            return materialObjects.Where(m => (m as IEnergyMaterialOpaque).Name.Contains(command.Filter)).ToList();
        }
    }
}

