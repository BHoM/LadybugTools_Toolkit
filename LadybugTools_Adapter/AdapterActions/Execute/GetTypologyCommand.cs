/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2025, the respective contributors. All rights reserved.
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
        private List<object> RunCommand(GetTypologyCommand command, ActionConfig actionConfig)
        {
            LadybugConfig config;

            if (actionConfig?.GetType() == typeof(LadybugConfig))
            {
                config = (LadybugConfig)actionConfig;
                config.JsonFile = new FileSettings()
                {
                    FileName = $"LBTBHoM_Typologies.json",
                    Directory = Path.GetTempPath()
                };
            }
            else
            {
                config = new LadybugConfig()
                {
                    JsonFile = new FileSettings()
                    {
                        FileName = $"LBTBHoM_Typologies.json",
                        Directory = Path.GetTempPath()
                    }
                };
            }

            TimeSpan timeSinceLastUpdate = DateTime.Now - File.GetCreationTime(config.JsonFile.GetFullFileName());
            if (timeSinceLastUpdate.Days > config.CacheFileMaximumAge)
                File.Delete(config.JsonFile.GetFullFileName());

            if (!File.Exists(config.JsonFile.GetFullFileName()))
            {
                string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "get_typology.py");

                string cmdCommand = $"{m_environment.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";

                Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);
            }

            List<object> typologyObjects = Pull(new FilterRequest(), actionConfig: config).ToList();

            m_executeSuccess = true;
            return typologyObjects.Where(m => (m as Typology).Name.Contains(command.Filter)).ToList();
        }
    }
}
