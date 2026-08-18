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
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        private List<object> RunCommand(HeatPlotCommand command, ActionConfig actionConfig)
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
                BH.Engine.Base.Compute.RecordError($"File '{command.EPWFile}' does not exist.");
                return null;
            }

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());


            //check if the colourmap is valid for user warning, but run with input anyway as the map could be defined separately.
            string colourMap = command.ColourMap;
            if (colourMap.ColourMapValidity())
                colourMap = colourMap.ToColourMap().FromColourMap();

            Dictionary<string, object> dict = new Dictionary<string, object>()
            {
                { "data_type_key", command.EPWKey.ToText() },
                { "colour_map", colourMap },
                { "save_path", command.OutputLocation.Replace('\\', '/') }
            };

            string json = dict.ToJson();

            // run the process
            List<string> args = new List<string>()
            {
                "-command", "plot/heatmap",
                "-e", epwFile.Replace('\\', '/'),
            };

            (string result, bool success) = ExecutePython(args, json);

            if (!success)
            {
                BH.Engine.Base.Compute.RecordError($"A python error occurred while running the command `{command.GetType().Name}`. Python output:\n{result}");
                m_executeSuccess = success;
                return new List<object>();
            }

            result = result.Split('\n').Last();

            try
            {
                PlotInformation info = (PlotInformation)BH.Engine.Serialiser.Convert.FromJson(result);
                m_executeSuccess = true;
                return new List<object>() { info };
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError(ex, $"An error occurred when deserialising the output from the script.\n Python output: {result}");
                m_executeSuccess = false;
                return new List<object>();
            }
        }
    }
}

