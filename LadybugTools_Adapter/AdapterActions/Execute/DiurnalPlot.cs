/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2024, the respective contributors. All rights reserved.
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
using BH.Engine.LadybugTools;
using BH.Engine.LadyBugTools;
using BH.oM.Adapter;
using BH.oM.LadybugTools;
using BH.oM.LadybugTools.ExecuteCommands;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        private List<object> RunCommand(DiurnalPlotCommand command, ActionConfig actionConfig)
        {
            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (!System.IO.File.Exists(command.EPWFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"File '{command.EPWFile}' does not exist.");
                return null;
            }

            if (command.Period == DiurnalPeriod.Undefined)
            {
                BH.Engine.Base.Compute.RecordError("Please provide a valid diurnal period.");
                return null;
            }

            if (command.EPWKey == EPWKey.Undefined)
            {
                BH.Engine.Base.Compute.RecordError("Please provide a valid EPW key.");
                return null;
            }

            command.Title = command.Title.SanitiseString();

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "diurnal.py");

            // run the process
            string cmdCommand = $"{m_environment.Executable} {script} -e \"{epwFile}\" -dtk \"{command.EPWKey.ToText()}\" -c \"{command.Colour.ToHexCode()}\" -t \"{command.Title}\" -ap \"{command.Period.ToString().ToLower()}\" -p \"{command.OutputLocation}\"";
            string result = Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);

            m_executeSuccess = true;
            return new List<object>() { result.Split('\n').Last() };
        }

        /**************************************************/

        private List<object> RunCommand(StackedDiurnalPlotCommand command, ActionConfig actionConfig)
        {
            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (!System.IO.File.Exists(command.EPWFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"File '{command.EPWFile}' does not exist.");
                return null;
            }

            if (command.Period == DiurnalPeriod.Undefined)
            {
                BH.Engine.Base.Compute.RecordError("Please provide a valid diurnal period.");
                return null;
            }

            if (command.EPWKeys.Count != command.Colours.Count)
            {
                BH.Engine.Base.Compute.RecordError($"The number of EPW keys must be the same as the number of colours: Keys: {command.EPWKeys.Count}, Colours: {command.Colours.Count}.");
                return null;
            }

            List<KeyValuePair<EPWKey, Color>> keyColours = new List<KeyValuePair<EPWKey, Color>>();

            int keyIndex = 0;
            foreach (EPWKey key in command.EPWKeys)
            {
                if (key == EPWKey.Undefined)
                {
                    BH.Engine.Base.Compute.RecordWarning($"The EPW key at index {keyIndex} was undefined, and has been ignored on plotting.");
                    keyIndex++;
                    continue;
                }

                keyColours.Add(new KeyValuePair<EPWKey, Color>(key, command.Colours[keyIndex]));
                keyIndex++;
            }

            if (keyColours.Count < 2)
            {
                BH.Engine.Base.Compute.RecordError($"The number of keys after filtering out invalid keys was {keyColours.Count}. Please provide at least 2 valid keys.");
                return null;
            }

            string keys = $"\"{string.Join("\" \"", keyColours.Select(x => x.Key.ToText()))}\"";
            string colours = $"\"{string.Join("\" \"", keyColours.Select(x => x.Value.ToHexCode()))}\"";

            command.Title = command.Title.SanitiseString();

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "stacked_diurnal.py");

            string cmdCommand = $"{m_environment.Executable} {script} -e \"{epwFile}\" -dtks {keys} -c {colours} -t \"{command.Title}\" -ap \"{command.Period.ToString().ToLower()}\" -p \"{command.OutputLocation}\"";
            string result = Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);

            m_executeSuccess = true;
            return new List<object>() { result.Split('\n').Last() };
        }
    }
}
