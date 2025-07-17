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
using BH.Engine.Base;
using BH.Engine.LadybugTools;
using BH.oM.Adapter;
using BH.oM.Base;
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
        private List<object> RunCommand(DiurnalPlotCommand command, ActionConfig actionConfig)
        {
            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            /*if (!System.IO.File.Exists(command.EPWFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"File '{command.EPWFile.GetFullFileName()}' does not exist.");
                return null;
            }*/

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

            //string returnFile = Path.GetTempFileName();

            // run the process
            List<string> args = new List<string>() { "--command", "plot/diurnal", "-e", epwFile.Replace('\\', '/'), "-dtk", command.EPWKey.ToText(), "--colour", command.Colour.ToHexCode(), "-t", command.Title, "-ap", command.Period.ToString().ToLower(), "-p", command.OutputLocation.Replace('\\', '/') };

            string result = "";
            bool success;
            if (m_useHost)
                (result, success) = Compute.RunLBTClientSocket(args, m_address, m_port);
            else
                success = false;

            if (!success)
            {
                //if the server was not running or some other error happened, try running the python directly.
                string script = Path.Combine(Engine.LadybugTools.Query.PythonCodeDirectory(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom", "run_wrapped.py");
                string cmdCommand = $"{m_environment.Executable} {script} {args.Select(x => x.Contains(' ') ? '"' + x + '"' : x).Aggregate((a, b) => a + " " + b)}";

                result = Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true).Split('\n').Last();
            }

            try
            {
                CustomObject obj = (CustomObject)BH.Engine.Serialiser.Convert.FromJson(result);
                PlotInformation info = Convert.ToPlotInformation(obj, new CollectionData());
                m_executeSuccess = true;
                return new List<object>() { info };
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError(ex, "An error occurred when deserialising the output from the script.");
                return new List<object>();
            }
        }
    }
}
