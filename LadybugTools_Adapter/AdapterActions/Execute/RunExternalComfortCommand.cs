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

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        private List<object> RunCommand(RunExternalComfortCommand command, ActionConfig actionConfig)
        {
            if (command.SimulationResult == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.SimulationResult)} input cannot be null.");
                return null;
            }

            if (command.Typology == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.Typology)} input cannot be null.");
                return null;
            }

            // construct the base object
            ExternalComfort externalComfort = new ExternalComfort()
            {
                SimulationResult = command.SimulationResult,
                Typology = command.Typology,
            };

            Dictionary<string, object> dict = new Dictionary<string, object>()
            {
                { "external_comfort", externalComfort }
            };

            string json = dict.ToJson();

            List<string> args = new List<string>()
            {
                "-c", "external_comfort"
            };

            (string result, bool success) = ExecutePython(args, json);

            if (!success)
            {
                BH.Engine.Base.Compute.RecordError($"A python error occurred while running the command `{command.GetType().Name}`. Python output:\n{result}");
                m_executeSuccess = success;
                return new List<object>();
            }

            string resultJson = result.Split('\n').Last();
            ExternalComfort ec = null;

            try
            {
                ec = (ExternalComfort)BH.Engine.Serialiser.Convert.FromJson(resultJson);
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError(ex, $"Could not deserialise python output into ExternalComfort. Python output:\n{result}");
                m_executeSuccess = false;
                return new List<object>();
            }

            return new List<object>() { ec };
        }
    }
}

