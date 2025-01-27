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
using BH.Engine.LadybugTools;
using BH.oM.Adapter;
using BH.oM.Adapter.Commands;
using BH.oM.Base;
using BH.oM.Data.Requests;
using BH.oM.LadybugTools;
using BH.oM.Python;
using BH.Engine.Python;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using BH.Engine.Base;
using System.Drawing;
using BH.Engine.Serialiser;
using System.Reflection;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        bool m_executeSuccess = false;
        public override Output<List<object>, bool> Execute(IExecuteCommand command, ActionConfig actionConfig = null)
        {
            m_executeSuccess = false;
            Output<List<object>, bool> output = new Output<List<object>, bool>() { Item1 = new List<object>(), Item2 = false };

            List<object> temp = IRunCommand(command, actionConfig);

            output.Item1 = temp;
            output.Item2 = m_executeSuccess;

            return output;
        }

        /**************************************************/
        /* Public methods - Interface                     */
        /**************************************************/

        public List<object> IRunCommand(IExecuteCommand command, ActionConfig actionConfig)
        {
            if (command == null)
            {
                BH.Engine.Base.Compute.RecordError("Please input a valid Ladybug Command to execute.");
                return new List<object>();
            }

            //See .cs files in Execute folder for possible commands
            return RunCommand(command as dynamic, actionConfig);
        }

        /**************************************************/
        /* Private methods - Fallback                     */
        /**************************************************/

        private List<object> RunCommand(IExecuteCommand command, ActionConfig actionConfig)
        {
            BH.Engine.Base.Compute.RecordError($"The command {command.GetType().FullName} is not valid for the LadybugTools Adapter. Please use a LadybugCommand, or use the correct adapter for the input command.");
            return new List<object>();
        }
    }
}


