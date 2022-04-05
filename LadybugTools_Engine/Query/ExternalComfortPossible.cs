/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2022, the respective contributors. All rights reserved.
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

using BH.oM.Base.Attributes;
using System.Collections.Generic;
using System.IO;
using System.ComponentModel;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Determine whether the External Comfort process is possible.")]
        [Output("bool", "True if External Comfort simulation is possible.")]
        public static bool ExternalComfortPossible()
        {
            string username = System.Environment.UserName;

            string ladybugToolsFolder = $"C:/Users/{username}/ladybug_tools";
            string defaultSimulationFolder = $"C:/Users/{username}/simulation";

            string openstudioPath = System.IO.Path.Combine(ladybugToolsFolder, "openstudio/bin");
            string energyplusPath = System.IO.Path.Combine(ladybugToolsFolder, "openstudio/EnergyPlus");
            string honeybeeOpenstudioGemPath = System.IO.Path.Combine(ladybugToolsFolder, "resources/measures/honeybee_openstudio_gem/lib");
            string radiancePath = System.IO.Path.Combine(ladybugToolsFolder, "radiance");

            foreach (string path in new List<string>() {
                ladybugToolsFolder,
                defaultSimulationFolder,
                openstudioPath,
                energyplusPath,
                honeybeeOpenstudioGemPath,
                radiancePath,
            })
            {
                if (!Directory.Exists(path))
                {
                    BH.Engine.Base.Compute.RecordError($"Install Ladybug Tools using the instructions found https://www.food4rhino.com/en/app/ladybug-tools in order to be able to run this method.");
                    return false;
                }
            }

            return true;
        }
    }
}

