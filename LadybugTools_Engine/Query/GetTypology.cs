/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2023, the respective contributors. All rights reserved.
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

using BH.Engine.Python;
using BH.oM.Base.Attributes;
using BH.oM.LadybugTools;
using System.Linq;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using BH.oM.Python;
using BH.Engine.Base;
using BH.oM.Base;
using Eto.Forms;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Returns a list of Typology objects from the Python predefined Typologies list.")]
        [Input("filter", "Text to filter the resultant list by. Filter applies to the typology name. Leave blank to return all typologies.")]
        [Output("typologies", "A list of Typology objects.")]
        [PreviousVersion("6.1", "BH.Engine.LadybugTools.Query.GetTypology(BH.oM.LadybugTools.Typologies)")]
        public static List<Typology> GetTypology(string filter = "")
        {
            if (string.IsNullOrEmpty(filter))
                filter = "";

            PythonEnvironment env = Python.Query.VirtualEnv(ToolkitName());

            string pythonScript = string.Join("\n", new List<string>()
            {
                "import traceback",
                "",
                "try:",
                "    from ladybugtools_toolkit.external_comfort.typology import Typologies",
                $"    typologies = [typology.value.to_json() for typology in Typologies if \"{filter}\".lower() in typology.value.name.lower()]",
                "    typologies = f\"[{', '.join(typologies)}]\"",
                "    print(typologies)",
                "except Exception as exc:",
                "    print(traceback.format_exc())",
            });

            string result = env.RunPythonString(pythonScript).Trim();

            List<object> lbtTypologies = Serialiser.Convert.FromJsonArray(result).ToList();

            List<Typology> typologies = lbtTypologies.Select(t => t as Typology).Where(t => t != null).ToList();
            
            return typologies;
        }
    }
}


