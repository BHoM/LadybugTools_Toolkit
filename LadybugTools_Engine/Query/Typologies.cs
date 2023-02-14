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
using System.Collections;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Return list of typologies from the Python predefined Typologies list.")]
        [Input("filter", "Text to filter the resultant list by.")]
        [Output("typologies", "A list of typologies.")]
        public static List<Typology> Typologies(string filter = "")
        {
            BH.oM.Python.PythonEnvironment env = Compute.InstallPythonEnv_LBT(true);

            string pythonScript = String.Join("\n", new List<string>()
            {
                "from ladybugtools_toolkit.external_comfort.typology import Typologies",
                "",
                "try:",
                $"    typologies = [typology.value.to_json() for typology in Typologies if \"{filter}\".lower() in typology.value.name.lower()]",
                "    typologies = f\"[{', '.join(typologies)}]\"",
                "    print(typologies)",
                "except Exception as exc:",
                "    print(exc)",
            });

            string result = env.RunPythonString(pythonScript).Trim();

            List<object> lbtTypologies = Serialiser.Convert.FromJsonArray(result).ToList();

            return lbtTypologies.Where(t => t as Typology != null).Select(t => (Typology)t).ToList();
        }
    }
}


