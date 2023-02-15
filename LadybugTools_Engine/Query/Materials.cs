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
using BH.oM.Python;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Returns a list of materials from the Python Materials list.")]
        [Input("filter", "Text to filter the resultant list by. Filter applies to the material identifier. Leave blank to return all materials.")]
        [Input("env", "Optional input to provide the Python environment. If no environment is provided, one will be created instead. However, this method will usually run much faster if the environment is provided to this input.")]
        [Output("materials", "A list of materials.")]
        public static List<ILBTMaterial> Materials(string filter = "")
        {
            PythonEnvironment env = Python.Query.VirtualEnv(ToolkitName());

            string pythonScript = String.Join("\n", new List<string>()
            {
                "from ladybugtools_toolkit.external_comfort.material import Materials",
                "",
                "try:",
                $"    materials = [material.value.to_json() for material in Materials if \"{filter}\".lower() in material.value.identifier.lower()]",
                "    materials = f\"[{', '.join(materials)}]\"",
                "    print(materials)",
                "except Exception as exc:",
                "    print(exc)",
            });
            
            string result = env.RunPythonString(pythonScript).Trim();

            List<object> lbtMaterials = Serialiser.Convert.FromJsonArray(result).ToList();

            return lbtMaterials.Select(m => m as ILBTMaterial).Where(m => m != null).ToList();
        }
    }
}


