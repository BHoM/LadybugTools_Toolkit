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

using BH.Engine.Python;
using BH.oM.Base;
using BH.oM.Base.Attributes;
using BH.oM.LadybugTools;
using BH.oM.Python;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Get a predefined ExternalComfortMaterial by it's name.")]
        [Input("materialName", "The name of a pre-defined material.")]
        [Output("ExternalComfortMaterial", "An ExternalComfortMaterial object.")]
        public static ExternalComfortMaterial ExternalComfortMaterial(string material)
        {
            BH.oM.Python.PythonEnvironment env = Compute.LadybugToolsToolkitPythonEnvironment(true);

            // get a list of materials that have been predefined in the Python code, as a custom object for each
            string pythonScript = string.Join("\n", new List<string>()
            {
                "from ladybugtools_toolkit.external_comfort import Materials",
                "from ladybugtools_toolkit.external_comfort.encoder.encoder import Encoder",
                "import json",
                "",
                "d = {}",
                "for material in Materials:",
                "    if material.value.__class__.__name__ == 'EnergyMaterial':",
                "        d[material.name] = material.value.to_dict()",
                "print(json.dumps(d, cls = Encoder))",
            });

            string output = env.RunCommandPythonString(pythonScript).Trim();
            CustomObject materials = Serialiser.Convert.FromJson(output) as CustomObject;
            List<string> materialIds = new List<string>();
            foreach (string materialIdentifiers in materials.CustomData.Keys)
            {
                materialIds.Add(materialIdentifiers);
            }

            if (!materialIds.Contains(material))
            {
                BH.Engine.Base.Compute.RecordError($"The typology given is not predefined in the Python source code. Please use one of [\n{String.Join(",\n    ", materialIds)}\n].");
            }

            // create the ECMaterial from the given string name of the Material
            CustomObject predefinedMaterial = (materials.CustomData[material] as CustomObject);


            BH.oM.LadybugTools.ExternalComfortMaterial ecMaterial = new ExternalComfortMaterial() {
                Identifier = (string)predefinedMaterial.CustomData["identifier"],
                Roughness = (Roughness)System.Enum.Parse(typeof(Roughness), (string)predefinedMaterial.CustomData["roughness"]),
                Thickness = (double)predefinedMaterial.CustomData["thickness"],
                Conductivity = (double)predefinedMaterial.CustomData["conductivity"],
                Density = (double)predefinedMaterial.CustomData["density"],
                SpecificHeat = (double)predefinedMaterial.CustomData["specific_heat"],
                ThermalAbsorptance = (double)predefinedMaterial.CustomData["thermal_absorptance"],
                SolarAbsorptance = (double)predefinedMaterial.CustomData["solar_absorptance"],
                VisibleAbsorptance = (double)predefinedMaterial.CustomData["visible_absorptance"],
            };

            // return the material
            return ecMaterial;
        }
    }
}

