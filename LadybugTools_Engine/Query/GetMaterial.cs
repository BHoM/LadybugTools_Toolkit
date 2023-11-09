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

using BH.oM.Base.Attributes;
using BH.oM.LadybugTools;
using System.Linq;
using System.Collections.Generic;
using System.ComponentModel;
using BH.oM.Python;
using System.IO;
using System;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Returns a list of materials from the Python Materials list.")]
        [Input("filter", "Text to filter the resultant list by. Filter applies to the material identifier. Leave blank to return all materials.")]
        [Output("materials", "A list of materials.")]
        public static List<IEnergyMaterialOpaque> GetMaterial(string filter = "")
        {
            PythonEnvironment env = Compute.InstallPythonEnv_LBT(true);

            string jsonFile = Path.Combine(Path.GetTempPath(), $"LBTBHoM_Materials_{DateTime.Now:yyyyMMdd}.json");

            if (!File.Exists(jsonFile))
            {
                string script = Path.Combine(Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "get_material.py");

                string command = $"{env.Executable} {script} -j \"{jsonFile}\"";

                Python.Compute.RunCommandStdout(command: command, hideWindows: true);
            }

            string jsonContent = File.ReadAllText(jsonFile);

            List<object> materials = (List<object>)BH.Engine.Serialiser.Convert.FromJsonArray(jsonContent);

            List<IEnergyMaterialOpaque> materialObjects = new List<IEnergyMaterialOpaque>();
            foreach (object materialObject in materials)
            {
                materialObjects.Add((IEnergyMaterialOpaque)materialObject);
            }

            return materialObjects.Where(m => m.Identifier.Contains(filter)).ToList();
        }
    }
}
