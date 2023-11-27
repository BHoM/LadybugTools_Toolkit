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
using BH.oM.Adapter;
using BH.Engine.Adapter;
using BH.Adapter.LadybugTools;
using BH.oM.Data.Requests;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Returns a list of Typology objects from the Python predefined Typologies list.")]
        [Input("filter", "Text to filter the resultant list by. Filter applies to the typology identifier. Leave blank to return all typologies.")]
        [Output("typologies", "A list of Typology objects.")]
        public static List<Typology> GetTypology(string filter = "")
        {
            PythonEnvironment env = Compute.InstallPythonEnv_LBT(true);
            LadybugToolsAdapter adapter = new LadybugToolsAdapter();
            LadybugConfig config = new LadybugConfig()
            {
                JsonFile = new FileSettings()
                {
                    FileName = $"LBTBHoM_Typologies_{DateTime.Now:yyyyMMdd}.json",
                    Directory = Path.GetTempPath()
                }
            };

            if (!File.Exists(config.JsonFile.GetFullFileName()))
            {
                string script = Path.Combine(Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "get_typology.py");

                string command = $"{env.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";

                Python.Compute.RunCommandStdout(command: command, hideWindows: true);
            }

            List<Typology> typologyObjects = adapter.Pull(new FilterRequest(), actionConfig: config).Cast<Typology>().ToList();
                new List<Typology>();

            return typologyObjects.Where(m => m.Name.Contains(filter)).ToList();
        }
    }
}
