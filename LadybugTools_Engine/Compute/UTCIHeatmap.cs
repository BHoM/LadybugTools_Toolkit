/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2024, the respective contributors. All rights reserved.
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

using BH.oM.Python;
using BH.oM.Base.Attributes;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using BH.oM.LadybugTools;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Convert an EPW file into a CSV and return the path to that CSV.")]
        [Input("epwFile", "An EPW file.")]
        [Input("includeAdditional", "Add sun position and psychrometric properties to the resultant CSV.")]
        [Output("csv", "The generated CSV file.")]
        public static string UTCIHeatmap(
            string epwFile, 
            EnergyMaterial groundMaterial, 
            EnergyMaterial shadeMaterial,
            Typology typology, 
            double evaporativeCooling = -1, 
            double windSpeedMultiplier = -1, 
            List<System.Drawing.Color> binColours = null, 
            string savePath = null
            )
        {
            if (epwFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(epwFile)} input cannot be null.");
                return null;
            }
            

            if (!System.IO.File.Exists(epwFile))
            {
                BH.Engine.Base.Compute.RecordError($"{epwFile} doesn't appear to exist!");
                return null;
            }

            PythonEnvironment env = InstallPythonEnv_LBT(true);

            epwFile = System.IO.Path.GetFullPath(epwFile);

            string script = Path.Combine(Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "utci_heatmap.py");

            // run the process
            string command = $"{env.Executable} {script} -e \"{epwFile}\" -gm \"{groundMaterial}\" -sm \"{shadeMaterial}\" -t \"{typology}\" -ec \"{evaporativeCooling}\" -ws \"{windSpeedMultiplier}\" -bc \"{binColours}\" -sp \"{savePath}\"";
            string result = Python.Compute.RunCommandStdout(command: command, hideWindows: true);

            return result;
        }
    }
}
