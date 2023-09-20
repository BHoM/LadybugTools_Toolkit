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
using BH.oM.Python;
using BH.oM.Base.Attributes;

using System.Collections.Generic;
using System.ComponentModel;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Convert an EPW file into a CSV and return the path to that CSV.")]
        [Input("epwFile", "An EPW file.")]
        [Input("includeAdditional", "Add sun position and psychrometric properties to the resultant CSV.")]
        [Output("csv", "The generated CSV file.")]
        public static string EPWtoCSV(string epwFile, bool includeAdditional = false)
        {
            if (epwFile == null)
            {
                BH.Engine.Base.Compute.RecordError("epwFile input cannot be null.");
                return null;
            }

            if (!System.IO.File.Exists(epwFile))
            {
                BH.Engine.Base.Compute.RecordError($"{epwFile} doesn't appear to exist!");
                return null;
            }

            PythonEnvironment env = InstallPythonEnv_LBT(true);
            string additionalProperties = includeAdditional ? "True" : "False";

            string pythonScript = string.Join("\n", new List<string>()
            {
                "import traceback",
                "from pathlib import Path",
                "from ladybug.epw import EPW",
                "",
                $"epw_path = Path(r'{epwFile}')",
                "csv_path = epw_path.with_suffix('.csv')",
                "try:",
                "    from ladybugtools_toolkit.ladybug_extension.epw import epw_to_dataframe",
                $"    epw_to_dataframe(EPW(epw_path.as_posix()), include_additional={additionalProperties}).to_csv(csv_path.as_posix())",
                "    print(csv_path)",
                "except Exception as exc:",
                "    print(traceback.format_exc())",
            });

            return env.RunPythonString(pythonScript).Trim();
        }
    }
}