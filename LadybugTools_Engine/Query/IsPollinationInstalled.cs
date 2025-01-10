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

using BH.oM.Base.Attributes;
using BH.oM.LadybugTools;
using System.Linq;
using System.Collections.Generic;
using System.ComponentModel;
using BH.oM.Python;
using System.IO;
using System;
using System.Diagnostics;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Return True if Pollination is installed to the currently supported version.")]
        [Input("targetPollinationVersion", "The target Pollination version that BHoM currently supports.")]
        [Output("bool", "True if Pollination is installed to the currently supported version.")]
        public static bool IsPollinationInstalled(string targetPollinationVersion = "1.50.1")
        {
            // check if referenced Python is installed
            string referencedExecutable = System.Environment.GetFolderPath(System.Environment.SpecialFolder.ProgramFiles) + @"\ladybug_tools\python\python.exe";
            if (!File.Exists(referencedExecutable))
            {
                Base.Compute.RecordError($"Could not find referenced python executable at {referencedExecutable}. Please install Pollination version {targetPollinationVersion} and try again.");
                return false;
            }

            // obtain version of pollination installed
            string referencedUninstaller = System.Environment.GetFolderPath(System.Environment.SpecialFolder.ProgramFiles) + @"\pollination\uninstall.exe";
            FileVersionInfo versionInfo = FileVersionInfo.GetVersionInfo(referencedUninstaller);
            if (versionInfo.ProductVersion != targetPollinationVersion)
            {
                Base.Compute.RecordError($"Pollination version installed ({versionInfo.ProductVersion}) is not the same as the version required for this code to function correctly ({targetPollinationVersion}).");
                return false;
            }

            return true;
        }
    }
}

