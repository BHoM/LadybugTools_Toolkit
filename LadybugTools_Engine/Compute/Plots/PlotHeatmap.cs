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
using BH.oM.LadybugTools;

using System.ComponentModel;
using System.IO;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Produces a heatmap from data in an epw file.")]
        [Input("epwFile", "An EPW file.")]
        [Input("epwKey", "Key representing data from the epw to plot.")]
        [Input("colourMap", "Matplotlib colour map. Corresponds to a value for 'cmap' in matplotlib. See https://matplotlib.org/stable/users/explain/colors/colormaps.html for examples of valid keys. Defaults to 'viridis'.")]
        [Input("outputLocation", "Full path (including file name) on where to save the plot. If left blank, a base64 string representation of the image will be returned instead.")]
        [Output("plot", "Either the path to the image, or the string representation of the image, depending on outputLocation input.")]
        public static string PlotHeatmap(string epwFile, EpwKey epwKey, string colourMap = "viridis", string outputLocation = null)
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

            string script = Path.Combine(Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "heatmap.py");

            // run the process
            string command = $"{env.Executable} {script} -e \"{epwFile}\" -dtk \"{epwKey}\" -cmap \"{colourMap}\" -p \"{outputLocation}\"";
            string result = Python.Compute.RunCommandStdout(command: command, hideWindows: true);

            return result;
        }
    }
}
