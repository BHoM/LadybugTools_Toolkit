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
using BH.oM.Python;
using BH.oM.Base.Attributes;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using BH.oM.LadybugTools;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Convert an EPW file into a time-indexed CSV version.")]
        [Input("epwFile", "An EPW file.")]
        [Input("emissionsScenario", "The emsissions scenario with which to forecast.")]
        [Input("forecastYear", "The future year in which to forecast.")]
        [Output("forecastEPW", "The input EPW forecast using the HadCM3 A2 model.")]
        public static string HadCM3ForecastEPW(string epwFile, HadCM3EmissionsScenario emissionsScenario, HadCM3ForecastYear forecastYear)
        {
            if (epwFile == null)
            {
                BH.Engine.Base.Compute.RecordError("epwFile input cannot be null.");
                return null;
            }

            if (emissionsScenario == HadCM3EmissionsScenario.Undefined)
            {
                BH.Engine.Base.Compute.RecordError("emissionsScenario cannot be Undefined.");
                return null;
            }

            if (forecastYear == HadCM3ForecastYear.Undefined)
            {
                BH.Engine.Base.Compute.RecordError("forecastYear cannot be Undefined.");
                return null;
            }

            if (!System.IO.File.Exists(epwFile))
            {
                BH.Engine.Base.Compute.RecordError($"{epwFile} doesn't appear to exist!");
                return null;
            }

            // get Python environment for number crunching
            BH.oM.Python.PythonEnvironment env = Compute.InstallPythonEnv_LBT(true);

            // run procedure to forecast input EPW file
            string pythonScript = string.Join("\n", new List<string>()
            {
                "# IMPORTS #",
                "from pathlib import Path",
                "from ladybug.epw import EPW",
                "from ladybugtools_toolkit.forecast.ccwwg import forecast_epw",
                "",
                "# INPUT PREPARATION #",
                $"epw_path = Path(r'{epwFile}')",
                $"emissions_scenario = '{emissionsScenario}'",
                $"forecast_year = int({forecastYear.ToString().Replace("_", "")})",
                "epw = EPW(epw_path)",
                "",
                "# PROCESS #",
                "forecast_epw_path = epw_path.parent / f'{epw_path.stem}__HadCM3_{emissions_scenario}_{forecast_year}{epw_path.suffix}'",
                "forecast_epw = forecast_epw(epw, emissions_scenario, forecast_year)",
                "forecast_epw.save(forecast_epw_path.as_posix())",
                "",
                "# RETURN #",
                "try:",
                "    print(forecast_epw_path.as_posix())",
                "except Exception as exc:",
                "    print(exc)",
            });

            string output = env.RunPythonString(pythonScript).Trim();

            return output;
        }
    }
}
