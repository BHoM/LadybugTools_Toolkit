/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2021, the respective contributors. All rights reserved.
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
using BH.oM.Reflection.Attributes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using BH.oM.LadybugTools.Enums;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Forecast an EPW using the HadCM3 emissions scenario datasets.")]
        [Input("epwFile", "A path to the EPW file.")]
        [Input("emissionsScenario", "The emissions scenario to apply to the input EPW file.")]
        [Input("forecastYear", "The year to be forecast.")]
        [Output("forecastEPW", "A path to the forecast EPW file.")]
        public static string ForecastEPW(string epwFile, EmissionsScenario emissionScenario = EmissionsScenario.Undefined, ForecastYear forecastYear = ForecastYear.Undefined)
        {
            PythonEnvironment pythonEnvironment = Python.Query.LoadPythonEnvironment(Query.ToolkitName());
            if (!pythonEnvironment.IsInstalled())
            {
                BH.Engine.Reflection.Compute.RecordError($"Install the {Query.ToolkitName()} Python environment before running this method (using {Query.ToolkitName()}.Compute.InstallPythonEnvironment).");
                return null;
            }

            if (emissionScenario == EmissionsScenario.Undefined || forecastYear == ForecastYear.Undefined)
            {
                BH.Engine.Reflection.Compute.RecordError($"Set emissionsScenario and forecastYear to run this method.");
                return null;
            }

            string pythonScript = String.Join("\n", new List<string>() 
            {
                "import sys",
                $"sys.path.append('{pythonEnvironment.CodeDirectory()}')",
                "",
                "from epw import BH_EPW",
                "from enums import ForecastYear, EmissionsScenario",
                "",
                $"epw = BH_EPW(r'{epwFile}')",
                $"new_epw = epw.forecast(EmissionsScenario.{emissionScenario}, ForecastYear.{forecastYear}, True)",
                "print(new_epw.file_path)",
            });

            string output = Python.Compute.RunPythonString(pythonEnvironment, pythonScript).Trim();

            return output;
        }
    }
}
