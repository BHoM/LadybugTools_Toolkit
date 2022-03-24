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
using BH.oM.Ladybug;
using BH.oM.Python;
using BH.oM.Base.Attributes;
using System;
using System.Collections.Generic;
using System.ComponentModel;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Run an Openfield simulation and return results.")]
        [Input("epw", "An EPW file.")]
        [Input("groundMaterial", "A pre-defined ground material.")]
        [Input("shadeMaterial", "A pre-defined shade material.")]
        [Output("openfield", "An openfield object containing simulation results.")]
        public static string Openfield(string epw, string groundMaterial = "ASPHALT", string shadeMaterial = "FABRIC")
        {
            PythonEnvironment pythonEnvironment = Python.Query.LoadPythonEnvironment(Query.ToolkitName());
            if (!pythonEnvironment.IsInstalled())
            {
                BH.Engine.Base.Compute.RecordError($"Install the {Query.ToolkitName()} Python environment before running this method (using {Query.ToolkitName()}.Compute.InstallPythonEnvironment).");
                return null;
            }

            string pythonScript = String.Join("\n", new List<string>() 
            {
                "import sys",
                $"sys.path.insert(0, '{pythonEnvironment.CodeDirectory()}')",
                "",
                "from external_comfort.openfield import EPW, Openfield",
                "",
                $"epw = EPW(r'{epw}')",
                $"of = Openfield(epw, '{groundMaterial}', '{shadeMaterial}', run=True)",
                "",
                "d = {}",
                $"d['EPW'] = r'{epw}'",
                $"d['GroundMaterial'] = '{groundMaterial}'",
                $"d['ShadeMaterial'] = '{shadeMaterial}'",

                "d['ShadedGroundSurfaceTemperature'] = of.shaded_below_temperature.values",
                "d['ShadeSurfaceTemperature'] = of.shaded_above_temperature.values",
                "d['ShadedDirectRadiation'] = of.shaded_direct_radiation.values",
                "d['ShadedDiffuseRadiation'] = of.shaded_diffuse_radiation.values",
                "d['ShadedLongwaveRadiantTemperature'] = of.shaded_longwave_mean_radiant_temperature.values",
                "d['ShadedMeanRadiantTemperature'] = of.shaded_mean_radiant_temperature.values",
                "d['ShadedUniversalThermalClimateIndex'] = None",

                "d['UnshadedGroundSurfaceTemperature'] = of.unshaded_below_temperature.values",
                "d['SkyTemperature'] = of.unshaded_above_temperature.values",
                "d['UnshadedDirectRadiation'] = of.unshaded_direct_radiation.values",
                "d['UnshadedDiffuseRadiation'] = of.unshaded_diffuse_radiation.values",
                "d['UnshadedLongwaveRadiantTemperature'] = of.unshaded_longwave_mean_radiant_temperature.values",
                "d['UnshadedMeanRadiantTemperature'] = of.unshaded_mean_radiant_temperature.values",
                "d['UnshadedUniversalThermalClimateIndex'] = None",

                "print(d)",
            });

            string output = Python.Compute.RunPythonString(pythonEnvironment, pythonScript).Trim();

            return output;
        }
    }
}

