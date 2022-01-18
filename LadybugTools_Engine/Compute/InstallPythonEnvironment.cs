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
using System.Collections.Generic;
using System.ComponentModel;

using BH.Engine.Python;
using BH.oM.Python;
using BH.oM.Base.Attributes;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("LadybugTools_Toolkit\nMethod used to create the Python environment used to run all Python scripts within this toolkit.")]
        [Input("run", "Starts the installation of the toolkit if true. Stays idle otherwise.")]
        [Input("force", "Forces re-installation of the toolkits Python environment if true. This is used when the environment has been updated or package versions have changed.")]
        [Output("pythonEnvironment", "The LadybugTools_Toolkit Python environment.")]
        public static PythonEnvironment InstallPythonEnvironment(bool run = false, bool force = false)
        {
            oM.Python.Enums.PythonVersion version = oM.Python.Enums.PythonVersion.v3_7_9;
            
            List<PythonPackage> packages = new List<PythonPackage>()
            {
                new PythonPackage(){ Name="ipykernel", Version="6.7.0" },
                new PythonPackage(){ Name="matplotlib", Version="3.5.1" },
                new PythonPackage(){ Name="numpy", Version="1.21.5" },
                new PythonPackage(){ Name="pandas", Version="1.3.5" },
                new PythonPackage(){ Name="scipy", Version="1.7.3" },
                new PythonPackage(){ Name="fortranformat", Version="1.1.1" },
                

                // NOTE: Ladybug code version below based on V1.4 Grasshopper release
                new PythonPackage(){ Name="dragonfly-core", Version="1.31.0" },
                new PythonPackage(){ Name="dragonfly-energy", Version="1.15.58" },
                new PythonPackage(){ Name="dragonfly-schema", Version="1.6.59" },
                new PythonPackage(){ Name="dragonfly-uwg", Version="0.5.145" },
                new PythonPackage(){ Name="honeybee-core", Version="1.49.5" },
                new PythonPackage(){ Name="honeybee-energy", Version="1.85.3" },
                new PythonPackage(){ Name="honeybee-energy-standards", Version="2.2.1" },
                new PythonPackage(){ Name="honeybee-radiance", Version="1.50.32" },
                new PythonPackage(){ Name="honeybee-radiance-command", Version="1.20.3" },
                new PythonPackage(){ Name="honeybee-radiance-folder", Version="2.8.0" },
                new PythonPackage(){ Name="honeybee-schema", Version="1.47.2" },
                new PythonPackage(){ Name="honeybee-standards", Version="2.0.5" },
                new PythonPackage(){ Name="ladybug-comfort", Version="0.13.8" },
                new PythonPackage(){ Name="ladybug-core", Version="0.39.37" },
                new PythonPackage(){ Name="ladybug-geometry", Version="1.23.26" },
                new PythonPackage(){ Name="ladybug-geometry-polyskel", Version="1.3.47" },
                new PythonPackage(){ Name="ladybug-rhino", Version="1.33.3" },
                new PythonPackage(){ Name="lbt-dragonfly", Version="0.8.367" },
                new PythonPackage(){ Name="lbt-honeybee", Version="0.5.270" },
                new PythonPackage(){ Name="lbt-ladybug", Version="0.25.111" },
                new PythonPackage(){ Name="lbt-recipes", Version="0.19.4" },
                new PythonPackage(){ Name="pollination-handlers", Version="0.8.4" },
                new PythonPackage(){ Name="queenbee", Version="1.26.5" },
                new PythonPackage(){ Name="queenbee-local", Version="0.3.18" },
                new PythonPackage(){ Name="uwg", Version="5.8.9" },
            };

            PythonEnvironment pythonEnvironment = Create.PythonEnvironment(Query.ToolkitName(), version, packages);

            return Python.Compute.InstallToolkitPythonEnvironment(pythonEnvironment, force, run);
        }
    }
}

