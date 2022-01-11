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
            oM.Python.Enums.PythonVersion version = oM.Python.Enums.PythonVersion.v3_7_3;

            List<PythonPackage> packages = new List<PythonPackage>()
            {
                new PythonPackage(){ Name="lbt-dragonfly", Version="0.7.255" },
                new PythonPackage(){ Name="queenbee-local", Version="0.3.13" },
                new PythonPackage(){ Name="lbt-recipes", Version="0.11.8" },
                new PythonPackage(){ Name="pandas", Version="1.2.4" },
                new PythonPackage(){ Name="numpy", Version="1.20.3" },
                new PythonPackage(){ Name="matplotlib", Version="3.4.2" },
            };

            PythonEnvironment pythonEnvironment = Create.PythonEnvironment(Query.ToolkitName(), version, packages);

            return Python.Compute.InstallToolkitPythonEnvironment(pythonEnvironment, force, run);
        }
    }
}

