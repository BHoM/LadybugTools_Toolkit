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
using System.Collections.Generic;
using System.ComponentModel;
using BH.Engine.Python;
using BH.oM.Python;
using BH.oM.Reflection;
using BH.oM.Reflection.Attributes;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Method used to create the Python environment used to run all Python scripts within this toolkit.")]
        [Input("run", "Starts the installation of the toolkit if true. Stays idle otherwise.")]
        [Input("force", "Forces re-installation of the toolkits Python environment if true. This is used when the environment has been updated or package versions have changed.")]
        [MultiOutput(0, "pythonEnvironment", "The LadybugTools_Toolkit Python environment.")]
        [MultiOutput(1, "packages", "The list of successfully installed packages.")]
        public static Output<PythonEnvironment, List<string>> InstallPythonEnvironment(bool run = false, bool force = false)
        {
            if (!run)
                return new Output<PythonEnvironment, List<string>>();

            List<string> packages = new List<string>()
            {
                "lbt-dragonfly==0.7.255",
                "queenbee-local==0.3.13",
                "lbt-recipes==0.11.8",
                "pandas==1.2.4",
                "numpy==1.20.3",
                "matplotlib==3.4.2",
            };
            PythonEnvironment pythonEnvironment = Python.Create.PythonEnvironment("LadybugTools_Toolkit", "3.7.3", packages, force);

            return new Output<PythonEnvironment, List<string>>() { Item1 = pythonEnvironment, Item2 = pythonEnvironment.InstalledPackages() };
        }
    }
}
