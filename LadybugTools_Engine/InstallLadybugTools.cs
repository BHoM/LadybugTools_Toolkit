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

using BH.oM.Reflection;
using BH.oM.Reflection.Attributes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        /*************************************/
        /**** Public Methods              ****/
        /*************************************/

        [Description("Install the Ladybug Python code supporting this toolkit.")]
        [Input("run", "Starts the installation of the toolkit if true. Stays idle otherwise.")]
        [Input("force", "If the toolkit is already installed it forces a reinstall of all the packages. It does not force a reinstall of Python.")]
        [MultiOutput(0, "success", "True if installation is successful, false otherwise.")]
        [MultiOutput(1, "packages", "The list of successfully installed packages.")]
        public static Output<bool, List<string>> InstallLadybugToolsToolkit(bool run = false, bool force = false)
        {
            bool success = false;
            List<string> installedPackages = new List<string>();

            if (!run)
                return new Output<bool, List<string>> { Item1 = success, Item2 = installedPackages };

            // install the python toolkit if necessary
            if (!BH.Engine.Python.Query.IsPythonInstalled())
                BH.Engine.Python.Compute.InstallPythonToolkit(run, force);

            // install from specific package/s
            Dictionary<string, string> packages = new Dictionary<string, string>()
            {
                { "lbt-dragonfly", "0.8.155" },
                { "queenbee-local", "0.3.16" },
                { "lbt-recipes", "0.17.14" },
                { "virtualenv", "20.8.1"},
                { "pandas", "1.3.3" },
                { "numpy", "1.21.2" },
                { "matplotlib", "3.4.3" },
                { "ipykernel", "6.4.1" },
                { "black", "21.9b0" }
            };

            Console.WriteLine("Installing required packages...");
            foreach (KeyValuePair<string, string> kvp in packages)
            {
                Python.Compute.PipInstall(kvp.Key);  // Version info being ignored for now to prevent dependency issues between contained packages, letting Pip determine which sub-packages work with each other
            }

            // TODO - add checks to see if modules installed correctly here! For now the list of packages are just being returned, which is dangerous. Also, LB packages are installed differently to other Python packages, meaning the usual checks don't work to see if they exist. Also a success/failure flag is retured by Pip which should be used rather than seeing if the module exists on file - that would be way more effective and prevent multiple stages!

            return new Output<bool, List<string>>() { Item1 = true, Item2 = installedPackages };
        }

        /*************************************/
    }
}
