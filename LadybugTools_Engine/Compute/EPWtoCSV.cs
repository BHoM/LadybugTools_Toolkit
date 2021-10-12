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

using BH.oM.Reflection.Attributes;

using System.ComponentModel;
using System.Diagnostics;
using System.IO;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Convert an EPW file into a time-indexed CSV version.")]
        [Input("epwFile", "An EPW file.")]
        [Output("csvFile", "The resultant CSV file path.")]
        public static string EPWtoCSV(string epwFile)
        {
            string scriptPath = @"C:\ProgramData\BHoM\Extensions\LadybugTools\epw_to_csv.py";
            if (!Query.CheckVirtualEnvironmentInstalled())
            {
                return null;
            }

            // Run the Python code
            Process p = new Process();
            p.StartInfo.CreateNoWindow = true;
            p.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
            p.StartInfo.UseShellExecute = false;
            p.StartInfo.RedirectStandardOutput = true;
            p.StartInfo.FileName = "cmd.exe";
            p.StartInfo.Arguments = $"/c {Python.Query.VirtualEnvironmentExecutable(VIRTUALENV_NAME)} {scriptPath} \"{epwFile}\"";
            p.Start();

            // To avoid deadlocks, always read the output stream first and then wait.  
            string output = p.StandardOutput.ReadToEnd();
            p.WaitForExit();

            //return output;
            return output;
        }
    }
}