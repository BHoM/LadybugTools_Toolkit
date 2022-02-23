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

using BH.oM.Base.Attributes;

using System.ComponentModel;
using System.IO;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Delete all the files inside a target directory.")]
        [Input("sourceDirectory", "The source directory.")]
        [Input("destinationDirectory", "The destination directory.")]
        [Input("overwrite", "Set to true to overwrite files in the target directory if they already exist.")]
        [Output("success", "True if the destination directory now contains the files from the source directory!")]
        public static bool CopyFileTree(string sourceDirectory, string destinationDirectory, bool overwrite = true)
        {
            PrepareDirectory(destinationDirectory, false);

            foreach (string directory in Directory.GetDirectories(sourceDirectory))
            {
                string dirName = Path.GetFileName(directory);
                if (!Directory.Exists(Path.Combine(destinationDirectory, dirName)))
                {
                    Directory.CreateDirectory(Path.Combine(destinationDirectory, dirName));
                }
                CopyFileTree(directory, Path.Combine(destinationDirectory, dirName));
            }

            foreach (var file in Directory.GetFiles(sourceDirectory))
            {
                File.Copy(file, Path.Combine(destinationDirectory, Path.GetFileName(file)), overwrite);
            }

            return true;
        }
    }
}

