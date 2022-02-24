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
        [Input("targetDirectory", "The target directory.")]
        [Input("removeDirectory", "Set to true to also remove the target directory.")]
        public static void NukeDirectory(string targetDirectory, bool removeDirectory = false)
        {
            // TODO - Should probably put some protections in here to stop users from nuking system32!

            DirectoryInfo di = new DirectoryInfo(targetDirectory);

            foreach (FileInfo file in di.EnumerateFiles())
            {
                try
                {
                    file.Delete();
                }
                catch (System.Exception ex)
                {
                    BH.Engine.Base.Compute.RecordError($"{file.FullName} not deleted due to {ex}");
                }
                
            }

            foreach (DirectoryInfo dir in di.EnumerateDirectories())
            {
                try
                {
                    dir.Delete(true);
                }
                catch (System.Exception ex)
                {
                    BH.Engine.Base.Compute.RecordError($"{dir.FullName} not deleted due to {ex}");
                }
                dir.Delete(true);
            }

            if (removeDirectory)
                try
                {
                    di.Delete();
                }
                catch (System.Exception ex)
                {
                    BH.Engine.Base.Compute.RecordError($"{di.FullName} not deleted due to {ex}");
                }
                
        }
    }
}

