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
using System.Net;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Download a file from a URL to a directory.")]
        [Input("url", "A valid URL string to a file.")]
        [Input("targetFile", "The path to the target file.")]
        [Input("mkdir", "Set to True to create the directory to contain the target file if doesn't exist.")]
        [Output("targetFile", "The path to the downloaded file, if it has been downloaded successfully.")]
        public static string DownloadFileByName(string url, string targetFile, bool mkdir = true)
        {

            if (mkdir)
                PrepareDirectory(Path.GetDirectoryName(targetFile), false);

            using (WebClient client = new WebClient())
            {
                try
                {
                    client.DownloadFile(url, targetFile);
                }
                catch (System.Exception ex)
                {
                    BH.Engine.Base.Compute.RecordError($"{url} not downloaded due to {ex}");
                }
            }

            if (File.Exists(targetFile))
                return targetFile;
            else
                return string.Empty;
        }
    }
}

