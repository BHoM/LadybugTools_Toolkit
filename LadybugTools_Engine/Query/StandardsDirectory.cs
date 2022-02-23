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
    public static partial class Query
    {
        [Description("Get the directory where standards distributed with Ladybug Tools are installed.")]
        [Output("gemDirectory", "The directory where standards are/should be.")]
        public static string StandardsDirectory()
        {
            string userDirectory = System.Environment.GetFolderPath(System.Environment.SpecialFolder.UserProfile);
            string standardsDirectory = Path.Combine(userDirectory, "ladybug_tools", "resources", "standards");
            Compute.PrepareDirectory(standardsDirectory, false);
            return standardsDirectory;
        }
    }
}

