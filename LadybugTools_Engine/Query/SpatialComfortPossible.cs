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
using System.Collections.Generic;
using System.IO;
using System.ComponentModel;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Determine whether a spatial comfort post-process is possible.")]
        [Output("bool", "True if spatial comfort post-process is possible.")]
        public static bool SpatialComfortPossible(string directory)
        {
            if (!ExternalComfortPossible())
                return false;

            string annualIrradiancePath = System.IO.Path.Combine(directory, "annual_irradiance");
            string skyViewPath = System.IO.Path.Combine(directory, "sky_view");

            foreach (string path in new List<string>() {
                annualIrradiancePath,
                skyViewPath,
            })
            {
                if (!Directory.Exists(path))
                {
                    BH.Engine.Base.Compute.RecordError($"You must run a {new DirectoryInfo(path).Name} simulation in order to make Spatial Comfort post-processing possible.");
                    return false;
                }
            }

            return true;
        }
    }
}

