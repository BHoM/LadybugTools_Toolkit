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
using BH.oM.LadybugTools;
using System.ComponentModel;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Get the Location from a given EPW file.")]
        [Output("location", "A location object.")]
        public static Location Location(this string epw_file)
        {
            List<string> locationData = File.ReadLines(epw_file).First().Split(',').ToList();
            return new Location() {
                City = locationData[1],
                State = locationData[2],
                Country = locationData[3],
                Source = locationData[4],
                StationId = locationData[5],
                Latitude = System.Convert.ToDouble(locationData[6]),
                Longitude = System.Convert.ToDouble(locationData[7]),
                TimeZone = System.Convert.ToDouble(locationData[8]),
                Elevation = System.Convert.ToDouble(locationData[9]),
            };
        }
    }
}

