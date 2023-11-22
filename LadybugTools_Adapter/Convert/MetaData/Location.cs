/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2023, the respective contributors. All rights reserved.
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

using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.Location ToLocation(Dictionary<string, object> oldObject)
        {
            return new oM.LadybugTools.Location()
            {
                City = (string)oldObject["city"],
                State = (string)oldObject["state"],
                Country = (string)oldObject["country"],
                Latitude = (double)oldObject["latitude"],
                Longitude = (double)oldObject["longitude"],
                TimeZone = (double)oldObject["time_zone"],
                Elevation = (double)oldObject["elevation"],
                StationId = (string)oldObject["station_id"],
                Source = (string)oldObject["source"]
            };
        }

        public static Dictionary<string, object> FromLocation(BH.oM.LadybugTools.Location location)
        {
            return new Dictionary<string, object>()
            {
                { "type", "Location" },
                { "city", location.City },
                { "state", location.State },
                { "country", location.Country },
                { "latitude", location.Latitude },
                { "longitude", location.Longitude },
                { "time_zone", location.TimeZone },
                { "elevation", location.Elevation },
                { "station_id", location.StationId },
                { "source", location.Source }
            };
        }
    }
}
