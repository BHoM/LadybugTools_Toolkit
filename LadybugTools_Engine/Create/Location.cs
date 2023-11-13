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

using BH.oM.Base.Attributes;
using BH.oM.LadybugTools;
using System.Collections.Generic;
using System.ComponentModel;

namespace BH.Engine.LadybugTools
{
    public static partial class Create
    {
        [Description("Create a Location object.")]
        [Input("city", "The city of the location.")]
        [Input("state", "The state of the location.")]
        [Input("country", "The country of the location.")]
        [Input("latitude", "The latitude of the location.")]
        [Input("longitude", "The longitude of the location.")]
        [Input("timeZone", "The time zone of the location.")]
        [Input("elevation", "The elevation of the location.")]
        [Input("stationId", "The station ID of the location.")]
        [Input("source", "The source of the location.")]
        [Output("location", "A Location object.")]
        public static Location Location(
            string city = "-",
            string state = "-",
            string country = "-",
            double latitude = 0,
            double longitude = 0,
            double timeZone = 0,
            double elevation = 0,
            string stationId = "",
            string source = ""
        )
        {
            if (latitude < -90 || latitude > 90)
            {
                BH.Engine.Base.Compute.RecordError("latitude must be within -90 to 90");
                return null;
            }

            if (longitude < -180 || longitude > 180)
            {
                BH.Engine.Base.Compute.RecordError("longitude must be within -180 to 180");
                return null;
            }

            if (elevation < -1000 || elevation > 10000)
            {
                BH.Engine.Base.Compute.RecordError("elevation must be within -1000 to 10000");
                return null;
            }

            if (timeZone < -12 || timeZone > 14)
            {
                BH.Engine.Base.Compute.RecordError("timeZone must be within -12 to 14");
                return null;
            }


            return new Location()
            {
                City = city,
                State = state,
                Country = country,
                Latitude = latitude,
                Longitude = longitude,
                TimeZone = timeZone,
                Elevation = elevation,
                StationId = stationId,
                Source = source,
            };
        }
    }
}
