/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2024, the respective contributors. All rights reserved.
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

using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.Location ToLocation(Dictionary<string, object> oldObject)
        {
            string city = "";
            string state = "";
            string country = "";
            double latitude = 0.0;
            double longitude = 0.0;
            double timeZone = 0.0;
            double elevation = 0.0;
            string stationID = "";
            string source = "BHoM LadybugTools_Toolkit default";

            try
            {
                city = (string)oldObject["city"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the city name of the Location. returning city as default (\"\").\n The error: {ex}");
            }

            try
            {
                state = (string)oldObject["state"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the state name of the Location. returning state as default (\"\").\n The error: {ex}");
            }

            try
            {
                country = (string)oldObject["country"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the country name of the Location. returning country as default (\"\").\n The error: {ex}");
            }

            try
            {
                latitude = (double)oldObject["latitude"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the latitude of the Location. returning latitude as default ({latitude}).\n The error: {ex}");
            }

            try
            {
                longitude = (double)oldObject["longitude"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the longitude of the Location. returning longitude as default ({longitude}).\n The error: {ex}");
            }

            try
            {
                timeZone = (double)oldObject["time_zone"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the time zone of the Location. returning time zone as default ({timeZone}).\n The error: {ex}");
            }

            try
            {
                elevation = (double)oldObject["elevation"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the elevation of the Location. returning elevation as default ({elevation}).\n The error: {ex}");
            }

            try
            {
                stationID = (string)oldObject["station_id"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the station ID of the Location. returning station ID as default (\"\").\n The error: {ex}");
            }

            try
            {
                source = (string)oldObject["source"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the source of the Location. returning source as default ({source}).\n The error: {ex}");
            }

            return new oM.LadybugTools.Location()
            {
                City = city,
                State = state,
                Country = country,
                Latitude = latitude,
                Longitude = longitude,
                TimeZone = timeZone,
                Elevation = elevation,
                StationId = stationID,
                Source = source
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

