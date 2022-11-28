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


using BH.oM.Base;
using Newtonsoft.Json;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    public class Location : BHoMObject
    {
        [JsonProperty("city")]
        [Description("The city.")]
        public string City { get; set; } = "";

        [JsonProperty("state")]
        [Description("The state.")]
        public string State { get; set; } = "";

        [JsonProperty("country")]
        [Description("The country.")]
        public string Country { get; set; } = "";

        [JsonProperty("latitude")]
        [Description("The latitude.")]
        public double Latitude { get; set; } = 0;

        [JsonProperty("longitude")]
        [Description("The longitude.")]
        public double Longitude { get; set; } = 0;

        [JsonProperty("time_zone")]
        [Description("The time_zone.")]
        public double TimeZone { get; set; } = 0;

        [JsonProperty("elevation")]
        [Description("The elevation.")]
        public double Elevation { get; set; } = 0;

        [JsonProperty("station_id")]
        [Description("The station_id.")]
        public string StationId { get; set; } = "";

        [JsonProperty("source")]
        [Description("The source.")]
        public string Source { get; set; } = "";

        [JsonProperty("type")]
        [Description("The type.")]
        public string Type { get; set; } = "Location";
    }
}

