/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2021, the respective contributors. All rights reserved.
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
using BH.oM.Geometry;
using LadybugTools_oM.Enums;
using System.Collections.Generic;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    public class Location : BHoMObject
    {
        [Description("City: Name of the city as a string.")]
        public virtual string City { get; set; } = "";

        [Description("State: Optional state in which the city is located.")]
        public virtual string State { get; set; } = "";

        [Description("Country: Name of the country as a string.")]
        public virtual string Country { get; set; } = "";

        [Description("Latitude: Location latitude between -90 and 90 (Default: 0).")]
        public virtual double Latitude { get; set; } = 0;

        [Description("Longitude: Location longitude between -180 (west) and 180 (east) (Default: 0).")]
        public virtual double Longitude { get; set; } = 0;

        [Description("TimeZone: Time zone between -12 hours (west) and +14 hours (east). If None, the time zone will be an estimated integer value derived from the longitude in accordance with solar time (Default: None).")]
        public virtual int TimeZone { get; set; } = 0;

        [Description("Elevation: A number for elevation of the location in meters. (Default: 0).")]
        public virtual double Elevation { get; set; } = 0;

        [Description("StationID: ID of the location if the location is representing a weather station.")]
        public virtual string StationID { get; set; } = "";

        [Description("Source: Source of data (e.g. TMY, TMY3).")]
        public virtual string Source { get; set; } = "";
    }
}
