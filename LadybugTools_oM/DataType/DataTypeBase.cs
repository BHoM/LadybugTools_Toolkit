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

using Newtonsoft.Json;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    public class DataTypeBase : ILadybugObject
    {
        [Description("Name")]
        [JsonProperty("name")]
        public virtual string Name { get; set; } = null;

        [Description("Units")]
        [JsonProperty("units")]
        public virtual string Units { get; set; } = null;

        [Description("SiUnits")]
        [JsonProperty("si_units")]
        public virtual string SiUnits { get; set; } = null;

        [Description("IpUnits")]
        [JsonProperty("ip_units")]
        public virtual string IpUnits { get; set; } = null;

        [Description("Min")]
        [JsonProperty("min")]
        public virtual string Min { get; set; } = null;

        [Description("Max")]
        [JsonProperty("max")]
        public virtual string Max { get; set; } = null;

        [Description("Abbreviation")]
        [JsonProperty("abbreviation")]
        public virtual string Abbreviation { get; set; } = null;

        [Description("UnitDescr")]
        [JsonProperty("unit_descr")]
        public virtual string UnitDescr { get; set; } = null;

        [Description("PointInTime")]
        [JsonProperty("point_in_time")]
        public virtual string PointInTime { get; set; } = null;

        [Description("Cumulative")]
        [JsonProperty("cumulative")]
        public virtual string Cumulative { get; set; } = null;

        [Description("NormalizedType")]
        [JsonProperty("normalized_type")]
        public virtual string NormalizedType { get; set; } = null;

        [Description("TimeAggregatedType")]
        [JsonProperty("time_aggregated_type")]
        public virtual string TimeAggregatedType { get; set; } = null;

        [Description("TimeAggregatedFactor")]
        [JsonProperty("time_aggregated_factor")]
        public virtual string TimeAggregatedFactor { get; set; } = null;
    }
}
