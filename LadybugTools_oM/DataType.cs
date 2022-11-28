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
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.Remoting.Messaging;

namespace BH.oM.LadybugTools
{
    public class DataType : BHoMObject
    {
        [JsonProperty("type")]
        [Description("The type.")]
        public string Type { get; set; } = "DataType";

        [JsonProperty("name")]
        [Description("The name.")]
        public override string Name { get; set; } = "";

        [JsonProperty("data_type")]
        [Description("The data_type.")]
        public string _DataType { get; set; } = "";

        [JsonProperty("base_unit")]
        [Description("The base_unit.")]
        public string BaseUnit { get; set; } = "";

        [JsonProperty("min")]
        [Description("The min.")]
        public double Min { get; set; } = double.NegativeInfinity;

        [JsonProperty("max")]
        [Description("The max.")]
        public double Max { get; set; } = double.PositiveInfinity;

        [JsonProperty("abbreviation")]
        [Description("The abbreviation.")]
        public string Abbreviation { get; set; } = "";

        [JsonProperty("unit_descr")]
        [Description("The unit_descr.")]
        public string UnitDescr { get; set; } = "";

        [JsonProperty("point_in_time")]
        [Description("The point_in_time.")]
        public bool PointInTime { get; set; } = false;

        [JsonProperty("cumulative")]
        [Description("The cumulative.")]
        public bool Cumulative { get; set; } = false;
    }
}

