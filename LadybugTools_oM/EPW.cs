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

namespace BH.oM.LadybugTools
{
    public class EPW : BHoMObject
    {
        [JsonProperty("location")]
        [Description("The location.")]
        public virtual Location Location { get; set; } = new Location();

        [JsonProperty("data_collections")]
        [Description("The data_collections.")]
        public virtual List<DataCollection> DataCollections { get; set; } = null;        

        [JsonProperty("daylight_savings_start")]
        [Description("The daylight_savings_start.")]
        public virtual string DaylightSavingsStart { get; set; } = "";

        [JsonProperty("daylight_savings_end")]
        [Description("The daylight_savings_end.")]
        public virtual string DaylightSavingsEnd { get; set; } = "";

        [JsonProperty("comments_1")]
        [Description("The comments_1.")]
        public virtual string Comments1 { get; set; } = "";

        [JsonProperty("comments_2")]
        [Description("The comments_2.")]
        public virtual string Comments2 { get; set; } = "";

        [JsonProperty("type")]
        [Description("The type.")]
        public virtual string Type { get; set; } = "EPW";
    }
}

