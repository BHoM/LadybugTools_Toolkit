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
using System.ComponentModel;
using Newtonsoft.Json;

namespace BH.oM.LadybugTools
{
    public class AnalysisPeriod : BHoMObject
    {
        [JsonProperty("st_month")]
        [Description("The start month.")]
        public int StMonth { get; set; } = 1;

        [JsonProperty("st_day")]
        [Description("The start day.")]
        public int StDay { get; set; } = 1;

        [JsonProperty("st_hour")]
        [Description("The start hour.")]
        public int StHour { get; set; } = 0;

        [JsonProperty("end_month")]
        [Description("The end month.")]
        public int EndMonth { get; set; } = 12;

        [JsonProperty("end_day")]
        [Description("The end day.")]
        public int EndDay { get; set; } = 31;

        [JsonProperty("end_hour")]
        [Description("The end hour.")]
        public int EndHour { get; set; } = 23;

        [JsonProperty("timestep")]
        [Description("The end timestep.")]
        public int Timestep { get; set; } = 1;

        [JsonProperty("is_leap_year")]
        [Description("The year leaped-ness.")]
        public bool IsLeapYear { get; set; } = false;

        [JsonProperty("type")]
        [Description("The LBT type of this object.")]
        public string Type { get; set; } = "AnalysisPeriod";
    }
}

