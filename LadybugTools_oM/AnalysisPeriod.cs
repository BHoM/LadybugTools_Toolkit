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
    public class AnalysisPeriod : ILadybugObject
    {
        [Description("Start month (1-12).")]
        [JsonProperty("st_month")]
        public virtual int StartMonth { get; set; } = 1;

        [Description("Start day (1-31).")]
        [JsonProperty("st_day")]
        public virtual int StartDay { get; set; } = 1;

        [Description("Start hour (0-23).")]
        [JsonProperty("st_hour")]
        public virtual int StartHour { get; set; } = 0;

        [Description("End month (1-12).")]
        [JsonProperty("end_month")]
        public virtual int EndMonth { get; set; } = 12;

        [Description("End day (1-31).")]
        [JsonProperty("end_day")]
        public virtual int EndDay { get; set; } = 31;

        [Description("End hour (0-23).")]
        [JsonProperty("end_hour")]
        public virtual int EndHour { get; set; } = 23;

        [Description("An integer number for the number of time steps per hours. Acceptable inputs include: [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60].")]
        [JsonProperty("timestep")]
        public virtual int Timestep { get; set; } = 1;

        [Description("A boolean to indicate whether the AnalysisPeriod represents a leap year.")]
        [JsonProperty("is_leap_year")]
        public virtual bool IsLeapYear { get; set; } = false;
    }
}
