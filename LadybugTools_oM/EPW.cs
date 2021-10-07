///*
// * This file is part of the Buildings and Habitats object Model (BHoM)
// * Copyright (c) 2015 - 2021, the respective contributors. All rights reserved.
// *
// * Each contributor holds copyright over their respective contributions.
// * The project versioning (Git) records all such contribution source information.
// *                                           
// *                                                                              
// * The BHoM is free software: you can redistribute it and/or modify         
// * it under the terms of the GNU Lesser General Public License as published by  
// * the Free Software Foundation, either version 3.0 of the License, or          
// * (at your option) any later version.                                          
// *                                                                              
// * The BHoM is distributed in the hope that it will be useful,              
// * but WITHOUT ANY WARRANTY; without even the implied warranty of               
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                 
// * GNU Lesser General Public License for more details.                          
// *                                                                            
// * You should have received a copy of the GNU Lesser General Public License     
// * along with this code. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>.      
// */

//using BH.oM.Base;
//using BH.oM.Geometry;
//using LadybugTools_oM.Enums;
//using Newtonsoft.Json;
//using System.Collections.Generic;
//using System.ComponentModel;

//namespace BH.oM.LadybugTools
//{
//    public class EPW : ILadybugObject
//    {
//        [Description("A Location object.")]
//        [JsonProperty("location")]
//        public virtual Location Location { get; set; } = new Location();

//        [Description("DataCollections")]
//        [JsonProperty("data_collections")]
//        public virtual List<HourlyContinuousCollection> DataCollections { get; set; } = new List<HourlyContinuousCollection>();

//        [Description("Dictionary of metadata written to DataCollection headers.\n\nKeys typically include 'source', 'country', and 'city').")]
//        [JsonProperty("metadata")]
//        public virtual Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();

//        [Description("Dictionary with ASHRAE HOF Climate Design Data for heating conditions.")]
//        [JsonProperty("heating_dict")]
//        public virtual Dictionary<string, string> HeatingDict { get; set; } = new Dictionary<string, string>();

//        [Description("Dictionary with ASHRAE HOF Climate Design Data for cooling conditions.")]
//        [JsonProperty("cooling_dict")]
//        public virtual Dictionary<string, string> CoolingDict { get; set; } = new Dictionary<string, string>();

//        [Description("Dictionary with ASHRAE HOF Climate Design Data for extreme conditions.")]
//        [JsonProperty("extremes_dict")]
//        public virtual Dictionary<string, string> ExtremesDict { get; set; } = new Dictionary<string, string>();

//        [Description("A dictionary with AnalysisPeriods for the hottest weeks within the EPW.")]
//        [JsonProperty("extreme_hot_weeks")]
//        public virtual Dictionary<string, AnalysisPeriod> ExtremeHotWeeks { get; set; } = new Dictionary<string, AnalysisPeriod>();

//        [Description("A dictionary with AnalysisPeriods for the coldest weeks within the EPW.")]
//        [JsonProperty("extreme_cold_weeks")]
//        public virtual Dictionary<string, AnalysisPeriod> ExtremeColdWeeks { get; set; } = new Dictionary<string, AnalysisPeriod>();

//        [Description("A dictionary with AnalysisPeriods for the typical weeks within the EPW.")]
//        [JsonProperty("typical_weeks")]
//        public virtual Dictionary<string, AnalysisPeriod> TypicalWeeks { get; set; } = new Dictionary<string, AnalysisPeriod>();

//        [Description("A dictionary of Monthly Data collections.\n\nThe keys of this dictionary are the depths at which each set of temperatures occurs.")]
//        [JsonProperty("monthly_ground_temps")]
//        public virtual Dictionary<string, MonthlyCollection> MonthlyGroundTemps { get; set; } = new Dictionary<string, MonthlyCollection>();

//        [Description("Boolean to denote whether the EPW is in IP units or not.")]
//        [JsonProperty("is_ip")]
//        public virtual bool IsIp { get; set; } = false;

//        [Description("Boolean to denote whether the EPW is a leap year or not.")]
//        [JsonProperty("is_leap_year")]
//        public virtual bool IsLeapYear { get; set; } = false;

//        [Description("Signify when daylight savings starts (hour integer in year).")]
//        [JsonProperty("daylight_savings_start")]
//        public virtual string DaylightSavingsStart { get; set; } = "0";

//        [Description("Signify when daylight savings ends (hour integer in year).")]
//        [JsonProperty("daylight_savings_end")]
//        public virtual string DaylightSavingsEnd { get; set; } = "0";

//        [Description("Comments associated with this EPW.")]
//        [JsonProperty("comments_1")]
//        public virtual string Comments1 { get; set; } = "";

//        [Description("Comments associated with this EPW.")]
//        [JsonProperty("comments_2")]
//        public virtual string Comments2 { get; set; } = "";
//    }
//}
