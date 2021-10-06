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
    public class EPW : BHoMObject
    {
        [Description("Location")]
        public virtual Location Location { get; set; } = new Location();

        [Description("DataCollections")]
        public virtual List<HourlyContinuousCollection> DataCollections { get; set; } = new List<HourlyContinuousCollection>();

        [Description("Metadata")]
        public virtual Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();

        [Description("HeatingDict")]
        public virtual Dictionary<string, string> HeatingDict { get; set; } = new Dictionary<string, string>();

        [Description("CoolingDict")]
        public virtual Dictionary<string, string> CoolingDict { get; set; } = new Dictionary<string, string>();

        [Description("ExtremesDict")]
        public virtual Dictionary<string, string> ExtremesDict { get; set; } = new Dictionary<string, string>();

        [Description("ExtremeHotWeeks")]
        public virtual Dictionary<string, AnalysisPeriod> ExtremeHotWeeks { get; set; } = new Dictionary<string, AnalysisPeriod>();

        [Description("ExtremeColdWeeks")]
        public virtual Dictionary<string, AnalysisPeriod> ExtremeColdWeeks { get; set; } = new Dictionary<string, AnalysisPeriod>();

        [Description("TypicalWeeks")]
        public virtual Dictionary<string, AnalysisPeriod> TypicalWeeks { get; set; } = new Dictionary<string, AnalysisPeriod>();

        [Description("MonthlyGroundTemps")]
        public virtual Dictionary<string, MonthlyCollection> MonthlyGroundTemps { get; set; } = new Dictionary<string, MonthlyCollection>();

        [Description("IsIp")]
        public virtual bool IsIp { get; set; } = false;

        [Description("IsLeapYear")]
        public virtual bool IsLeapYear { get; set; } = false;

        [Description("DaylightSavingsStart")]
        public virtual string DaylightSavingsStart { get; set; } = "0";

        [Description("DaylightSavingsEnd")]
        public virtual string DaylightSavingsEnd { get; set; } = "0";

        [Description("Comments")]
        public virtual List<string> Comments { get; set; } = new List<string>();
    }
}
