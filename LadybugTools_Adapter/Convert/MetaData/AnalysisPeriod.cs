/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2023, the respective contributors. All rights reserved.
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
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.AnalysisPeriod ToAnalysisPeriod(Dictionary<string, object> oldObject)
        {
            try
            {
                return new oM.LadybugTools.AnalysisPeriod()
                {
                    StartMonth = (int)oldObject["st_month"],
                    StartDay = (int)oldObject["st_day"],
                    StartHour = (int)oldObject["st_hour"],
                    EndMonth = (int)oldObject["end_month"],
                    EndDay = (int)oldObject["end_day"],
                    EndHour = (int)oldObject["end_hour"],
                    IsLeapYear = (bool)oldObject["is_leap_year"],
                    TimeStep = (int)oldObject["timestep"]
                };
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error ocurred during conversion of a {typeof(AnalysisPeriod).FullName}. Returning a default {typeof(AnalysisPeriod).FullName}:\n The error: {ex}");
                return new AnalysisPeriod();
            }
        }

        public static Dictionary<string, object> FromAnalysisPeriod(BH.oM.LadybugTools.AnalysisPeriod analysisPeriod)
        {
            return new Dictionary<string, object>
            {
                { "type", "AnalysisPeriod" },
                { "st_month", analysisPeriod.StartMonth },
                { "st_day", analysisPeriod.StartDay },
                { "st_hour", analysisPeriod.StartHour },
                { "end_month", analysisPeriod.EndMonth },
                { "end_day", analysisPeriod.EndDay },
                { "end_hour", analysisPeriod.EndHour },
                { "is_leap_year", analysisPeriod.IsLeapYear },
                { "timestep", analysisPeriod.TimeStep }
            };
        }
    }
}
